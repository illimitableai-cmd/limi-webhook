// api/webhook.js
export const config = { api: { bodyParser: false } };

import OpenAI from "openai";
import twilio from "twilio";
import { createClient } from "@supabase/supabase-js";

/** --- ENV ------------------------------------------------------------ */
const openai = new OpenAI({ apiKey: process.env.OPENAI_KEY });
const twilioClient = twilio(process.env.TWILIO_SID, process.env.TWILIO_AUTH);
const supabase =
  process.env.SUPABASE_URL && process.env.SUPABASE_KEY
    ? createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY)
    : null;

// 🔒 GPT-5 only (no model fallback)
const CHAT_MODEL        = process.env.OPENAI_CHAT_MODEL || "gpt-5";

// Token budgets for Responses API (controls visible output only)
const CHAT_MAX_TOKENS   = Number(process.env.CHAT_MAX_TOKENS   || 180);
const CHAT_RETRY_TOKENS = Number(process.env.CHAT_RETRY_TOKENS || 240);

const LLM_TIMEOUT_MS    = Number(process.env.LLM_TIMEOUT_MS || 11000);
const WATCHDOG_MS       = Number(process.env.WATCHDOG_MS || 12500);

const MAX_SMS_CHARS     = Number(process.env.SMS_MAX_CHARS || 320);
const TWILIO_WA_FROM    = (process.env.TWILIO_WHATSAPP_FROM || "").trim();

// 🔧 Output shaping (tweak by env)
const FINAL_MAX_CHARS     = Number(process.env.FINAL_MAX_CHARS || 240);
const FINAL_MAX_SENTENCES = Number(process.env.FINAL_MAX_SENTENCES || 2);

/** --- Debug logging -------------------------------------------------- */
async function dbg(step, payload) {
  try { console.log("[dbg]", step, payload); } catch {}
  try { if (supabase) await supabase.from("debug_logs").insert([{ step, payload }]); }
  catch (e) { console.log("[dbg-fail]", step, String(e?.message || e)); }
}

/** --- Utils ---------------------------------------------------------- */
function toXml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}
function readRawBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", c => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}
function smsNumber(s = "") {
  let v = String(s).replace(/^whatsapp:/i, "").trim();
  if (v.startsWith("00")) v = "+" + v.slice(2);
  if (v.startsWith("0")) v = "+44" + v.slice(1);
  return v;
}
const safeSlice = (obj, n = 300) => {
  try { return JSON.stringify(obj).slice(0, n); } catch { return String(obj).slice(0, n); }
};
const extractFinal = (s = "") => {
  const m = String(s).match(/<final>([\s\S]*?)<\/final>/i);
  return m ? m[1].trim() : "";
};

// Robustly pull text out of a Responses API object
function deepFindText(resp) {
  const out = [];
  const visit = (node) => {
    if (!node || typeof node !== "object") return;
    if (typeof node.output_text === "string") out.push(node.output_text);
    if (typeof node.text === "string") out.push(node.text);
    if (node.text && typeof node.text === "object" && typeof node.text.value === "string") out.push(node.text.value);
    if (Array.isArray(node)) { for (const it of node) visit(it); return; }
    if (Array.isArray(node.content)) for (const it of node.content) visit(it);
    for (const k of Object.keys(node)) { const v=node[k]; if (v && typeof v === "object") visit(v); }
  };
  if (Array.isArray(resp?.output)) visit(resp.output); else visit(resp);
  return out.join("").trim();
}

/** --- Finalization helpers ------------------------------------------ */
function trimSentences(text, maxSentences = FINAL_MAX_SENTENCES) {
  const parts = String(text || "").trim().split(/(?<=[.!?])\s+/);
  return parts.slice(0, Math.max(1, maxSentences)).join(" ").trim();
}
function truncateChars(s, limit = FINAL_MAX_CHARS) {
  const txt = (s || "").trim();
  if (txt.length <= limit) return txt;
  return txt.slice(0, limit - 1) + "…";
}
function coerceFinal(raw) {
  const got = extractFinal(raw || "");
  if (got) return `<final>${truncateChars(trimSentences(got))}</final>`;
  const asText = String(raw || "").replace(/\s+/g, " ").trim();
  if (!asText) return "<final>I’m not sure.</final>";
  const concise = truncateChars(trimSentences(asText));
  return `<final>${concise}</final>`;
}

/** --- Timed wrapper -------------------------------------------------- */
function withTimeout(promise, label, ctx = {}) {
  dbg(label + "_request", ctx);
  return new Promise((resolve) => {
    const t = setTimeout(() => resolve({ __timeout: true }), LLM_TIMEOUT_MS);
    promise.then((r) => { clearTimeout(t); resolve(r); })
           .catch((e) => { clearTimeout(t); resolve({ __error: e }); });
  });
}

/** --- OpenAI: Responses API (primary for GPT-5) --------------------- */
// Keep it minimal: only model, input, max_output_tokens (others caused 400s in your logs)
async function safeResponsesCompletion({ messages, maxTokens }) {
  const req = { model: CHAT_MODEL, input: messages, max_output_tokens: maxTokens };
  await dbg("openai_responses_request", { model: CHAT_MODEL, maxTokens, input_preview: safeSlice(messages) });

  const r = await withTimeout(openai.responses.create(req), "openai_responses", { model: CHAT_MODEL, maxTokens });

  if (r?.__timeout) { await dbg("openai_responses_error", { message: "timeout" }); return ""; }
  if (r?.__error)   { await dbg("openai_responses_error", { message: String(r.__error?.message || r.__error) }); return ""; }

  const text = deepFindText(r);
  await dbg("openai_responses_reply", { has_text: !!text, output_len: text.length, usage: r?.usage });
  return (text || "").trim();
}

/** --- OpenAI: Chat Completions (fallback, still GPT-5) --------------- */
// Only used if Responses returns empty; keeps GPT-5 and uses max_completion_tokens.
async function safeChatCompletion({ messages, maxTokens }) {
  const req = { model: CHAT_MODEL, messages, max_completion_tokens: maxTokens };
  await dbg("openai_chat_request", { model: CHAT_MODEL, maxTokens, input_preview: safeSlice(messages) });

  const r = await withTimeout(openai.chat.completions.create(req), "openai_chat", { model: CHAT_MODEL, maxTokens });

  if (r?.__timeout) { await dbg("openai_chat_error", { message: "timeout" }); return ""; }
  if (r?.__error)   { await dbg("openai_chat_error", { message: String(r.__error?.message || r.__error) }); return ""; }

  const txt = r?.choices?.[0]?.message?.content ?? "";
  await dbg("openai_chat_reply", { has_content: !!txt, usage: r?.usage, finish_reason: r?.choices?.[0]?.finish_reason });
  return (txt || "").trim();
}

/** --- LLM reply helpers --------------------------------------------- */
function systemInstruction() {
  return [
    "You are Limi’s SMS/WhatsApp brain.",
    `Return the final answer wrapped as <final>...</final>, up to ${FINAL_MAX_SENTENCES} short sentence(s), max ${FINAL_MAX_CHARS} characters inside the tags.`,
    "No preamble, no code blocks, no extra lines.",
    "If uncertain, reply: <final>I’m not sure.</final>"
  ].join("\n");
}

async function llmReply(userMsg) {
  // Attempt 1: Responses API
  const msgsA = [
    { role: "system", content: systemInstruction() },
    { role: "user",   content: userMsg }
  ];
  let txt = await safeResponsesCompletion({ messages: msgsA, maxTokens: CHAT_MAX_TOKENS });

  // Attempt 2: Responses API retry with simpler instruction + larger cap
  if (!extractFinal(txt || "")) {
    const msgsB = [
      { role: "system", content: `Return ONLY: <final>…</final>. Max ${FINAL_MAX_SENTENCES} sentences, ${FINAL_MAX_CHARS} chars.` },
      { role: "user",   content: userMsg }
    ];
    const retry = await safeResponsesCompletion({ messages: msgsB, maxTokens: CHAT_RETRY_TOKENS });
    if (retry) txt = retry;
  }

  // Fallback: Chat Completions (still GPT-5)
  if (!extractFinal(txt || "")) {
    const msgsC = [
      { role: "system", content: systemInstruction() },
      { role: "user",   content: userMsg }
    ];
    const alt = await safeChatCompletion({ messages: msgsC, maxTokens: CHAT_MAX_TOKENS });
    if (alt) txt = alt;
  }

  return coerceFinal(txt || "");
}

/** --- Twilio WhatsApp helper ---------------------------------------- */
async function sendWhatsApp(to, body) {
  if (!TWILIO_WA_FROM) throw new Error("Missing TWILIO_WHATSAPP_FROM");
  const waTo = to.startsWith("whatsapp:") ? to : `whatsapp:${to}`;
  const r = await twilioClient.messages.create({ from: TWILIO_WA_FROM, to: waTo, body });
  await dbg("twilio_send_wa_ok", { sid: r.sid, to: waTo });
  return r;
}

/** --- Handler -------------------------------------------------------- */
export default async function handler(req, res) {
  try {
    await dbg("handler_enter", { ts: Date.now(), has_openai_key: !!process.env.OPENAI_KEY });
    if (req.method !== "POST") { res.status(405).send("Method Not Allowed"); return; }

    const raw = await readRawBody(req);
    let Body = null, From = null, NumMedia = 0;

    if (raw.trim().startsWith("{")) {
      const j = JSON.parse(raw);
      Body = j.Body ?? j.body ?? "";
      From = j.From ?? j.from ?? "";
      if ((j.channel || "").toLowerCase() === "whatsapp" && From && !/^whatsapp:/i.test(From)) From = `whatsapp:${From}`;
      NumMedia = Number(j.NumMedia || j.numMedia || 0);
    } else {
      const p = new URLSearchParams(raw);
      Body = p.get("Body");
      From = p.get("From");
      NumMedia = Number(p.get("NumMedia") || 0);
    }

    Body = (Body || "").trim();
    From = (From || "").trim();

    const isWhatsApp = /^whatsapp:/i.test(From);
    const cleanFrom = smsNumber(From);

    await dbg("webhook_in", { from: cleanFrom, channel: isWhatsApp ? "whatsapp" : "sms", body_len: Body.length, numMedia: NumMedia });

    if (!Body) {
      res.setHeader("Content-Type","text/xml");
      res.status(200).send(`<Response><Message>Empty message. Please send text.</Message></Response>`);
      return;
    }

    // UK SMS photo guard
    if (!isWhatsApp && NumMedia > 0) {
      res.setHeader("Content-Type","text/xml");
      res.status(200).send(`<Response><Message>Pics don’t work over UK SMS. WhatsApp this same number instead 👍</Message></Response>`);
      return;
    }

    // Build the reply with a watchdog (no model fallback)
    const replyPromise = llmReply(Body);
    const watchdog = new Promise((resolve) =>
      setTimeout(() => { dbg("watchdog_fired", { ms: WATCHDOG_MS }); resolve(null); }, WATCHDOG_MS)
    );
    const final = await Promise.race([replyPromise, watchdog]);

    const reply = final || "<final>Sorry—service is a bit slow right now. Please try again.</final>";
    const safe = reply.length > MAX_SMS_CHARS ? reply.slice(0, MAX_SMS_CHARS - 1) + "…" : reply;

    if (isWhatsApp && TWILIO_WA_FROM) {
      try {
        // WhatsApp body without the <final> tags
        const waBody = safe.replace(/^<final>|<\/final>$/g, "");
        await sendWhatsApp(cleanFrom, waBody);
        res.setHeader("Content-Type","text/xml");
        res.status(200).send("<Response/>");
        return;
      } catch (e) {
        await dbg("twilio_send_wa_error", { to: cleanFrom, message: String(e?.message || e) });
      }
    }

    // SMS reply (TwiML)
    res.setHeader("Content-Type","text/xml");
    res.status(200).send(`<Response><Message>${toXml(safe)}</Message></Response>`);
    await dbg("twiml_sent", { to: cleanFrom, len: safe.length, mode: "sms_twiML" });

  } catch (e) {
    const msg = String(e?.message || e);
    await dbg("handler_error", { message: msg });
    res.setHeader("Content-Type","text/xml");
    res.status(200).send(`<Response><Message>Unhandled error: ${toXml(msg)}</Message></Response>`);
  }
}

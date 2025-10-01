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

// 🔒 GPT-5 only (no fallback)
const CHAT_MODEL        = process.env.OPENAI_CHAT_MODEL || "gpt-5";
const CHAT_MAX_TOKENS   = Number(process.env.CHAT_MAX_TOKENS || 120);  // modest, we return a short answer
const CHAT_RETRY_TOKENS = Number(process.env.CHAT_RETRY_TOKENS || 180);

const LLM_TIMEOUT_MS    = Number(process.env.LLM_TIMEOUT_MS || 11000);
const WATCHDOG_MS       = Number(process.env.WATCHDOG_MS || 12500);

const MAX_SMS_CHARS     = Number(process.env.SMS_MAX_CHARS || 320);
const TWILIO_WA_FROM    = (process.env.TWILIO_WHATSAPP_FROM || "").trim();

// 🔧 Output shaping (tweak without code changes)
const FINAL_MAX_CHARS       = Number(process.env.FINAL_MAX_CHARS || 240); // keep room for SMS
const FINAL_MAX_SENTENCES   = Number(process.env.FINAL_MAX_SENTENCES || 2);

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
const safeSlice = (obj, n = 220) => {
  try { return JSON.stringify(obj).slice(0, n); } catch { return String(obj).slice(0, n); }
};
const extractFinal = (s = "") => {
  const m = String(s).match(/<final>([\s\S]*?)<\/final>/i);
  return m ? m[1].trim() : "";
};

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
  // 1) If already wrapped, extract > trim sentences > truncate
  const got = extractFinal(raw || "");
  if (got) return `<final>${truncateChars(trimSentences(got))}</final>`;

  // 2) Otherwise, make a concise snippet and wrap
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

/** --- OpenAI: GPT-5 chat.completions (no temperature) ---------------- */
async function safeChatCompletion({ messages, model = CHAT_MODEL, maxTokens = CHAT_MAX_TOKENS }) {
  const isGpt5 = /^gpt-5/i.test(model);
  const req = { model, messages, stop: ["</final>"] }; // encourage early stop at closing tag
  if (isGpt5) req.max_completion_tokens = maxTokens;
  else { req.max_tokens = maxTokens; req.temperature = 1; }

  await dbg("openai_chat_request", { model, maxTokens, input_preview: safeSlice(messages) });
  const r = await withTimeout(openai.chat.completions.create(req), "openai_chat", { model, maxTokens });

  if (r?.__timeout) { await dbg("openai_chat_error", { message: "timeout" }); return null; }
  if (r?.__error)   { await dbg("openai_chat_error", { message: String(r.__error?.message || r.__error) }); return null; }

  const txt = r?.choices?.[0]?.message?.content ?? "";
  await dbg("openai_chat_reply", { has_content: !!txt, usage: r?.usage, finish_reason: r?.choices?.[0]?.finish_reason });
  return (txt || "").trim();
}

/** --- LLM reply helpers --------------------------------------------- */
function systemInstruction() {
  return [
    "You must respond wrapped in <final> and </final>.",
    `Goal: a clear, helpful answer in up to ${FINAL_MAX_SENTENCES} short sentence(s), max ${FINAL_MAX_CHARS} characters total inside the tags.`,
    "Format example: <final>Answer here.</final>",
    "Hard rules:",
    "- Start with '<final>' and end with '</final>'.",
    `- No more than ${FINAL_MAX_SENTENCES} sentence(s).`,
    `- No more than ${FINAL_MAX_CHARS} characters inside the tags.`,
    "- No preamble, no code blocks, no extra lines.",
    "- If uncertain, write: <final>I’m not sure.</final>"
  ].join("\n");
}

async function llmReply(userMsg) {
  // Attempt 1: strict instruction with modest cap
  const messagesA = [
    { role: "system", content: systemInstruction() },
    { role: "user",   content: userMsg }
  ];
  let txt = await safeChatCompletion({ messages: messagesA, maxTokens: CHAT_MAX_TOKENS });

  // Attempt 2: lighter instruction + slightly larger cap
  if (!extractFinal(txt || "")) {
    const messagesB = [
      { role: "system", content: "Return a concise answer wrapped as <final>...</final> (max two short sentences, ~240 chars)." },
      { role: "user",   content: userMsg }
    ];
    const retry = await safeChatCompletion({ messages: messagesB, maxTokens: CHAT_RETRY_TOKENS });
    if (retry) txt = retry;
  }

  // Guarantee valid wrapper
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
        await sendWhatsApp(cleanFrom, safe.replace(/^<final>|<\/final>$/g, "")); // WA doesn't need tags
        res.setHeader("Content-Type","text/xml");
        res.status(200).send("<Response/>");
        return;
      } catch (e) {
        await dbg("twilio_send_wa_error", { to: cleanFrom, message: String(e?.message || e) });
      }
    }

    // SMS reply (TwiML) – keep tags; your client extracts or just shows nicely
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

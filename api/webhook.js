// api/webhook.js
export const config = { api: { bodyParser: false } };
export const runtime = "nodejs";           // ensure Node runtime on Vercel
export const dynamic = "force-dynamic";    // avoid caching of the function

import OpenAI from "openai";
import twilio from "twilio";
import { createClient } from "@supabase/supabase-js";

/** --- ENV ---
 * Required (for full functionality): OPENAI_KEY, TWILIO_SID, TWILIO_AUTH, TWILIO_FROM
 * Optional: TWILIO_WHATSAPP_FROM (format: whatsapp:+447...)
 * Optional: OPENAI_CHAT_MODEL (default: gpt-5), LLM_TIMEOUT_MS, SMS_MAX_CHARS, CHAT_MAX_TOKENS, CHAT_RETRY_TOKENS
 * For debug logs: SUPABASE_URL, SUPABASE_KEY and a table `debug_logs`
 */

// -------- Lazy factories (avoid top-level throws) --------
let _openai = null, _supabase = null, _twilio = null;

function getOpenAI() {
  if (_openai) return _openai;
  const key = process.env.OPENAI_KEY || "";
  _openai = new OpenAI({ apiKey: key });
  return _openai;
}

function getSupabase() {
  if (_supabase !== null) return _supabase;
  if (process.env.SUPABASE_URL && process.env.SUPABASE_KEY) {
    _supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);
  } else {
    _supabase = null;
  }
  return _supabase;
}

function getTwilio() {
  if (_twilio) return _twilio;
  const sid  = (process.env.TWILIO_SID  || "").trim();
  const auth = (process.env.TWILIO_AUTH || "").trim();
  if (!sid || !auth) return null; // don’t crash cold start
  _twilio = twilio(sid, auth);
  return _twilio;
}

// -------- Config / tuning --------
const CHAT_MODEL = process.env.OPENAI_CHAT_MODEL || "gpt-5";
const LLM_TIMEOUT_MS = Number(process.env.LLM_TIMEOUT_MS || 8000);
const TWILIO_FROM = (process.env.TWILIO_FROM || "").trim();
const TWILIO_WA_FROM = (process.env.TWILIO_WHATSAPP_FROM || "").trim();
const MAX_SMS_CHARS = Number(process.env.SMS_MAX_CHARS || 320);
const MAX_COMPLETION_TOKENS = Number(process.env.CHAT_MAX_TOKENS || 160);
const CHAT_RETRY_TOKENS = Number(process.env.CHAT_RETRY_TOKENS || 800);

// ---------------- Debug logging ----------------
async function dbg(step, payload) {
  try {
    const supabase = getSupabase();
    if (!supabase) { console.log("[dbg]", step, payload); return; }
    await supabase.from("debug_logs").insert([{ step, payload }]);
  } catch (e) {
    console.log("[dbg-fail]", step, String(e?.message || e));
  }
}

// ---------------- Small utils ----------------
function toXml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
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
function safeSlice(obj, n = 1000) {
  try { return JSON.stringify(obj).slice(0, n); } catch { return String(obj).slice(0, n); }
}
function deepFindJson(resp) {
  let found = null;
  const visit = (node) => {
    if (!node || typeof node !== "object" || found) return;
    if (node.json && typeof node.json === "object") { found = node.json; return; }
    if (typeof node.type === "string" && /json/i.test(node.type)) {
      if (node.json && typeof node.json === "object") { found = node.json; return; }
      if (typeof node.text === "string") { try { const j = JSON.parse(node.text); if (j && typeof j === "object") { found = j; return; } } catch {}
    }
    if (typeof node.text === "string" && node.text.trim().startsWith("{")) {
      try { const j = JSON.parse(node.text); if (j && typeof j === "object") { found = j; return; } } catch {}
    }
    if (Array.isArray(node)) { for (const it of node) visit(it); return; }
    for (const k of Object.keys(node)) visit(node[k]);
  };
  if (resp?.output) visit(resp.output); else visit(resp);
  return found;
}
function deepFindText(resp) {
  const out = [];
  const visit = (node) => {
    if (!node || typeof node !== "object") return;
    if (typeof node.output_text === "string" && node.output_text) out.push(node.output_text);
    if (typeof node.text === "string" && node.text) out.push(node.text);
    if (node.text && typeof node.text === "object" && typeof node.text.value === "string") out.push(node.text.value);
    if (node.json && typeof node.json === "object" && typeof node.json.final === "string") out.push(node.json.final);
    if (Array.isArray(node)) { for (const it of node) visit(it); return; }
    if (Array.isArray(node.content)) { for (const it of node.content) visit(it); }
    if (node.assistant_response && typeof node.assistant_response === "object") visit(node.assistant_response);
    for (const k of Object.keys(node)) { const v = node[k]; if (v && typeof v === "object") visit(v); }
  };
  if (Array.isArray(resp?.output)) visit(resp.output); else visit(resp);
  return out.join("").trim();
}
function extractFinalTag(s = "") {
  const m = s.match(/<final>([\s\S]*?)<\/final>/i);
  return m ? m[1].trim() : "";
}

// ---------------- OpenAI (GPT-5) ----------------
async function gpt5Reply(userMsg) {
  const openai = getOpenAI();
  const INSTRUCTIONS =
    "Return ONLY the final answer. Be direct and clear in 1–4 sentences. " +
    "Wrap the final answer like this: <final>...your answer...</final>.";

  async function withTimeout(promise, label, reqPreview) {
    await dbg(label + "_request", reqPreview);
    return new Promise((resolve) => {
      const t = setTimeout(() => resolve({ __timeout: true }), LLM_TIMEOUT_MS);
      promise.then((r) => { clearTimeout(t); resolve(r); })
             .catch((e) => { clearTimeout(t); resolve({ __error: e }); });
    });
  }

  async function chatOnce(maxTokens) {
    try {
      const chatReq = {
        model: CHAT_MODEL,
        messages: [
          { role: "system", content: "You are a concise SMS assistant for Limi. Reply in 1–2 short sentences. No preamble, no bullets, no markdown. Keep it friendly and direct." },
          { role: "user", content: userMsg }
        ],
        max_completion_tokens: maxTokens
      };
      const r = await withTimeout(
        openai.chat.completions.create(chatReq),
        "gpt5_chat",
        { model: chatReq.model, input_preview: safeSlice(chatReq.messages, 200), maxTokens }
      );
      if (r?.__timeout) { await dbg("gpt5_chat_error", { message: "timeout", maxTokens }); return null; }
      if (r?.__error)   { await dbg("gpt5_chat_error", { message: String(r.__error?.message || r.__error), maxTokens }); return null; }
      const choice = r?.choices?.[0];
      const content = choice?.message?.content?.trim() || "";
      await dbg("gpt5_chat_reply", { has_content: !!content, finish_reason: choice?.finish_reason, usage: r?.usage, maxTokens });
      return { content, finish_reason: choice?.finish_reason };
    } catch (e) {
      await dbg("gpt5_chat_throw", { message: String(e?.message || e) });
      return null;
    }
  }

  let cr = await chatOnce(MAX_COMPLETION_TOKENS);
  if (cr?.content) return cr.content;
  if (!cr || cr.finish_reason === "length") {
    cr = await chatOnce(CHAT_RETRY_TOKENS);
    if (cr?.content) return cr.content;
  }

  // Responses API fallback A (structured)
  try {
    const req2 = {
      model: CHAT_MODEL,
      max_output_tokens: 300,
      instructions: INSTRUCTIONS,
      input: [{ role: "user", content: [{ type: "input_text", text: userMsg }]}]
    };
    let r = await withTimeout(getOpenAI().responses.create(req2), "gpt5_items", {
      model: req2.model, input_preview: safeSlice(req2.input, 200)
    });
    if (!r?.__timeout && !r?.__error) {
      const text = deepFindText(r);
      const final = extractFinalTag(text) || text;
      if (final) return final;
    }
  } catch (e) {
    await dbg("gpt5_items_throw", { message: String(e?.message || e) });
  }

  // Responses API fallback B (string input)
  try {
    const req3 = {
      model: CHAT_MODEL,
      max_output_tokens: 300,
      instructions: INSTRUCTIONS,
      input: `User: ${userMsg}\n\n<final>`
    };
    let r = await withTimeout(getOpenAI().responses.create(req3), "gpt5_string", {
      model: req3.model, input_preview: safeSlice(req3.input, 200)
    });
    if (!r?.__timeout && !r?.__error) {
      const text = deepFindText(r);
      const final = extractFinalTag(text) || text;
      if (final) return final;
    }
  } catch (e) {
    await dbg("gpt5_string_throw", { message: String(e?.message || e) });
  }

  // JSON fallback
  try {
    const req1 = {
      model: CHAT_MODEL,
      max_output_tokens: 300,
      text: { format: { type: "json_object" } },
      instructions: "You are an SMS assistant. Output JSON only.",
      input: [
        { role: "system", content: [{ type: "input_text", text: "You must reply in JSON only. Output a single JSON object with shape {\"final\":\"...\"}." }]},
        { role: "user", content: [{ type: "input_text", text: userMsg }]}
      ]
    };
    let r = await getOpenAI().responses.create(req1);
    const j = deepFindJson(r);
    if (j && typeof j.final === "string" && j.final.trim()) return j.final.trim();
    const text = deepFindText(r);
    if (text) return text;
  } catch (e) {
    await dbg("gpt5_jsonfmt_throw", { message: String(e?.message || e) });
  }

  return "I’ll keep it brief: I couldn’t generate a response.";
}

// ---------------- Twilio send (WA direct) ----------------
async function sendWhatsApp(to, body) {
  if (!TWILIO_WA_FROM) throw new Error("Missing TWILIO_WHATSAPP_FROM");
  const client = getTwilio();
  if (!client) throw new Error("Twilio credentials missing (TWILIO_SID/TWILIO_AUTH).");
  const waTo = to.startsWith("whatsapp:") ? to : `whatsapp:${to}`;
  const r = await client.messages.create({ from: TWILIO_WA_FROM, to: waTo, body });
  await dbg("twilio_send_wa_ok", { sid: r.sid, to: waTo });
  return r;
}

// ---------------- Handler ----------------
export default async function handler(req, res) {
  try {
    if (req.method !== "POST") { res.status(405).send("Method Not Allowed"); return; }

    const raw = await readRawBody(req);
    let Body = null, From = null, NumMedia = 0;

    if ((raw || "").trim().startsWith("{")) {
      // JSON payload (manual tests)
      let j = {};
      try { j = JSON.parse(raw); } catch {}
      Body = j.Body ?? j.body ?? "";
      From = j.From ?? j.from ?? "";
      if (j.channel === "whatsapp" && From && !/^whatsapp:/i.test(From)) From = `whatsapp:${From}`;
      NumMedia = Number(j.NumMedia || j.numMedia || 0);
    } else {
      // Twilio form-encoded
      const p = new URLSearchParams(raw);
      Body = p.get("Body");
      From = p.get("From");
      NumMedia = Number(p.get("NumMedia") || 0);
    }

    Body = (Body || "").trim();
    From = (From || "").trim();
    const isWhatsApp = /^whatsapp:/i.test(From);
    const cleanFrom = smsNumber(From);

    await dbg("webhook_in", {
      from: cleanFrom,
      channel: isWhatsApp ? "whatsapp" : "sms",
      body_len: Body.length,
      numMedia: NumMedia,
      envs: {
        has_OPENAI_KEY: !!(process.env.OPENAI_KEY),
        has_TWILIO_SID: !!(process.env.TWILIO_SID),
        has_TWILIO_AUTH: !!(process.env.TWILIO_AUTH),
        has_TWILIO_FROM: !!(TWILIO_FROM),
        has_TWILIO_WA_FROM: !!(TWILIO_WA_FROM)
      }
    });

    if (!Body) {
      res.setHeader("Content-Type", "text/xml");
      res.status(200).send(`<Response><Message>Empty message. Please send text.</Message></Response>`);
      return;
    }

    // UK SMS photo guard
    if (!isWhatsApp && NumMedia > 0) {
      res.setHeader("Content-Type", "text/xml");
      res.status(200).send(`<Response><Message>Pics don’t work over UK SMS. WhatsApp this same number instead 👍</Message></Response>`);
      return;
    }

    const reply = await gpt5Reply(Body);
    const safe = reply.length > MAX_SMS_CHARS ? reply.slice(0, MAX_SMS_CHARS - 1) + "…" : reply;

    // If inbound is WhatsApp and we have a WA sender, reply via Messages API and return empty TwiML
    if (isWhatsApp && TWILIO_WA_FROM) {
      try {
        await sendWhatsApp(cleanFrom, safe);
        res.setHeader("Content-Type","text/xml");
        res.status(200).send("<Response/>");
        await dbg("twiml_sent_wa", { to: cleanFrom, len: safe.length });
        return;
      } catch (e) {
        await dbg("twilio_send_wa_error", { to: cleanFrom, message: String(e?.message || e) });
        // fall through to TwiML SMS style just in case
      }
    }

    res.setHeader("Content-Type","text/xml");
    res.status(200).send(`<Response><Message>${toXml(safe)}</Message></Response>`);
    await dbg("twiml_sent", { to: cleanFrom, len: safe.length });
  } catch (e) {
    const msg = String(e?.message || e);
    await dbg("handler_error", { message: msg });
    // Always return 200 with TwiML so Twilio doesn’t retry, and you still see the error in logs.
    res.setHeader("Content-Type","text/xml");
    res.status(200).send(`<Response><Message>Unhandled error: ${toXml(msg)}</Message></Response>`);
  }
}

// api/webhook.js
export const config = { api: { bodyParser: false } };

import OpenAI from "openai";
import twilio from "twilio";
import { createClient } from "@supabase/supabase-js";

/** ────────────────────────── ENV ──────────────────────────
 * Required: OPENAI_KEY, TWILIO_SID, TWILIO_AUTH, TWILIO_FROM
 * Optional: TWILIO_WHATSAPP_FROM (format: whatsapp:+447...)
 * Optional: OPENAI_CHAT_MODEL (default gpt-5)
 * Optional: LLM_TIMEOUT_MS, WATCHDOG_MS, CHAT_MAX_TOKENS, CHAT_RETRY_TOKENS, SMS_MAX_CHARS
 * For debug logs: SUPABASE_URL, SUPABASE_KEY  (table: debug_logs)
 */
const openai = new OpenAI({ apiKey: process.env.OPENAI_KEY });
const twilioClient = twilio(process.env.TWILIO_SID, process.env.TWILIO_AUTH);
const supabase =
  process.env.SUPABASE_URL && process.env.SUPABASE_KEY
    ? createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY)
    : null;

const CHAT_MODEL = (process.env.OPENAI_CHAT_MODEL || "gpt-5").trim();

// Keep headroom under Twilio’s 15s total timeout.
const LLM_TIMEOUT_MS   = Number(process.env.LLM_TIMEOUT_MS || 6500);
// “Bail early” watchdog so we send *some* reply before Twilio gives up.
const WATCHDOG_MS      = Number(process.env.WATCHDOG_MS || 5200);

// Token caps
const CHAT_MAX_TOKENS  = Number(process.env.CHAT_MAX_TOKENS  || 120); // quick path
const CHAT_RETRY_TOKENS= Number(process.env.CHAT_RETRY_TOKENS|| 480); // one retry

// SMS length safety (Twilio will segment; we keep it short for UX)
const MAX_SMS_CHARS    = Number(process.env.SMS_MAX_CHARS || 320);

// Senders
const TWILIO_FROM   = (process.env.TWILIO_FROM || "").trim();
const TWILIO_WA_FROM= (process.env.TWILIO_WHATSAPP_FROM || "").trim();

/* ─────────────────────── Debug logging ─────────────────────── */
async function dbg(step, payload) {
  // always log to console
  try { console.log("[dbg]", step, payload); } catch {}
  // optionally log to supabase
  try {
    if (supabase) await supabase.from("debug_logs").insert([{ step, payload }]);
  } catch (e) {
    console.log("[dbg-fail]", step, String(e?.message || e));
  }
}

/* ───────────────────────── Utils ───────────────────────── */
function toXml(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}
function readRawBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", c => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}
function smsNumber(s="") {
  let v = String(s).replace(/^whatsapp:/i,"").trim();
  if (v.startsWith("00")) v = "+" + v.slice(2);
  if (v.startsWith("0"))  v = "+44" + v.slice(1);
  return v;
}
function safeSlice(obj, n=200) {
  try { return JSON.stringify(obj).slice(0, n); } catch { return String(obj).slice(0,n); }
}
function extractFinalTag(s="") {
  const m = s.match(/<final>([\s\S]*?)<\/final>/i);
  return m ? m[1].trim() : "";
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

/* ─────────────────── OpenAI helpers (timeouts) ─────────────────── */
function withTimeout(promise, ms, label, preview) {
  dbg(label + "_request", preview);
  return new Promise((resolve) => {
    const t = setTimeout(() => resolve({ __timeout: true }), ms);
    promise.then(r => { clearTimeout(t); resolve(r); })
           .catch(e => { clearTimeout(t); resolve({ __error: e }); });
  });
}

/* ───────────────── GPT-5: “fast path” + fallbacks ───────────────── */
async function gpt5Reply(userMsg) {
  // 1) Fast chat (short cap). The prompt strongly coerces a one-liner + <final> wrapper.
  const sys = "Return ONLY one short, friendly sentence. " +
              "Wrap the answer exactly as <final>your answer</final>. " +
              "No preamble. No markdown. Keep it direct.";

  const messages = [
    { role: "system", content: sys },
    { role: "user",   content: userMsg }
  ];

  const attempt = async (maxTokens, tag) => {
    const req = { model: CHAT_MODEL, messages, max_completion_tokens: maxTokens };
    const r = await withTimeout(openai.chat.completions.create(req), LLM_TIMEOUT_MS, tag, {
      model: req.model, input_preview: safeSlice(messages, 220), maxTokens
    });
    if (r?.__timeout) { await dbg("gpt5_chat_error", { message: "timeout", maxTokens }); return ""; }
    if (r?.__error)   { await dbg("gpt5_chat_error", { message: String(r.__error?.message||r.__error), maxTokens }); return ""; }

    const choice  = r?.choices?.[0];
    const content = choice?.message?.content?.trim() || "";
    await dbg(tag + "_reply", { has_content: !!content, usage: r?.usage, finish_reason: choice?.finish_reason });

    if (!content) return "";

    // Prefer <final>…</final>, otherwise whole content.
    return extractFinalTag(content) || content;
  };

  // Try quick → retry (higher cap) if the model spent everything on reasoning.
  let final = await attempt(CHAT_MAX_TOKENS, "gpt5_quick");
  if (!final) final = await attempt(CHAT_RETRY_TOKENS, "gpt5_chat");

  if (final) return final;

  // 2) Responses API – minimal string input with <final>
  const INSTRUCTIONS =
    "Return ONLY the final answer. One short sentence. " +
    "Wrap like this: <final>your answer</final>.";
  const req3 = { model: CHAT_MODEL, max_output_tokens: 240, instructions: INSTRUCTIONS,
                 input: `User: ${userMsg}\n\n<final>` };

  let r = await withTimeout(openai.responses.create(req3), LLM_TIMEOUT_MS, "gpt5_string", {
    model: req3.model, input_preview: safeSlice(req3.input, 220)
  });
  if (!r?.__timeout && !r?.__error) {
    const txt = deepFindText(r);
    const val = extractFinalTag(txt) || txt;
    await dbg("gpt5_items_shape", {
      final_len: (val||"").length,
      output0_keys: Array.isArray(r.output) && r.output[0] ? Object.keys(r.output[0]) : []
    });
    if (val) return val;
  } else if (r?.__timeout) {
    await dbg("gpt5_string_error", { message: "timeout" });
  } else {
    await dbg("gpt5_string_error", { message: String(r.__error?.message || r.__error) });
  }

  // 3) Responses API – items with input_text
  const req2 = {
    model: CHAT_MODEL,
    max_output_tokens: 240,
    instructions: INSTRUCTIONS,
    input: [{ role: "user", content: [{ type: "input_text", text: userMsg }] }]
  };

  r = await withTimeout(openai.responses.create(req2), LLM_TIMEOUT_MS, "gpt5_items", {
    model: req2.model, input_preview: safeSlice(req2.input, 220)
  });
  if (!r?.__timeout && !r?.__error) {
    const txt = deepFindText(r);
    const val = extractFinalTag(txt) || txt;
    await dbg("gpt5_items_shape", {
      final_len: (val||"").length,
      output0_keys: Array.isArray(r.output) && r.output[0] ? Object.keys(r.output[0]) : []
    });
    if (val) return val;
  } else if (r?.__timeout) {
    await dbg("gpt5_items_error", { message: "timeout" });
  } else {
    await dbg("gpt5_items_error", { message: String(r.__error?.message || r.__error) });
  }

  return ""; // let caller decide the static fallback
}

/* ───────────────── Twilio helpers ───────────────── */
function sendTwiml(res, body) {
  res.setHeader("Content-Type", "text/xml");
  res.status(200).send(`<Response><Message>${toXml(body)}</Message></Response>`);
}
async function sendWhatsApp(to, body) {
  const waTo = to.startsWith("whatsapp:") ? to : `whatsapp:${to}`;
  const r = await twilioClient.messages.create({ from: TWILIO_WA_FROM, to: waTo, body });
  await dbg("twilio_send_wa_ok", { sid: r.sid, to: waTo });
}

/* ───────────────── HTTP Handler ───────────────── */
export default async function handler(req, res) {
  const enterTs = Date.now();
  await dbg("handler_enter", { ts: enterTs, has_openai_key: !!process.env.OPENAI_KEY });

  try {
    if (req.method !== "POST") { res.status(405).send("Method Not Allowed"); return; }

    const raw = await readRawBody(req);
    let Body=null, From=null, NumMedia=0, Channel=null;

    if (raw.trim().startsWith("{")) {
      const j = JSON.parse(raw);
      Body = j.Body ?? j.body ?? "";
      From = j.From ?? j.from ?? "";
      Channel = (j.channel || "").toLowerCase();
      if (Channel === "whatsapp" && From && !/^whatsapp:/i.test(From)) From = `whatsapp:${From}`;
      NumMedia = Number(j.NumMedia || j.numMedia || 0);
    } else {
      const p = new URLSearchParams(raw);
      Body = p.get("Body");
      From = p.get("From");
      NumMedia = Number(p.get("NumMedia") || 0);
      // Twilio will pass whatsapp: in From for WA automatically
    }

    Body = (Body || "").trim();
    From = (From || "").trim();
    const isWhatsApp = /^whatsapp:/i.test(From);
    const cleanFrom = smsNumber(From);

    await dbg("webhook_in", {
      from: cleanFrom, channel: isWhatsApp ? "whatsapp" : "sms",
      body_len: Body.length, numMedia: NumMedia
    });

    if (!Body) { sendTwiml(res, "Empty message. Please send text."); return; }

    if (!isWhatsApp && NumMedia > 0) {
      sendTwiml(res, "Pics don’t work over UK SMS. WhatsApp this same number instead 👍");
      return;
    }

    // WATCHDOG: if model hasn’t answered by WATCHDOG_MS, send a friendly fallback
    let responded = false;
    const watchdog = setTimeout(() => {
      if (responded) return;
      const msg = "Sorry—service is a bit slow right now. Please try again.";
      sendTwiml(res, msg);
      responded = true;
      dbg("twiml_sent", { to: cleanFrom, len: msg.length, mode: "sms_twiML" });
    }, WATCHDOG_MS);

    // Get the GPT-5 reply
    const reply = await gpt5Reply(Body);
    if (responded) {
      // We already sent the watchdog message; just log the late model reply.
      clearTimeout(watchdog);
      await dbg("late_model_reply", { len: reply.length });
      return;
    }
    clearTimeout(watchdog);

    const final = (reply && reply.trim()) ? reply.trim() : "Sorry—service is a bit slow right now. Please try again.";
    const safe = final.length > MAX_SMS_CHARS ? final.slice(0, MAX_SMS_CHARS-1) + "…" : final;

    // Prefer WA API when inbound was WA; otherwise TwiML SMS
    if (isWhatsApp && TWILIO_WA_FROM) {
      try {
        await sendWhatsApp(cleanFrom, safe);
        res.setHeader("Content-Type","text/xml");
        res.status(200).send("<Response/>");
        responded = true;
        await dbg("twiml_sent", { to: cleanFrom, len: safe.length, mode: "whatsapp_api" });
        return;
      } catch (e) {
        await dbg("twilio_send_wa_error", { to: cleanFrom, message: String(e?.message||e) });
      }
    }

    sendTwiml(res, safe);
    responded = true;
    await dbg("twiml_sent", { to: cleanFrom, len: safe.length, mode: "sms_twiML" });

  } catch (e) {
    const msg = String(e?.message || e);
    await dbg("handler_error", { message: msg });
    // Best-effort graceful response so Twilio doesn’t 5xx
    try {
      sendTwiml(res, "Sorry—something went wrong. Please try again.");
    } catch {}
  }
}

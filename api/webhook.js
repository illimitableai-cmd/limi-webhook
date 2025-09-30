// api/webhook.js
export const config = { api: { bodyParser: false } };

import OpenAI from "openai";
import twilio from "twilio";
import { createClient } from "@supabase/supabase-js";

/** --- ENV ---
 * Required: OPENAI_KEY, TWILIO_SID, TWILIO_AUTH, TWILIO_FROM
 * Optional: TWILIO_WHATSAPP_FROM (format: whatsapp:+447...)
 * Optional: OPENAI_CHAT_MODEL (default: gpt-5), LLM_TIMEOUT_MS, SMS_MAX_CHARS, CHAT_MAX_TOKENS, WATCHDOG_MS
 * For debug logs: SUPABASE_URL, SUPABASE_KEY and a table `debug_logs`
 */
const openai = new OpenAI({ apiKey: process.env.OPENAI_KEY });
const twilioClient = twilio(process.env.TWILIO_SID, process.env.TWILIO_AUTH);
const supabase =
  process.env.SUPABASE_URL && process.env.SUPABASE_KEY
    ? createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY)
    : null;

const CHAT_MODEL = process.env.OPENAI_CHAT_MODEL || "gpt-5";
const LLM_TIMEOUT_MS = Number(process.env.LLM_TIMEOUT_MS || 5500); // beneath Twilio window
const WATCHDOG_MS = Number(process.env.WATCHDOG_MS || 9000);
const TWILIO_WA_FROM = (process.env.TWILIO_WHATSAPP_FROM || "").trim();
const MAX_SMS_CHARS = Number(process.env.SMS_MAX_CHARS || 320);
const MAX_COMPLETION_TOKENS = Number(process.env.CHAT_MAX_TOKENS || 160);

/* ---------------- logging (non-blocking) ---------------- */
function dbg(step, payload) {
  try { console.log("[dbg]", step, payload); } catch {}
  if (!supabase) return;
  supabase.from("debug_logs").insert([{ step, payload }]).catch(e =>
    console.log("[dbg-fail]", step, String(e?.message || e))
  );
}

/* ---------------- utils ---------------- */
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
function smsNumber(s="") {
  let v = String(s).replace(/^whatsapp:/i,"").trim();
  if (v.startsWith("00")) v = "+" + v.slice(2);
  if (v.startsWith("0")) v = "+44" + v.slice(1);
  return v;
}
function extractFinalTag(s="") {
  const m = s.match(/<final>([\s\S]*?)<\/final>/i);
  return m ? m[1].trim() : "";
}
function deepFindText(resp) {
  const out = [];
  const visit = (n) => {
    if (!n || typeof n !== "object") return;
    if (typeof n.output_text === "string" && n.output_text) out.push(n.output_text);
    if (typeof n.text === "string" && n.text) out.push(n.text);
    if (Array.isArray(n)) { for (const it of n) visit(it); return; }
    if (Array.isArray(n.content)) for (const it of n.content) visit(it);
    if (n.assistant_response) visit(n.assistant_response);
    for (const k of Object.keys(n)) {
      const v = n[k]; if (v && typeof v === "object") visit(v);
    }
  };
  if (Array.isArray(resp?.output)) visit(resp.output); else visit(resp);
  return out.join("").trim();
}

/* ---------------- GPT-5 hedged callers ---------------- */
// A. Chat Completions with "final only" contract
function chatFinal(userMsg) {
  const messages = [
    { role: "system",
      content:
        "Return ONLY the final answer. One short friendly sentence. " +
        "No preamble, no bullets, no markdown, no chain-of-thought." },
    { role: "user", content: userMsg }
  ];
  const req = {
    model: CHAT_MODEL,
    messages,
    max_completion_tokens: MAX_COMPLETION_TOKENS
  };
  dbg("gpt5_chat_request", { model: req.model, input_preview: JSON.stringify(messages).slice(0,160) });

  const p = openai.chat.completions.create(req);
  return new Promise((resolve) => {
    const t = setTimeout(() => resolve(""), LLM_TIMEOUT_MS);
    p.then(r => {
       clearTimeout(t);
       const txt = r?.choices?.[0]?.message?.content?.trim() || "";
       dbg("gpt5_chat_reply", {
         has_content: !!txt,
         finish_reason: r?.choices?.[0]?.finish_reason,
         usage: r?.usage
       });
       resolve(txt);
    }).catch(e => {
       clearTimeout(t);
       dbg("gpt5_chat_error", { message: String(e?.message || e) });
       resolve("");
    });
  });
}

// B. Responses API “<final> … </final>” contract
function responsesFinal(userMsg) {
  const req = {
    model: CHAT_MODEL,
    max_output_tokens: 200,
    instructions:
      "Return ONLY the final answer in 1 short sentence. " +
      "Wrap it exactly like this: <final>your answer</final>.",
    input: [{ role: "user", content: [{ type: "input_text", text: userMsg }]}]
  };
  dbg("gpt5_items_request", { model: req.model, input_preview: JSON.stringify(req.input).slice(0,160) });

  const p = openai.responses.create(req);
  return new Promise((resolve) => {
    const t = setTimeout(() => resolve(""), LLM_TIMEOUT_MS);
    p.then(r => {
      clearTimeout(t);
      const text = deepFindText(r);
      const final = extractFinalTag(text) || text || "";
      dbg("gpt5_items_shape", {
        output0_keys: Array.isArray(r.output) && r.output[0] ? Object.keys(r.output[0]) : [],
        final_len: final.length
      });
      resolve(final.trim());
    }).catch(e => {
      clearTimeout(t);
      dbg("gpt5_items_error", { message: String(e?.message || e) });
      resolve("");
    });
  });
}

/* Hedge both GPT-5 paths and return the first non-empty string */
async function gpt5Reply(userMsg) {
  const [a, b] = await Promise.all([chatFinal(userMsg), responsesFinal(userMsg)]);
  const txt = (a && a.trim()) || (b && b.trim()) || "";
  return txt || "Sorry—couldn’t answer just now.";
}

/* ---------------- WA direct send ---------------- */
async function sendWhatsApp(to, body) {
  if (!TWILIO_WA_FROM) throw new Error("Missing TWILIO_WHATSAPP_FROM");
  const waTo = to.startsWith("whatsapp:") ? to : `whatsapp:${to}`;
  const r = await twilioClient.messages.create({ from: TWILIO_WA_FROM, to: waTo, body });
  dbg("twilio_send_wa_ok", { sid: r.sid, to: waTo });
  return r;
}

/* ---------------- Handler with watchdog ---------------- */
export default async function handler(req, res) {
  res.setHeader("Cache-Control", "no-store");

  try {
    if (req.method !== "POST") { res.status(405).send("Method Not Allowed"); return; }

    const raw = await readRawBody(req);
    let Body=null, From=null, NumMedia=0, Channel=null;

    if (raw.trim().startsWith("{")) {
      const j = JSON.parse(raw);
      Body = j.Body ?? j.body ?? "";
      From = j.From ?? j.from ?? "";
      Channel = j.channel || null;
      if (Channel === "whatsapp" && From && !/^whatsapp:/i.test(From)) From = `whatsapp:${From}`;
      NumMedia = Number(j.NumMedia || j.numMedia || 0);
    } else {
      const p = new URLSearchParams(raw);
      Body = p.get("Body");
      From = p.get("From");
      NumMedia = Number(p.get("NumMedia") || 0);
      Channel = /^whatsapp:/i.test(From || "") ? "whatsapp" : "sms";
    }

    Body = (Body || "").trim();
    From = (From || "").trim();
    const isWhatsApp = /^whatsapp:/i.test(From);
    const cleanFrom = smsNumber(From);

    dbg("webhook_in", { from: cleanFrom, channel: isWhatsApp ? "whatsapp" : "sms", body_len: Body.length, numMedia: NumMedia });

    // Watchdog: guarantee Twilio gets a 200 within ~9s
    let responded = false;
    const sendOnce = (xml) => {
      if (responded) return false;
      try {
        res.setHeader("Content-Type","text/xml");
        res.status(200).send(xml);
        responded = true;
        return true;
      } catch { return false; }
    };
    const watchdog = setTimeout(() => {
      const fallback = "Sorry—service is a bit slow right now. Please try again.";
      if (sendOnce(`<Response><Message>${toXml(fallback)}</Message></Response>`)) {
        dbg("twiml_sent_watchdog", { len: fallback.length, ms: WATCHDOG_MS });
      }
    }, WATCHDOG_MS);

    if (!Body) {
      sendOnce(`<Response><Message>Empty message. Please send text.</Message></Response>`);
      clearTimeout(watchdog);
      return;
    }

    if (!isWhatsApp && NumMedia > 0) {
      sendOnce(`<Response><Message>Pics don’t work over UK SMS. WhatsApp this same number instead 👍</Message></Response>`);
      clearTimeout(watchdog);
      return;
    }

    const reply = (await gpt5Reply(Body)).trim();
    const safe = reply.length > MAX_SMS_CHARS ? reply.slice(0, MAX_SMS_CHARS - 1) + "…" : reply;

    if (isWhatsApp && TWILIO_WA_FROM) {
      try {
        await sendWhatsApp(cleanFrom, safe);
        sendOnce("<Response/>");
        clearTimeout(watchdog);
        dbg("twiml_sent", { to: cleanFrom, len: safe.length, mode: "wa_outbound" });
        return;
      } catch (e) {
        dbg("twilio_send_wa_error", { to: cleanFrom, message: String(e?.message || e) });
      }
    }

    sendOnce(`<Response><Message>${toXml(safe)}</Message></Response>`);
    clearTimeout(watchdog);
    dbg("twiml_sent", { to: cleanFrom, len: safe.length, mode: isWhatsApp ? "wa_twiML" : "sms_twiML" });
  } catch (e) {
    const msg = String(e?.message || e);
    dbg("handler_error", { message: msg });
    try {
      res.setHeader("Content-Type","text/xml");
      res.status(200).send(`<Response><Message>Unhandled error: ${toXml(msg)}</Message></Response>`);
    } catch {}
  }
}

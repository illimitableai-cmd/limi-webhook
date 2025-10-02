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

// GPT-5 only; mini is same-family fallback (NOT GPT-4)
const CHAT_MODEL          = process.env.OPENAI_CHAT_MODEL || "gpt-5";
const GPT5_FALLBACK_MODEL = process.env.GPT5_FALLBACK_MODEL || "gpt-5-mini";

// Token budgets
const CHAT_MAX_TOKENS     = Number(process.env.CHAT_MAX_TOKENS   || 180);
const CHAT_RETRY_TOKENS   = Number(process.env.CHAT_RETRY_TOKENS || 240);

// Twilio + runtime
const LLM_TIMEOUT_MS      = Number(process.env.LLM_TIMEOUT_MS || 11000);
const WATCHDOG_MS         = Number(process.env.WATCHDOG_MS || 12500);
const MAX_SMS_CHARS       = Number(process.env.SMS_MAX_CHARS || 320);
const TWILIO_WA_FROM      = (process.env.TWILIO_WHATSAPP_FROM || "").trim();
const NO_REPLY            = (process.env.NO_REPLY || "0") === "1";  // <- no-charge mode

// Output shaping
const FINAL_MAX_CHARS       = Number(process.env.FINAL_MAX_CHARS || 240);
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
const safeSlice = (obj, n = 300) => {
  try { return JSON.stringify(obj).slice(0, n); } catch { return String(obj).slice(0, n); }
};

// Responses API text digger
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
function wrapFinal(raw) {
  const got = extractFinal(raw || "");
  const use = got ? got : String(raw || "").replace(/\s+/g, " ").trim();
  if (!use) return "<final>I’m not sure.</final>";
  return `<final>${truncateChars(trimSentences(use))}</final>`;
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

/** --- Tools for chat fallback --------------------------------------- */
const FINALIZE_TOOL = [{
  type: "function",
  function: {
    name: "finalize",
    description: "Return the final user-facing answer text, concise and helpful.",
    parameters: {
      type: "object",
      properties: {
        text: {
          type: "string",
          description: `Final answer. Max ${FINAL_MAX_SENTENCES} short sentence(s), <= ${FINAL_MAX_CHARS} chars.`
        }
      },
      required: ["text"],
      additionalProperties: false
    }
  }
}];

/** --- OpenAI calls --------------------------------------------------- */
// A) Responses API with strict JSON schema (minimal, supported fields only)
async function responsesJsonFinal({ userMsg, maxTokens }) {
  const sys = [
    "You are Limi’s SMS/WhatsApp brain.",
    `Output MUST be valid JSON matching this schema: {"final": string up to ${FINAL_MAX_CHARS} chars, <= ${FINAL_MAX_SENTENCES} short sentence(s)}.`,
    "No extra keys, no explanations."
  ].join("\n");

  const req = {
    model: CHAT_MODEL,
    input: [
      { role: "system", content: sys },
      { role: "user",   content: userMsg }
    ],
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "limi_final",
        strict: true,
        schema: {
          type: "object",
          additionalProperties: false,
          properties: {
            final: { type: "string", maxLength: FINAL_MAX_CHARS }
          },
          required: ["final"]
        }
      }
    },
    max_output_tokens: maxTokens
  };

  const r = await withTimeout(openai.responses.create(req), "responses_json", { maxTokens });
  if (r?.__timeout || r?.__error) {
    await dbg("responses_json_error", { timeout: !!r?.__timeout, error: String(r?.__error?.message || r?.__error || "") });
    return "";
  }
  const text = deepFindText(r);
  try {
    const j = JSON.parse(text || "{}");
    const out = typeof j?.final === "string" ? j.final : "";
    return out.trim();
  } catch {
    return "";
  }
}

// B) Responses API plain text (fallback)
async function responsesPlain({ userMsg, maxTokens }) {
  const msgs = [
    { role: "system", content: `Return ONLY: <final>…</final>. Max ${FINAL_MAX_SENTENCES} sentences, ${FINAL_MAX_CHARS} chars.` },
    { role: "user",   content: userMsg }
  ];
  const r = await withTimeout(openai.responses.create({ model: CHAT_MODEL, input: msgs, max_output_tokens: maxTokens }), "responses_plain", {});
  if (r?.__timeout || r?.__error) {
    await dbg("responses_plain_error", { timeout: !!r?.__timeout, error: String(r?.__error?.message || r?.__error || "") });
    return "";
  }
  return deepFindText(r);
}

// C) Chat Completions with forced tool call on gpt-5-mini (same family)
async function chatFinalizeMini({ userMsg, maxTokens }) {
  const messages = [
    { role: "system", content: [
        "Call the function 'finalize' with {\"text\":\"...\"} containing ONLY the final answer.",
        `Use up to ${FINAL_MAX_SENTENCES} short sentence(s), <= ${FINAL_MAX_CHARS} characters.`,
        "No preamble or extra messages."
      ].join("\n")
    },
    { role: "user", content: userMsg }
  ];

  const req = {
    model: GPT5_FALLBACK_MODEL,
    messages,
    tools: FINALIZE_TOOL,
    tool_choice: { type: "function", function: { name: "finalize" } },
    max_completion_tokens: maxTokens
  };

  const r = await withTimeout(openai.chat.completions.create(req), "chat_finalize_mini", { maxTokens });
  if (r?.__timeout || r?.__error) {
    await dbg("chat_finalize_mini_error", { timeout: !!r?.__timeout, error: String(r?.__error?.message || r?.__error || "") });
    return "";
  }
  const msg = r?.choices?.[0]?.message || {};
  if (Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0) {
    try {
      const args = JSON.parse(msg.tool_calls[0]?.function?.arguments || "{}");
      return String(args?.text || "").trim();
    } catch {}
  }
  return "";
}

/** --- High-level LLM flow ------------------------------------------- */
async function llmReply(body) {
  // 1) JSON schema (Responses API on gpt-5)
  let out = await responsesJsonFinal({ userMsg: body, maxTokens: CHAT_MAX_TOKENS });

  // 2) If empty, plain Responses text (still gpt-5)
  if (!out) {
    const txt = await responsesPlain({ userMsg: body, maxTokens: CHAT_RETRY_TOKENS });
    if (txt) return wrapFinal(txt);
  }

  // 3) If still empty, gpt-5-mini with forced tool call
  if (!out) {
    const toolText = await chatFinalizeMini({ userMsg: body, maxTokens: CHAT_RETRY_TOKENS });
    if (toolText) return wrapFinal(toolText);
  }

  // 4) Last resort
  return wrapFinal(out);
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

    // Build the reply (no cross-family fallback)
    const replyPromise = llmReply(Body);
    const watchdog = new Promise((resolve) =>
      setTimeout(() => { dbg("watchdog_fired", { ms: WATCHDOG_MS }); resolve(null); }, WATCHDOG_MS)
    );
    const final = await Promise.race([replyPromise, watchdog]);
    const reply = final || "<final>Sorry—service is a bit slow right now. Please try again.</final>";
    const safeReply = reply.length > MAX_SMS_CHARS ? reply.slice(0, MAX_SMS_CHARS - 1) + "…" : reply;

    // ➤ NO-CHARGE TESTING: don't send any outbound message
    if (NO_REPLY) {
      await dbg("no_reply_mode", { would_send: safeReply, to: cleanFrom, channel: isWhatsApp ? "whatsapp" : "sms" });
      res.setHeader("Content-Type","text/xml");
      res.status(200).send("<Response/>"); // Twilio sends nothing → no charge
      return;
    }

    // WhatsApp: strip tags, SMS: keep tags (your client can parse)
    if (isWhatsApp && TWILIO_WA_FROM) {
      try {
        const waBody = safeReply.replace(/^<final>|<\/final>$/g, "");
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
    res.status(200).send(`<Response><Message>${toXml(safeReply)}</Message></Response>`);
    await dbg("twiml_sent", { to: cleanFrom, len: safeReply.length, mode: "sms_twiML" });

  } catch (e) {
    const msg = String(e?.message || e);
    await dbg("handler_error", { message: msg });
    res.setHeader("Content-Type","text/xml");
    res.status(200).send(`<Response><Message>Unhandled error: ${toXml(msg)}</Message></Response>`);
  }
}

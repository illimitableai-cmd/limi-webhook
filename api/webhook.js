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

// Give GPT-5 enough headroom to think + answer, but keep it modest for SMS.
const CHAT_MAX_TOKENS   = Number(process.env.CHAT_MAX_TOKENS   || 360);
const CHAT_RETRY_TOKENS = Number(process.env.CHAT_RETRY_TOKENS || 640);

const LLM_TIMEOUT_MS    = Number(process.env.LLM_TIMEOUT_MS || 11000);
const WATCHDOG_MS       = Number(process.env.WATCHDOG_MS || 12500);

const MAX_SMS_CHARS     = Number(process.env.SMS_MAX_CHARS || 320);
const TWILIO_WA_FROM    = (process.env.TWILIO_WHATSAPP_FROM || "").trim();

// 🔧 Output shaping (tweak without code changes)
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

/** --- Tools: force the answer via function call --------------------- */
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
          description: `The final answer text. Max ${FINAL_MAX_SENTENCES} short sentence(s), <= ${FINAL_MAX_CHARS} chars.`
        }
      },
      required: ["text"],
      additionalProperties: false
    }
  }
}];

/** --- OpenAI: Chat Completions for GPT-5 (no temp/stop) -------------- */
async function chatCreate(reqBase) {
  return withTimeout(openai.chat.completions.create(reqBase), "openai_chat", {
    model: reqBase?.model, maxTokens: reqBase?.max_completion_tokens || reqBase?.max_tokens
  });
}

async function safeChatCompletion({ messages, model = CHAT_MODEL, maxTokens = CHAT_MAX_TOKENS, tools, tool_choice }) {
  const isGpt5 = /^gpt-5/i.test(model);

  const req = { model, messages };
  if (tools) req.tools = tools;
  if (tool_choice) req.tool_choice = tool_choice;

  if (isGpt5) {
    req.max_completion_tokens = maxTokens; // GPT-5 param
    // Don’t send temperature/stop (unsupported for reasoning models). :contentReference[oaicite:1]{index=1}
  } else {
    req.max_tokens = maxTokens; // non-GPT-5 compat
    req.temperature = 1;
  }

  await dbg("openai_chat_request", { model, maxTokens, has_tools: !!tools, input_preview: safeSlice(messages) });
  const r = await chatCreate(req);

  if (r?.__timeout) { await dbg("openai_chat_error", { message: "timeout" }); return { text: "", toolText: "" }; }
  if (r?.__error)   { await dbg("openai_chat_error", { message: String(r.__error?.message || r.__error) }); return { text: "", toolText: "" }; }

  const choice = r?.choices?.[0];
  const msg = choice?.message || {};
  const text = (msg.content || "").trim();

  // If the model used the tool, extract the arguments
  let toolText = "";
  if (Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0) {
    try {
      const call = msg.tool_calls[0];
      const args = JSON.parse(call?.function?.arguments || "{}");
      toolText = String(args?.text || "").trim();
    } catch {}
  }

  await dbg("openai_chat_reply", {
    has_content: !!text,
    has_tool_text: !!toolText,
    usage: r?.usage,
    finish_reason: choice?.finish_reason
  });

  return { text, toolText };
}

/** --- LLM reply helpers --------------------------------------------- */
function systemInstruction() {
  return [
    "You are Limi’s SMS/WhatsApp brain.",
    `You must produce the final answer by CALLING the function 'finalize' with a JSON { \"text\": \"...\" } containing up to ${FINAL_MAX_SENTENCES} short sentence(s), <= ${FINAL_MAX_CHARS} characters total.`,
    "Do any reasoning silently; do NOT include extra messages.",
    "The 'text' must be the complete, user-ready answer (no disclaimers, no tags)."
  ].join("\n");
}

async function llmReply(userMsg) {
  // Attempt 1: force a tool call
  const messagesA = [
    { role: "system", content: systemInstruction() },
    { role: "user",   content: userMsg }
  ];
  let { text, toolText } = await safeChatCompletion({
    messages: messagesA,
    maxTokens: CHAT_MAX_TOKENS,
    tools: FINALIZE_TOOL,
    tool_choice: { type: "function", function: { name: "finalize" } } // hard requirement
  });

  // If tool produced text, prefer it
  let out = toolText || text;

  // Attempt 2: fallback to plain text formatting if tools ignored
  if (!out) {
    const messagesB = [
      { role: "system", content: [
          "Return ONLY the final answer, no preamble.",
          `Use up to ${FINAL_MAX_SENTENCES} short sentence(s), <= ${FINAL_MAX_CHARS} characters.`,
          "Wrap it exactly as <final>…</final>."
        ].join("\n")
      },
      { role: "user", content: userMsg }
    ];
    const second = await safeChatCompletion({ messages: messagesB, maxTokens: CHAT_RETRY_TOKENS });
    out = second.text || second.toolText || "";
  }

  // Guarantee a wrapped, trimmed result
  if (toolText) {
    return `<final>${truncateChars(trimSentences(toolText))}</final>`;
  }
  return coerceFinal(out);
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

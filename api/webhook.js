// api/webhook.js
export const config = { api: { bodyParser: false } };

import OpenAI from "openai";
import twilio from "twilio";
import { createClient } from "@supabase/supabase-js";

/** --- ENV ---
 * Required: OPENAI_KEY, TWILIO_SID, TWILIO_AUTH, TWILIO_FROM
 * Optional: TWILIO_WHATSAPP_FROM (format: whatsapp:+447...)
 * Optional: OPENAI_CHAT_MODEL (default: gpt-5), LLM_TIMEOUT_MS, SMS_MAX_CHARS, CHAT_MAX_TOKENS, CHAT_RETRY_TOKENS
 * For debug logs: SUPABASE_URL, SUPABASE_KEY and a table `debug_logs`
 */
const openai = new OpenAI({ apiKey: process.env.OPENAI_KEY });
const twilioClient = twilio(process.env.TWILIO_SID, process.env.TWILIO_AUTH);
const supabase =
  process.env.SUPABASE_URL && process.env.SUPABASE_KEY
    ? createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY)
    : null;

const CHAT_MODEL = process.env.OPENAI_CHAT_MODEL || "gpt-5";
const LLM_TIMEOUT_MS = Number(process.env.LLM_TIMEOUT_MS || 8000);
const TWILIO_FROM = (process.env.TWILIO_FROM || "").trim();
const TWILIO_WA_FROM = (process.env.TWILIO_WHATSAPP_FROM || "").trim();
const MAX_SMS_CHARS = Number(process.env.SMS_MAX_CHARS || 320);
const MAX_COMPLETION_TOKENS = Number(process.env.CHAT_MAX_TOKENS || 160);
const CHAT_RETRY_TOKENS = Number(process.env.CHAT_RETRY_TOKENS || 800);

/* ---------------- Debug logging ---------------- */
async function dbg(step, payload) {
  try {
    if (!supabase) { console.log("[dbg]", step, payload); return; }
    await supabase.from("debug_logs").insert([{ step, payload }]);
  } catch (e) {
    console.log("[dbg-fail]", step, String(e?.message || e));
  }
}

/* ---------------- Small utils ---------------- */
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

/* ---------------- Deep extraction helpers ---------------- */
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
function extractFinalTag(s="") {
  const m = s.match(/<final>([\s\S]*?)<\/final>/i);
  return m ? m[1].trim() : "";
}

/* ---------------- OpenAI (GPT-5) ---------------- */
export async function gpt5Reply(userMsg) {
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

  // helper: one chat call with a chosen token cap
  async function chatOnce(maxTokens) {
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
    if (r?.__timeout) {
      await dbg("gpt5_chat_error", { message: "timeout", maxTokens });
      return null;
    }
    if (r?.__error) {
      await dbg("gpt5_chat_error", { message: String(r.__error?.message || r.__error), maxTokens });
      return null;
    }
    const choice = r?.choices?.[0];
    const content = choice?.message?.content?.trim() || "";
    await dbg("gpt5_chat_reply", {
      has_content: !!content,
      finish_reason: choice?.finish_reason,
      usage: r?.usage,
      maxTokens
    });
    return { content, finish_reason: choice?.finish_reason, usage: r?.usage };
  }

  // --- Attempt A: Chat Completions (short cap)
  let cr = await chatOnce(MAX_COMPLETION_TOKENS);
  if (cr?.content) return cr.content;

  // If the model hit the cap (finish_reason: "length") or produced no content (reasoning ate the budget), retry with a bigger cap
  if (!cr || cr.finish_reason === "length") {
    cr = await chatOnce(CHAT_RETRY_TOKENS);
    if (cr?.content) return cr.content;
  }

  // --- Attempt B: Responses plain text with <final>
  const req2 = {
    model: CHAT_MODEL,
    max_output_tokens: 300,
    instructions: INSTRUCTIONS,
    input: [{ role: "user", content: [{ type: "input_text", text: userMsg }]}]
  };
  let r = await withTimeout(openai.responses.create(req2), "gpt5_items", {
    model: req2.model,
    input_preview: safeSlice(req2.input, 200)
  });
  if (r?.__timeout) {
    await dbg("gpt5_timeout", { attempt: "items", ms: LLM_TIMEOUT_MS });
  } else if (!r?.__error) {
    const output0 = Array.isArray(r.output) ? r.output[0] : null;
    await dbg("gpt5_items_shape", { output0_keys: output0 ? Object.keys(output0) : [], output0_shape: safeSlice(output0) });

    const text = deepFindText(r);
    const final = extractFinalTag(text) || text;
    if (final) {
      await dbg("gpt5_reply", { attempt: "items", len: final.length });
      return final;
    }
  } else {
    await dbg("gpt5_error", { attempt: "items", message: String(r.__error?.message || r.__error) });
  }

  // --- Attempt C: Responses minimal string with <final>
  const req3 = {
    model: CHAT_MODEL,
    max_output_tokens: 300,
    instructions: INSTRUCTIONS,
    input: `User: ${userMsg}\n\n<final>`
  };
  r = await withTimeout(openai.responses.create(req3), "gpt5_string", {
    model: req3.model,
    input_preview: safeSlice(req3.input, 200)
  });
  if (r?.__timeout) {
    await dbg("gpt5_timeout_retry", { attempt: "string" });
  } else if (!r?.__error) {
    const output0 = Array.isArray(r.output) ? r.output[0] : null;
    await dbg("gpt5_string_shape", { output0_keys: output0 ? Object.keys(output0) : [], output0_shape: safeSlice(output0) });

    const text = deepFindText(r);
    const final = extractFinalTag(text) || text;
    if (final) {
      await dbg("gpt5_reply_retry", { attempt: "string", len: final.length });
      return final;
    }
  } else {
    await dbg("gpt5_error_retry", { attempt: "string", message: String(r.__error?.message || r.__error) });
  }

  // --- Attempt D: Responses JSON using text.format + “JSON” in input
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
  r = await withTimeout(openai.responses.create(req1), "gpt5_jsonfmt", {
    model: req1.model,
    input_preview: safeSlice(req1.input, 200)
  });
  if (r?.__timeout) {
    await dbg("gpt5_timeout", { attempt: "jsonfmt", ms: LLM_TIMEOUT_MS });
  } else if (!r?.__error) {
    const output0 = Array.isArray(r.output) ? r.output[0] : null;
    await dbg("gpt5_jsonfmt_shape", { output0_keys: output0 ? Object.keys(output0) : [], output0_shape: safeSlice(output0) });

    const j = deepFindJson(r);
    if (j && typeof j.final === "string" && j.final.trim()) {
      const final = j.final.trim();
      await dbg("gpt5_reply", { attempt: "jsonfmt", len: final.length });
      return final;
    }
    const text = deepFindText(r);
    if (text) {
      await dbg("gpt5_reply", { attempt: "jsonfmt_text", len: text.length });
      return text;
    }
  } else {
    await dbg("gpt5_error", { attempt: "jsonfmt", message: String(r.__error?.message || r.__error) });
  }

  await dbg("gpt5_empty_after_retry", { model: CHAT_MODEL });
  return "I’ll keep it brief: I couldn’t generate a response.";
}

/* ---------------- Twilio send (WA direct) ---------------- */
async function sendWhatsApp(to, body) {
  if (!TWILIO_WA_FROM) throw new Error("Missing TWILIO_WHATSAPP_FROM");
  const waTo = to.startsWith("whatsapp:") ? to : `whatsapp:${to}`;
  const r = await twilioClient.messages.create({ from: TWILIO_WA_FROM, to: waTo, body });
  await dbg("twilio_send_wa_ok", { sid: r.sid, to: waTo });
  return r;
}

/* ---------------- Handler ---------------- */
export default async function handler(req, res) {
  try {
    if (req.method !== "POST") { res.status(405).send("Method Not Allowed"); return; }

    const raw = await readRawBody(req);
    let Body=null, From=null, NumMedia=0;

    if (raw.trim().startsWith("{")) {
      const j = JSON.parse(raw);
      Body = j.Body ?? j.body ?? "";
      From = j.From ?? j.from ?? "";
      if (j.channel === "whatsapp" && From && !/^whatsapp:/i.test(From)) From = `whatsapp:${From}`;
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

    const reply = await gpt5Reply(Body);
    const safe = reply.length > MAX_SMS_CHARS ? reply.slice(0, MAX_SMS_CHARS - 1) + "…" : reply;

    if (isWhatsApp && TWILIO_WA_FROM) {
      try {
        await sendWhatsApp(cleanFrom, safe);
        res.setHeader("Content-Type","text/xml");
        res.status(200).send("<Response/>");
        return;
      } catch (e) {
        await dbg("twilio_send_wa_error", { to: cleanFrom, message: String(e?.message || e) });
      }
    }

    res.setHeader("Content-Type","text/xml");
    res.status(200).send(`<Response><Message>${toXml(safe)}</Message></Response>`);
    await dbg("twiml_sent", { to: cleanFrom, len: safe.length });
  } catch (e) {
    const msg = String(e?.message || e);
    await dbg("handler_error", { message: msg });
    res.setHeader("Content-Type","text/xml");
    res.status(200).send(`<Response><Message>Unhandled error: ${toXml(msg)}</Message></Response>`);
  }
}

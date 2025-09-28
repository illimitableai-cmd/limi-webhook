// api/webhook.js
export const config = { api: { bodyParser: false } };

import OpenAI from "openai";
import twilio from "twilio";
import { createClient } from "@supabase/supabase-js";

/** --- ENV ---
 * Required: OPENAI_KEY, TWILIO_SID, TWILIO_AUTH, TWILIO_FROM
 * Optional: TWILIO_WHATSAPP_FROM (format: whatsapp:+447...)
 * Optional: OPENAI_CHAT_MODEL (default: gpt-5), LLM_TIMEOUT_MS
 * For debug logs: SUPABASE_URL, SUPABASE_KEY and a table `debug_logs`
 */
const openai = new OpenAI({ apiKey: process.env.OPENAI_KEY });
const twilioClient = twilio(process.env.TWILIO_SID, process.env.TWILIO_AUTH);
const supabase =
  process.env.SUPABASE_URL && process.env.SUPABASE_KEY
    ? createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY)
    : null;

const CHAT_MODEL = process.env.OPENAI_CHAT_MODEL || "gpt-5"; // full GPT-5 (rolling)
const LLM_TIMEOUT_MS = Number(process.env.LLM_TIMEOUT_MS || 8000);
const TWILIO_FROM = (process.env.TWILIO_FROM || "").trim();
const TWILIO_WA_FROM = (process.env.TWILIO_WHATSAPP_FROM || "").trim();

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
  if (v.startsWith("0")) v = "+44" + v.slice(1);
  return v;
}
function collapseResponsesText(resp) {
  if (!resp) return "";
  if (typeof resp.output_text === "string" && resp.output_text) return resp.output_text;
  try {
    const out = [];
    for (const item of resp.output || []) {
      for (const c of item?.content || []) {
        if (typeof c?.text === "string") out.push(c.text);
        else if (typeof c?.output_text === "string") out.push(c.output_text);
      }
    }
    return out.join("");
  } catch { return ""; }
}

/* ---------------- OpenAI (Responses API, GPT-5) ---------------- */
export async function gpt5Reply(userMsg) {
  function extractFinalBlock(s="") {
    const m = s.match(/<final>([\s\S]*?)<\/final>/i);
    return m ? m[1].trim() : "";
  }
  async function callWithTimeout(params, timeoutMs, dbgLabel) {
    await dbg(dbgLabel + "_request", {
      model: params.model,
      input_preview: typeof params.input === "string"
        ? params.input.slice(0,160)
        : (JSON.stringify(params.input)?.slice(0,160) || "")
    });
    const p = openai.responses.create(params);
    return new Promise((resolve) => {
      const t = setTimeout(() => resolve({ __timeout: true }), timeoutMs);
      p.then((r) => { clearTimeout(t); resolve(r); })
       .catch((e) => { clearTimeout(t); resolve({ __error: e }); });
    });
  }
  function getVisibleText(resp) {
    const raw = (collapseResponsesText(resp) || "").trim();
    const finalOnly = extractFinalBlock(raw);
    return (finalOnly || raw).trim();
  }

  // Shared knobs
  const INSTRUCTIONS =
    "Answer the user directly and clearly in 1–4 sentences. " +
    "Put ONLY your final answer between <final> and </final>. No reasoning text, no preamble.";

  // Attempt 1: items with input_text (most compatible)
  const params1 = {
    model: CHAT_MODEL,
    max_output_tokens: 300, // longer replies
    instructions: INSTRUCTIONS,
    input: [
      { role: "user", content: [{ type: "input_text", text: userMsg }] }
    ]
  };

  const r1 = await callWithTimeout(params1, LLM_TIMEOUT_MS, "gpt5A");
  if (r1?.__timeout) { await dbg("gpt5_timeout", { attempt: "A", ms: LLM_TIMEOUT_MS }); return "Sorry—took too long to respond."; }
  if (r1?.__error)   { await dbg("gpt5_error", { attempt: "A", message: String(r1.__error?.message || r1.__error) }); }
  else {
    const text1 = getVisibleText(r1);
    await dbg("gpt5_reply", { model: r1?.model, usage: r1?.usage, len: text1.length, preview: text1.slice(0,160) });
    if (text1) return text1;
  }

  // Attempt 2: plain string input (some snapshots prefer this minimal shape)
  const params2 = {
    model: CHAT_MODEL,
    max_output_tokens: 300,
    instructions: INSTRUCTIONS,
    input: `User: ${userMsg}\n\n<final>`
  };

  const r2 = await callWithTimeout(params2, Math.max(3000, Math.floor(LLM_TIMEOUT_MS/2)), "gpt5S");
  if (r2?.__timeout) { await dbg("gpt5_timeout_retry", { attempt: "S" }); return "Sorry—took too long to respond."; }
  if (r2?.__error)   { await dbg("gpt5_error_retry", { attempt: "S", message: String(r2.__error?.message || r2.__error) }); }
  else {
    const text2 = getVisibleText(r2);
    await dbg("gpt5_reply_retry", { model: r2?.model, usage: r2?.usage, len: text2.length, preview: text2.slice(0,160) });
    if (text2) return text2;
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

    // SMS photo guard (UK)
    if (!isWhatsApp && NumMedia > 0) {
      res.setHeader("Content-Type","text/xml");
      res.status(200).send(`<Response><Message>Pics don’t work over UK SMS. WhatsApp this same number instead 👍</Message></Response>`);
      return;
    }

    const reply = await gpt5Reply(Body);
    const safe = reply.length > 1200 ? reply.slice(0,1190) + "…" : reply;

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

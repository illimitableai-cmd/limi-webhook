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

const CHAT_MODEL = process.env.OPENAI_CHAT_MODEL || "gpt-5";
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

/* ---------------- Robust extractor for Responses API ---------------- */
function extractResponseText(resp) {
  if (!resp) return "";
  if (typeof resp.output_text === "string" && resp.output_text) return resp.output_text.trim();

  const chunks = [];
  const visit = (node) => {
    if (!node || typeof node !== "object") return;

    // common text fields
    if (typeof node.text === "string" && node.text) chunks.push(node.text);
    if (typeof node.output_text === "string" && node.output_text) chunks.push(node.output_text);

    // arrays / nested content
    if (Array.isArray(node)) { for (const it of node) visit(it); return; }
    if (Array.isArray(node.content)) { for (const it of node.content) visit(it); }
    if (node.assistant_response && typeof node.assistant_response === "object") visit(node.assistant_response);

    for (const k of Object.keys(node)) {
      const v = node[k];
      if (v && typeof v === "object") visit(v);
    }
  };
  if (Array.isArray(resp.output)) visit(resp.output);
  if (!chunks.length && typeof resp === "object") visit(resp);

  return chunks.join("").trim();
}
function summariseContentTypes(resp) {
  const counts = {};
  const visit = (node) => {
    if (!node || typeof node !== "object") return;
    if (Array.isArray(node)) { for (const it of node) visit(it); return; }
    if (node.type) counts[node.type] = (counts[node.type] || 0) + 1;
    if (Array.isArray(node.content)) for (const it of node.content) visit(it);
    if (node.assistant_response) visit(node.assistant_response);
    for (const k of Object.keys(node)) {
      const v = node[k];
      if (v && typeof v === "object") visit(v);
    }
  };
  if (resp && typeof resp === "object") visit(resp);
  return counts;
}

/* ---------------- OpenAI (Responses API, GPT-5) ---------------- */
export async function gpt5Reply(userMsg) {
  const INSTRUCTIONS =
    "Return ONLY the final answer. Be direct and clear in 1–4 sentences. " +
    "Wrap the final answer like this: <final>...your answer...</final>.";

  const extractFinal = (s="") => {
    const m = s.match(/<final>([\s\S]*?)<\/final>/i);
    return m ? m[1].trim() : "";
  };

  async function callWithTimeout(params, dbgLabel) {
    await dbg(dbgLabel + "_request", {
      model: params.model,
      input_preview:
        typeof params.input === "string"
          ? params.input.slice(0,160)
          : (JSON.stringify(params.input || "").slice(0,160))
    });
    const p = openai.responses.create(params);
    return new Promise((resolve) => {
      const t = setTimeout(() => resolve({ __timeout: true }), LLM_TIMEOUT_MS);
      p.then((r) => { clearTimeout(t); resolve(r); })
       .catch((e) => { clearTimeout(t); resolve({ __error: e }); });
    });
  }

  // Attempt 1 — JSON via top-level text.format (validator needs the word "json/JSON" in input)
  const paramsJson = {
    model: CHAT_MODEL,
    max_output_tokens: 300,
    text: { format: { type: "json_object" } }, // ✅ correct for Responses API
    instructions: "You are an SMS assistant. Output JSON only.",
    input: [
      {
        role: "system",
        content: [
          { type: "input_text", text: "You must reply in JSON only. Output a single JSON object with shape {\"final\":\"...\"}." } // contains “JSON”
        ]
      },
      { role: "user", content: [{ type: "input_text", text: userMsg }]}
    ],
  };

  let r = await callWithTimeout(paramsJson, "gpt5_jsonfmt");
  if (r?.__timeout) {
    await dbg("gpt5_timeout", { attempt: "jsonfmt", ms: LLM_TIMEOUT_MS });
    return "Sorry—took too long to respond.";
  }
  if (!r?.__error) {
    const all = extractResponseText(r);
    let final = "";
    try {
      const start = all.indexOf("{"); const end = all.lastIndexOf("}");
      const raw = (start >= 0 && end >= 0) ? all.slice(start, end + 1) : all;
      const j = JSON.parse(raw);
      if (j && typeof j.final === "string") final = j.final.trim();
    } catch { /* ignore and continue */ }
    await dbg("gpt5_reply", {
      attempt: "jsonfmt", model: r?.model, usage: r?.usage,
      len: (final||"").length, types: summariseContentTypes(r),
      preview: (final||"").slice(0,160)
    });
    if (final) return final;
  } else {
    await dbg("gpt5_error", { attempt: "jsonfmt", message: String(r.__error?.message || r.__error) });
  }

  // Attempt 2 — items + <final> contract (plain text, no tools)
  const paramsItems = {
    model: CHAT_MODEL,
    max_output_tokens: 300,
    instructions: INSTRUCTIONS,
    input: [{ role: "user", content: [{ type: "input_text", text: userMsg }]}],
  };
  r = await callWithTimeout(paramsItems, "gpt5_items");
  if (r?.__timeout) {
    await dbg("gpt5_timeout", { attempt: "items", ms: LLM_TIMEOUT_MS });
    return "Sorry—took too long to respond.";
  }
  if (!r?.__error) {
    const all = extractResponseText(r);
    const final = extractFinal(all) || all;
    await dbg("gpt5_reply", {
      attempt: "items", model: r?.model, usage: r?.usage,
      len: (final||"").length, types: summariseContentTypes(r),
      preview: (final||"").slice(0,160)
    });
    if (final) return final;
  } else {
    await dbg("gpt5_error", { attempt: "items", message: String(r.__error?.message || r.__error) });
  }

  // Attempt 3 — plain string + <final> (no tools)
  const paramsString = {
    model: CHAT_MODEL,
    max_output_tokens: 300,
    instructions: INSTRUCTIONS,
    input: `User: ${userMsg}\n\n<final>`,
  };
  r = await callWithTimeout(paramsString, "gpt5_string");
  if (r?.__timeout) {
    await dbg("gpt5_timeout_retry", { attempt: "string" });
    return "Sorry—took too long to respond.";
  }
  if (!r?.__error) {
    const all = extractResponseText(r);
    const final = extractFinal(all) || all;
    await dbg("gpt5_reply_retry", {
      attempt: "string", model: r?.model, usage: r?.usage,
      len: (final||"").length, types: summariseContentTypes(r),
      preview: (final||"").slice(0,160)
    });
    if (final) return final;
  } else {
    await dbg("gpt5_error_retry", { attempt: "string", message: String(r.__error?.message || r.__error) });
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

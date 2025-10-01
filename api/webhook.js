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

const CHAT_MODEL = process.env.OPENAI_CHAT_MODEL || "gpt-5";
const LLM_TIMEOUT_MS = Number(process.env.LLM_TIMEOUT_MS || 9500);
const WATCHDOG_MS = Number(process.env.WATCHDOG_MS || 11500); // Twilio hard-stop is 15s
const TWILIO_FROM = (process.env.TWILIO_FROM || "").trim();
const TWILIO_WA_FROM = (process.env.TWILIO_WHATSAPP_FROM || "").trim();
const MAX_SMS_CHARS = Number(process.env.SMS_MAX_CHARS || 320);
const CHAT_MAX_TOKENS = Number(process.env.CHAT_MAX_TOKENS || 120);
const CHAT_RETRY_TOKENS = Number(process.env.CHAT_RETRY_TOKENS || 480);

/** --- Debug logging -------------------------------------------------- */
async function dbg(step, payload) {
  try { console.log("[dbg]", step, payload); } catch {}
  try {
    if (supabase) await supabase.from("debug_logs").insert([{ step, payload }]);
  } catch (e) {
    console.log("[dbg-fail]", step, String(e?.message || e));
  }
}

/** --- Utils ---------------------------------------------------------- */
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
const safeSlice = (obj, n=200) => {
  try { return JSON.stringify(obj).slice(0,n); } catch { return String(obj).slice(0,n); }
};
const extractFinal = (s="") => {
  const m = String(s).match(/<final>([\s\S]*?)<\/final>/i);
  return m ? m[1].trim() : "";
};
function deepFindText(resp) {
  const out = [];
  const visit = (node) => {
    if (!node || typeof node !== "object") return;
    if (typeof node.output_text === "string") out.push(node.output_text);
    if (typeof node.text === "string") out.push(node.text);
    if (node.text && typeof node.text === "object" && typeof node.text.value === "string") out.push(node.text.value);
    if (node.json && typeof node.json === "object" && typeof node.json.final === "string") out.push(node.json.final);
    if (Array.isArray(node)) { for (const it of node) visit(it); return; }
    if (Array.isArray(node.content)) for (const it of node.content) visit(it);
    for (const k of Object.keys(node)) { const v=node[k]; if (v && typeof v === "object") visit(v); }
  };
  if (Array.isArray(resp?.output)) visit(resp.output); else visit(resp);
  return out.join("").trim();
}

/** --- OpenAI helpers ------------------------------------------------- */
function withTimeout(promise, label, ctx={}) {
  dbg(label + "_request", ctx);
  return new Promise((resolve) => {
    const t = setTimeout(() => resolve({ __timeout: true }), LLM_TIMEOUT_MS);
    promise.then((r) => { clearTimeout(t); resolve(r); })
           .catch((e) => { clearTimeout(t); resolve({ __error: e }); });
  });
}

async function attemptQuickChat(userMsg, maxTokens=CHAT_MAX_TOKENS) {
  const messages = [
    { role: "system", content: "Answer in 6–14 words. No analysis. If uncertain, say 'Not sure.' Wrap exactly as <final>…</final>. Return only that." },
    { role: "user", content: userMsg },
  ];
  const req = { model: CHAT_MODEL, messages, max_completion_tokens: maxTokens };
  const r = await withTimeout(openai.chat.completions.create(req), "gpt5_quick", {
    model: req.model, maxTokens, input_preview: safeSlice(messages)
  });

  if (r?.__timeout) { await dbg("gpt5_quick_error", { message:"timeout" }); return null; }
  if (r?.__error)   { await dbg("gpt5_quick_error", { message:String(r.__error?.message || r.__error) }); return null; }

  const choice = r?.choices?.[0];
  const txt = choice?.message?.content?.trim() || "";
  await dbg("gpt5_quick_reply", { has_content: !!txt, usage: r?.usage });
  const final = extractFinal(txt);
  return final || null;
}

async function attemptItems(userMsg) {
  const req = {
    model: CHAT_MODEL,
    max_output_tokens: 200,
    instructions: "Return only one sentence wrapped like <final>…</final>. No other text.",
    input: [{ role:"user", content:[{ type:"input_text", text: userMsg }]}],
  };
  const r = await withTimeout(openai.responses.create(req), "gpt5_items", {
    model: req.model, input_preview: safeSlice(req.input)
  });

  if (r?.__timeout) { await dbg("gpt5_items_error", { message:"timeout" }); return null; }
  if (r?.__error)   { await dbg("gpt5_items_error", { message:String(r.__error?.message || r.__error) }); return null; }

  const text = deepFindText(r);
  const final = extractFinal(text) || text;
  const output0 = Array.isArray(r.output) ? r.output[0] : null;
  await dbg("gpt5_items_shape", { final_len: (final||"").length, output0_keys: output0 ? Object.keys(output0) : [] });
  return final || null;
}

async function attemptChatRetry(userMsg, maxTokens=CHAT_RETRY_TOKENS) {
  const messages = [
    { role: "system", content: "Return only one short friendly sentence. Wrap exactly as <final>…</final>." },
    { role: "user", content: userMsg },
  ];
  const req = { model: CHAT_MODEL, messages, max_completion_tokens: maxTokens };
  const r = await withTimeout(openai.chat.completions.create(req), "gpt5_chat", {
    model: req.model, input_preview: safeSlice(messages), maxTokens
  });

  if (r?.__timeout) { await dbg("gpt5_chat_error", { message:"timeout", maxTokens }); return null; }
  if (r?.__error)   { await dbg("gpt5_chat_error", { message:String(r.__error?.message || r.__error), maxTokens }); return null; }

  const text = r?.choices?.[0]?.message?.content?.trim() || "";
  await dbg("gpt5_chat_reply", { has_content: !!text, usage: r?.usage, finish_reason: r?.choices?.[0]?.finish_reason });
  const final = extractFinal(text);
  return final || null;
}

/** Gatherer: launch staggered attempts and take the first good one */
async function gpt5Reply(userMsg) {
  const tasks = [];

  // start QUICK immediately
  tasks.push(attemptQuickChat(userMsg));

  // stagger: ITEMS after 150ms
  tasks.push((async () => { await new Promise(r => setTimeout(r,150)); return attemptItems(userMsg); })());

  // stagger: CHAT RETRY after 600ms
  tasks.push((async () => { await new Promise(r => setTimeout(r,600)); return attemptChatRetry(userMsg); })());

  // race for the first non-empty result (with watchdog)
  let settled = false;
  const watchdog = new Promise((resolve) => {
    setTimeout(() => {
      if (!settled) { dbg("watchdog_fired", { ms: WATCHDOG_MS }); resolve("__WATCHDOG__"); }
    }, WATCHDOG_MS);
  });

  const first = await Promise.race([
    (async () => {
      for await (const p of tasks) {
        try {
          const v = await p;
          if (v && !settled) { settled = true; return v; }
        } catch {}
      }
      // if none produced content, wait for any to finish (reduces unhandled rejects)
      const any = await Promise.any(tasks.map(async t => await t || null)).catch(() => null);
      return any || null;
    })(),
    watchdog
  ]);

  if (first === "__WATCHDOG__") return null;
  return first || null;
}

/** --- Twilio send (WA direct) --------------------------------------- */
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

    if (!isWhatsApp && NumMedia > 0) {
      res.setHeader("Content-Type","text/xml");
      res.status(200).send(`<Response><Message>Pics don’t work over UK SMS. WhatsApp this same number instead 👍</Message></Response>`);
      return;
    }

    // Ask the model(s)
    const final = await gpt5Reply(Body);

    // If watchdog fired or nothing worked, send graceful fallback
    const reply = final || "Sorry—service is a bit slow right now. Please try again.";
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

    // SMS response (Twiml)
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

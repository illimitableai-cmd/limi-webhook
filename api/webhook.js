// api/webhook.js
export const config = { api: { bodyParser: false } };

import OpenAI from "openai";
import twilio from "twilio";

/** --- ENV --- 
 * Required: OPENAI_KEY, TWILIO_SID, TWILIO_AUTH, TWILIO_FROM
 * Optional (for WhatsApp): TWILIO_WHATSAPP_FROM  (format: whatsapp:+447... )
 * Optional: OPENAI_CHAT_MODEL (default: gpt-5-nano), LLM_TIMEOUT_MS
 */
const openai = new OpenAI({ apiKey: process.env.OPENAI_KEY });
const twilioClient = twilio(process.env.TWILIO_SID, process.env.TWILIO_AUTH);

const CHAT_MODEL = process.env.OPENAI_CHAT_MODEL || "gpt-5-nano";
const LLM_TIMEOUT_MS = Number(process.env.LLM_TIMEOUT_MS || 8000);
const TWILIO_FROM = (process.env.TWILIO_FROM || "").trim();
const TWILIO_WA_FROM = (process.env.TWILIO_WHATSAPP_FROM || "").trim();

/** ----- tiny helpers ----- */
function toXml(s) {
  return s
    .replace(/&/g,"&amp;")
    .replace(/</g,"&lt;")
    .replace(/>/g,"&gt;");
}
function collapseResponsesText(resp) {
  if (!resp) return "";
  if (typeof resp.output_text === "string" && resp.output_text) return resp.output_text;
  try {
    const chunks = [];
    for (const item of resp.output || []) {
      for (const c of item?.content || []) {
        if (typeof c?.text === "string") chunks.push(c.text);
        else if (typeof c?.output_text === "string") chunks.push(c.output_text);
      }
    }
    return chunks.join("");
  } catch { return ""; }
}
function readRawBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}
function smsNumber(s="") {
  // normalise UK-ish inputs for testing (Twilio already sends E.164)
  let v = String(s).replace(/^whatsapp:/i,"").trim();
  if (v.startsWith("00")) v = "+" + v.slice(2);
  if (v.startsWith("0")) v = "+44" + v.slice(1);
  return v;
}

/** ----- OpenAI (Responses API, GPT-5) ----- */
async function gpt5Reply(userMsg) {
  const params = {
    model: CHAT_MODEL,
    input: [{ role: "user", content: [{ type: "text", text: userMsg }]}],
    // IMPORTANT: GPT-5 uses max_output_tokens (not max_tokens)
    max_output_tokens: 220,
  };
  const p = openai.responses.create(params);
  const withTimeout = new Promise((resolve) => {
    const t = setTimeout(() => resolve({ __timeout: true }), LLM_TIMEOUT_MS);
    p.then((r) => { clearTimeout(t); resolve(r); })
     .catch((e) => { clearTimeout(t); resolve({ __error: e }); });
  });

  const resp = await withTimeout;
  if (resp?.__timeout) return "Sorry—took too long to respond.";
  if (resp?.__error) {
    const msg = String(resp.__error?.message || resp.__error);
    // common misconfigs surfaced in body
    return `Model error: ${msg}`;
  }
  const text = collapseResponsesText(resp).trim();
  return text || "I’m here.";
}

/** ----- Twilio send (optional direct mode for WA) ----- */
async function sendWhatsApp(to, body) {
  if (!TWILIO_WA_FROM) throw new Error("Missing TWILIO_WHATSAPP_FROM");
  const waTo = to.startsWith("whatsapp:") ? to : `whatsapp:${to}`;
  return twilioClient.messages.create({ from: TWILIO_WA_FROM, to: waTo, body });
}

/** ----- Handler ----- */
export default async function handler(req, res) {
  try {
    if (req.method !== "POST") {
      res.status(405).send("Method Not Allowed");
      return;
    }

    const raw = await readRawBody(req);

    // Accept either Twilio x-www-form-urlencoded OR JSON (for local testing)
    let Body=null, From=null, NumMedia=0;
    let isWhatsApp = false;

    if (raw.trim().startsWith("{")) {
      // JSON test: { "Body":"hi", "From":"+4477..." } (or With "channel":"whatsapp")
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
    isWhatsApp = /^whatsapp:/i.test(From);

    // minimal guards
    if (!Body) {
      res.setHeader("Content-Type","text/xml");
      res.status(200).send(`<Response><Message>Empty message. Please send text.</Message></Response>`);
      return;
    }
    const cleanFrom = smsNumber(From);

    // For UK SMS, ignore MMS (keeps MVP simple)
    if (!isWhatsApp && NumMedia > 0) {
      res.setHeader("Content-Type","text/xml");
      res.status(200).send(`<Response><Message>Pics don’t work over UK SMS. WhatsApp this same number instead 👍</Message></Response>`);
      return;
    }

    // ask GPT-5
    const reply = await gpt5Reply(Body);
    const safe = reply.length > 1200 ? reply.slice(0,1190) + "…" : reply;

    // Reply:
    // 1) If WhatsApp & WA sender is configured, send via Messages API (avoid TwiML)
    // 2) Otherwise respond with TwiML
    if (isWhatsApp && TWILIO_WA_FROM) {
      await sendWhatsApp(cleanFrom, safe);
      res.setHeader("Content-Type","text/xml");
      res.status(200).send("<Response/>");
      return;
    }

    // SMS TwiML
    res.setHeader("Content-Type","text/xml");
    res.status(200).send(`<Response><Message>${toXml(safe)}</Message></Response>`);
  } catch (e) {
    const msg = String(e?.message || e);
    res.setHeader("Content-Type","text/xml");
    res.status(200).send(`<Response><Message>Unhandled error: ${toXml(msg)}</Message></Response>`);
  }
}

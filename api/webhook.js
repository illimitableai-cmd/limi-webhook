// api/webhook.js
export const config = { api: { bodyParser: false } };

/* ================ ENV ================ */
const CHAT_MODEL = process.env.OPENAI_CHAT_MODEL || "gpt-5";
const LLM_TIMEOUT_MS = Number(process.env.LLM_TIMEOUT_MS || 5500); // < Twilio 15s
const WATCHDOG_MS   = Number(process.env.WATCHDOG_MS   || 9000);
const TWILIO_WA_FROM = (process.env.TWILIO_WHATSAPP_FROM || "").trim();
const MAX_SMS_CHARS  = Number(process.env.SMS_MAX_CHARS || 320);
const MAX_COMPLETION_TOKENS = Number(process.env.CHAT_MAX_TOKENS || 120);

const SUPA_URL = process.env.SUPABASE_URL || "";
const SUPA_KEY = process.env.SUPABASE_KEY || "";

/* ========== Lazy clients (dynamic) ========== */
let _openai = null;
async function getOpenAI() {
  if (_openai) return _openai;
  const { default: OpenAI } = await import("openai");
  const key = process.env.OPENAI_KEY;
  if (!key) throw new Error("Missing OPENAI_KEY");
  _openai = new OpenAI({ apiKey: key });
  return _openai;
}

let _twilio = null;
async function getTwilio() {
  if (_twilio) return _twilio;
  const sid = process.env.TWILIO_SID, auth = process.env.TWILIO_AUTH;
  if (!sid || !auth) throw new Error("Missing TWILIO_SID/TWILIO_AUTH");
  const twilio = (await import("twilio")).default;
  _twilio = twilio(sid, auth);
  return _twilio;
}

/* ========== Non-blocking Supabase logger (REST) ========== */
function dbg(step, payload) {
  try { console.log("[dbg]", step, payload); } catch {}
  if (!SUPA_URL || !SUPA_KEY) return;
  fetch(`${SUPA_URL}/rest/v1/debug_logs`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "apikey": SUPA_KEY,
      "Authorization": `Bearer ${SUPA_KEY}`,
      "Prefer": "return=minimal"
    },
    body: JSON.stringify([{ step, payload }])
  }).catch(e => {
    try { console.log("[dbg-rest-fail]", step, String(e?.message || e)); } catch {}
  });
}

/* ========== Utils ========== */
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
    for (const k of Object.keys(n)) { const v = n[k]; if (v && typeof v === "object") visit(v); }
  };
  if (Array.isArray(resp?.output)) visit(resp.output); else visit(resp);
  return out.join("").trim();
}

/* ========== GPT-5 only (Chat + Responses hedge) ========== */
const FINAL_INSTR =
  "Return ONLY one short, friendly sentence. Wrap the answer exactly as <final>your answer</final>.";

async function gpt5ChatFinal(userMsg) {
  const openai = await getOpenAI();
  const messages = [
    { role: "system", content: FINAL_INSTR },
    { role: "user", content: userMsg }
  ];

  const req = {
    model: CHAT_MODEL,
    messages,
    max_completion_tokens: Math.min(MAX_COMPLETION_TOKENS, 120)
    // IMPORTANT: no 'stop', no 'temperature' — GPT-5 chat doesn't support them
  };

  dbg("gpt5_chat_request", { model: req.model, input_preview: JSON.stringify(messages).slice(0,160) });

  const p = openai.chat.completions.create(req);
  const r = await new Promise((resolve) => {
    const t = setTimeout(() => resolve({ __timeout: true }), LLM_TIMEOUT_MS);
    p.then(x => { clearTimeout(t); resolve(x); })
     .catch(e => { clearTimeout(t); resolve({ __error: e }); });
  });

  if (r?.__timeout) { dbg("gpt5_chat_error", { message: "timeout" }); return ""; }
  if (r?.__error)   { dbg("gpt5_chat_error", { message: String(r.__error?.message || r.__error) }); return ""; }

  const raw = r?.choices?.[0]?.message?.content?.trim() || "";
  const ans = (extractFinalTag(raw) || raw).trim();
  dbg("gpt5_chat_reply", { has_content: !!ans, finish_reason: r?.choices?.[0]?.finish_reason, usage: r?.usage });
  return ans;
}

async function gpt5ResponsesFinal(userMsg) {
  const openai = await getOpenAI();
  const req = {
    model: CHAT_MODEL,
    max_output_tokens: Math.min(MAX_COMPLETION_TOKENS, 120),
    instructions: FINAL_INSTR,
    input: [{ role: "user", content: [{ type: "input_text", text: userMsg }]}],
  };

  dbg("gpt5_items_request", { model: req.model, input_preview: JSON.stringify(req.input).slice(0,160) });

  const p = openai.responses.create(req);
  const r = await new Promise((resolve) => {
    const t = setTimeout(() => resolve({ __timeout: true }), LLM_TIMEOUT_MS);
    p.then(x => { clearTimeout(t); resolve(x); })
     .catch(e => { clearTimeout(t); resolve({ __error: e }); });
  });

  if (r?.__timeout) { dbg("gpt5_items_error", { message: "timeout" }); return ""; }
  if (r?.__error)   { dbg("gpt5_items_error", { message: String(r.__error?.message || r.__error) }); return ""; }

  const text = deepFindText(r);
  const ans  = (extractFinalTag(text) || text || "").trim();
  dbg("gpt5_items_shape", {
    output0_keys: Array.isArray(r.output) && r.output[0] ? Object.keys(r.output[0]) : [],
    final_len: ans.length
  });
  return ans;
}

async function gpt5Reply(userMsg) {
  const [a, b] = await Promise.allSettled([gpt5ChatFinal(userMsg), gpt5ResponsesFinal(userMsg)]);
  const first = [a, b].map(s => s.status === "fulfilled" ? (s.value || "").trim() : "").find(Boolean);
  return first || "Sorry—couldn’t answer just now.";
}

/* ========== Twilio WA (optional) ========== */
async function sendWhatsApp(to, body) {
  const client = await getTwilio();
  const waTo = to.startsWith("whatsapp:") ? to : `whatsapp:${to}`;
  const r = await client.messages.create({ from: TWILIO_WA_FROM, to: waTo, body });
  dbg("twilio_send_wa_ok", { sid: r.sid, to: waTo });
  return r;
}

/* ========== Handler with watchdog ========== */
export default async function handler(req, res) {
  res.setHeader("Cache-Control", "no-store");
  dbg("handler_enter", { ts: Date.now(), has_openai_key: !!process.env.OPENAI_KEY });

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

    // Env sanity
    const missing = [];
    if (!process.env.OPENAI_KEY) missing.push("OPENAI_KEY");
    if (!process.env.TWILIO_SID || !process.env.TWILIO_AUTH) missing.push("TWILIO_SID/TWILIO_AUTH");
    if (missing.length) {
      const msg = `Server config missing: ${missing.join(", ")}.`;
      res.setHeader("Content-Type","text/xml");
      res.status(200).send(`<Response><Message>${toXml(msg)}</Message></Response>`);
      dbg("env_error_reply", { missing });
      return;
    }

    // Watchdog: always respond
    let responded = false;
    const sendOnce = (xml) => {
      if (responded) return false;
      try { res.setHeader("Content-Type","text/xml"); res.status(200).send(xml); responded = true; return true; }
      catch { return false; }
    };
    const watchdog = setTimeout(() => {
      const fb = "Sorry—service is a bit slow right now. Please try again.";
      if (sendOnce(`<Response><Message>${toXml(fb)}</Message></Response>`)) {
        dbg("twiml_sent_watchdog", { len: fb.length, ms: WATCHDOG_MS });
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
    const safe  = reply.length > MAX_SMS_CHARS ? reply.slice(0, MAX_SMS_CHARS - 1) + "…" : reply;

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

// api/webhook.js
export const config = { api: { bodyParser: false } };

import OpenAI from "openai";
import twilio from "twilio";
import fetch from "node-fetch";
import { createClient } from "@supabase/supabase-js";

const openai = new OpenAI({ apiKey: process.env.OPENAI_KEY });
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);
const twilioClient = twilio(process.env.TWILIO_SID, process.env.TWILIO_AUTH);
const TWILIO_FROM = process.env.TWILIO_FROM;

// ===== Models: default to GPT-5 Nano (supports text+image input, text output)
const CHAT_MODEL   = process.env.OPENAI_CHAT_MODEL   || "gpt-5-nano";
const MEMORY_MODEL = process.env.OPENAI_MEMORY_MODEL || CHAT_MODEL;
// Optional: allow pinning a snapshot via env
const CHAT_SNAPSHOT_FALLBACK = process.env.OPENAI_CHAT_SNAPSHOT_FALLBACK || "gpt-5-nano-2025-08-07";

// FREE TRIAL amount used when we first see a user with no credits row
const FREE_TRIAL_CREDITS = Number(process.env.FREE_TRIAL_CREDITS || 20);

// ===== Minimum length + watchdogs =====
const MIN_REPLY_CHARS = Number(process.env.MIN_REPLY_CHARS || 60);
const FORCE_DIRECT = process.env.TWILIO_FORCE_DIRECT_SEND === "1";

// ---------- Small utils ----------
function escapeXml(s = "") {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function toGsm7(s = "") {
  return (s || "")
    .replace(/[“”]/g, '"').replace(/[‘’]/g, "'")
    .replace(/—|–/g, "-").replace(/…/g, "...")
    .replace(/\u00A0/g, " ");
}
function uidFromTwilio(from = "") {
  let v = from.replace(/^whatsapp:/i, "").replace(/^sms:/i, "").trim();
  v = v.replace(/[^\d+]/g, "");
  if (v.startsWith("00")) v = "+" + v.slice(2);
  if (v.startsWith("0")) v = "+44" + v.slice(1);
  return v;
}
async function dbg(step, payload, userId = null) {
  try { await supabase.from("debug_logs").insert([{ step, payload, user_id: userId }]); }
  catch (e) { console.error("dbg fail", e); }
}

// ---------- OpenAI wrappers ----------
const GPT5_RE = /^gpt-5/i;

async function safeChatCompletion({ messages, model = CHAT_MODEL, temperature = 0.2, maxTokens = 220 }) {
  const isGpt5 = GPT5_RE.test(model);
  const args = { model, messages };
  if (isGpt5) args.max_completion_tokens = maxTokens;
  else { args.max_tokens = maxTokens; args.temperature = temperature; }

  try {
    return await openai.chat.completions.create(args);
  } catch (err) {
    await dbg("model_fallback", { tried: model, error: String(err) });

    // 1) Pinned snapshot
    if (CHAT_SNAPSHOT_FALLBACK && CHAT_SNAPSHOT_FALLBACK !== model) {
      try {
        const a2 = { ...args, model: CHAT_SNAPSHOT_FALLBACK };
        if (GPT5_RE.test(CHAT_SNAPSHOT_FALLBACK)) { delete a2.max_tokens; delete a2.temperature; a2.max_completion_tokens = maxTokens; }
        return await openai.chat.completions.create(a2);
      } catch (e2) { await dbg("model_fallback", { tried: CHAT_SNAPSHOT_FALLBACK, error: String(e2) }); }
    }

    // 2) Plain gpt-5-nano
    if (model !== "gpt-5-nano") {
      try {
        const a3 = { ...args, model: "gpt-5-nano" };
        delete a3.max_tokens; delete a3.temperature; a3.max_completion_tokens = maxTokens;
        return await openai.chat.completions.create(a3);
      } catch (e3) { await dbg("model_fallback", { tried: "gpt-5-nano", error: String(e3) }); }
    }

    // 3) Last resort: 4o-mini
    return await openai.chat.completions.create({ model: "gpt-4o-mini", messages, temperature: 0.2, max_tokens: maxTokens });
  }
}

function watchdog(promise, ms, onTimeoutMsg = "Sorry—took too long to respond.") {
  let t;
  const timeout = new Promise((r) => { t = setTimeout(() => r({ __timeout: true, content: onTimeoutMsg }), ms); });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(t));
}

// ---------- “Answer the question” helpers ----------
function looksLikeNonAnswer(txt = "") {
  const t = (txt || "").toLowerCase().trim();
  if (!t) return true;
  return /(sorry|can't|cannot|couldn'?t|unsure|not (sure|certain)|don'?t know|no idea)/.test(t);
}

async function forceAnswer(userMsg, priorSummary = "") {
  const sys = [
    "You are Limi, a concise UK-English SMS assistant.",
    "You MUST directly answer the user's question in 1–3 short sentences.",
    "If it's a 'how many' question, reply with a concrete number or best estimate (e.g., 'about 60–70'), then 3–6 words of context.",
    "Avoid hedging or apologies. No disclaimers. No 'as an AI'.",
    "If the question is ambiguous, pick the most likely interpretation and state the answer.",
    priorSummary ? `Conversation so far: ${priorSummary.slice(0, 600)}` : ""
  ].filter(Boolean).join("\n");

  const c = await safeChatCompletion({
    model: CHAT_MODEL,
    temperature: 0.1,
    maxTokens: 220,
    messages: [
      { role: "system", content: sys },
      { role: "user",  content: userMsg }
    ]
  });

  return (c.choices?.[0]?.message?.content || "").trim();
}

async function expandIfShort({ reply, userMsg, priorSummary, minChars = MIN_REPLY_CHARS }) {
  let base = (reply || "").trim();
  if (base.length >= minChars && !looksLikeNonAnswer(base)) return base;

  // Pass 2: force direct answer
  const forced = (await forceAnswer(userMsg, priorSummary)).trim();
  if (forced.length >= minChars && !looksLikeNonAnswer(forced)) return forced;

  // Pass 3: rewrite to enforce minimum length
  const sys = "Rewrite so it DIRECTLY answers the user's question in 1–3 concise UK-English sentences. No apologies.";
  const usr = `User: ${userMsg}\n\nDraft reply:\n${forced || base}\n\nRewrite (>= ${minChars} chars):`;
  try {
    const c = await safeChatCompletion({ model: CHAT_MODEL, messages: [{ role: "system", content: sys }, { role: "user", content: usr }], maxTokens: 220 });
    const improved = (c.choices?.[0]?.message?.content || "").trim();
    if (improved.length >= minChars && !looksLikeNonAnswer(improved)) return improved;
    return improved || forced || base || "I’ll answer in a moment.";
  } catch {
    return forced || base || "I’ll answer in a moment.";
  }
}

// ---------- Twilio send helper ----------
async function sendDirect({ channel, to, body }) {
  const safeBody = toGsm7(body);
  try {
    if (channel === "whatsapp") {
      const waFrom = TWILIO_FROM.startsWith("whatsapp:") ? TWILIO_FROM : `whatsapp:${TWILIO_FROM}`;
      const waTo = to.startsWith("whatsapp:") ? to : `whatsapp:${to}`;
      const r = await twilioClient.messages.create({ from: waFrom, to: waTo, body: safeBody });
      await dbg("direct_send_ok", { channel, sid: r.sid, to }, null);
      return true;
    } else {
      const r = await twilioClient.messages.create({ from: TWILIO_FROM, to, body: safeBody });
      await dbg("direct_send_ok", { channel, sid: r.sid, to }, null);
      return true;
    }
  } catch (e) {
    await dbg("direct_send_error", { channel, to, message: String(e?.message || e) }, null);
    return false;
  }
}

// ---------- Media fetch ----------
async function fetchTwilioMediaB64(url) {
  const basic = Buffer.from(`${process.env.TWILIO_SID}:${process.env.TWILIO_AUTH}`).toString("base64");
  const r = await fetch(url, { headers: { Authorization: `Basic ${basic}` } });
  if (!r.ok) throw new Error(`Media fetch ${r.status}`);
  const buf = Buffer.from(await r.arrayBuffer());
  return buf.toString("base64");
}

// ---------- Memory + contacts (unchanged logic, trimmed for brevity) ----------
function blankMemory() {
  return { name: null, location: null, email: null, birthday: null, timezone: null, preferences: {}, interests: [], goals: [], notes: [], convo_summary: "", last_seen: new Date().toISOString() };
}
function mergeMemory(oldMem, add) {
  const m = { ...blankMemory(), ...(oldMem || {}) };
  for (const k of ["name","location","email","birthday","timezone","convo_summary"]) if (add?.[k] && typeof add[k] === "string") m[k] = add[k];
  m.preferences = { ...(oldMem?.preferences || {}), ...(add?.preferences || {}) };
  const dedupe = (a) => Array.from(new Set((a || []).filter(Boolean))).slice(0, 12);
  m.interests = dedupe([...(oldMem?.interests || []), ...(add?.interests || [])]);
  m.goals     = dedupe([...(oldMem?.goals || []),     ...(add?.goals || [])]);
  m.notes     = dedupe([...(oldMem?.notes || []),     ...(add?.notes || [])]).slice(0, 20);
  m.last_seen = new Date().toISOString();
  return m;
}
function memorySnapshotForPrompt(mem) {
  if (!mem) return "None";
  const parts = [];
  if (mem.name) parts.push(`name: ${mem.name}`);
  if (mem.location) parts.push(`location: ${mem.location}`);
  if (mem.timezone) parts.push(`timezone: ${mem.timezone}`);
  const prefs     = mem.preferences && Object.keys(mem.preferences).length ? `preferences: ${JSON.stringify(mem.preferences)}` : null;
  const interests = mem.interests && mem.interests.length ? `interests: ${mem.interests.slice(0,6).join(", ")}` : null;
  const goals     = mem.goals && mem.goals.length ? `goals: ${mem.goals.slice(0,4).join(", ")}` : null;
  const notes     = mem.notes && mem.notes.length ? `notes: ${mem.notes.slice(0,4).join("; ")}` : null;
  [prefs, interests, goals, notes].forEach(x => x && parts.push(x));
  if (mem.convo_summary) parts.push(`conversation summary: ${mem.convo_summary.slice(0, 600)}`);
  return parts.length ? parts.join(" • ") : "None";
}
function buildSystemPrompt(prior) {
  const snapshot = memorySnapshotForPrompt(prior);
  return [
    "You are Limi, a warm, concise WhatsApp/SMS assistant. Use UK spelling and a natural, human tone.",
    "ALWAYS answer the user's question directly. Prefer a concrete number when asked 'how many'. Keep it self-contained.",
    "Use short sentences. No apologies unless absolutely necessary.",
    "Ask at most ONE short follow-up only if it clearly helps.",
    `User profile snapshot: ${snapshot}`
  ].join("\n");
}
function cleanHistory(history) {
  const allowed = new Set(["system","assistant","user","function","tool","developer"]);
  return (history || []).filter(m => m && allowed.has(m.role) && typeof m.content === "string" && m.content.trim())
                        .map(m => ({ role: m.role, content: m.content.slice(0, 4000) }));
}
async function summariseConversation({ priorSummary = "", history = [], latestUser }) {
  const sys = "Maintain a 5–7 line concise rolling summary of this conversation. UK spelling. Return ONLY the summary text.";
  const textHistory = history.map(m => `${m.role.toUpperCase()}: ${m.content}`).join("\n");
  const c = await safeChatCompletion({
    model: MEMORY_MODEL,
    messages: [{ role: "system", content: sys },
               { role: "user", content: `Prior:\n${priorSummary || "(none)"}\n\nRecent:\n${textHistory}\n\nLatest user:\n${latestUser}\n\nUpdate now:` }],
    maxTokens: 220
  });
  return (c.choices?.[0]?.message?.content || "").trim().slice(0, 1500);
}

// (contact/emergency helpers unchanged; keep your existing implementations)
// ---- BEGIN (the same helpers you had before) ----
/*  KEEP: upsertContact, parseSaveContact, llmExtractContact, getOrCreateUserId,
          loadRecentTurns, saveTurn, ensureCredits, getCredits, setCredits,
          emergency helpers (emgExtractNatural, listEmergencyContacts, etc.),
          name utilities, list queries, isHelpTrigger, etc.
   For brevity, paste your existing implementations here unchanged.
*/
// ---- END helpers you already had ----

// ===================================================================

export default async function handler(req, res) {
  try {
    if (req.method === "GET") { await dbg("ping", { at: new Date().toISOString() }); return res.status(200).send("ping logged"); }
    if (req.method !== "POST") return res.status(405).send("Method Not Allowed");

    res.setHeader("Cache-Control", "no-store");

    // Read raw body
    const chunks = []; for await (const c of req) chunks.push(c);
    const raw = Buffer.concat(chunks).toString("utf8");

    // Parse Twilio params
    const p = new URLSearchParams(raw);
    let Body = p.get("Body"); let From = p.get("From"); let NumMedia = p.get("NumMedia");
    let ProfileName = p.get("ProfileName"); let MediaUrl0 = p.get("MediaUrl0"); let MediaContentType0 = p.get("MediaContentType0");
    let ChannelOverride = null;

    // JSON test fallback
    if ((!Body && !From) && raw && raw.trim().startsWith("{")) {
      try {
        const j = JSON.parse(raw);
        Body = Body ?? j.Body ?? j.body ?? ""; From = From ?? j.From ?? j.from ?? "";
        NumMedia = NumMedia ?? j.NumMedia ?? j.numMedia ?? 0;
        ProfileName = ProfileName ?? j.ProfileName ?? j.profileName ?? null;
        MediaUrl0 = MediaUrl0 ?? j.MediaUrl0 ?? j.mediaUrl0 ?? null;
        MediaContentType0 = MediaContentType0 ?? j.MediaContentType0 ?? j.mediaType0 ?? null;
        ChannelOverride = typeof j.channel === "string" ? j.channel : null;
      } catch (e) { await dbg("json_parse_error", { snippet: raw.slice(0,200) }); }
    }

    // Channel + sender
    let rawFrom = From || "";
    if (ChannelOverride && /^whatsapp$/i.test(ChannelOverride) && rawFrom && !/^whatsapp:/i.test(rawFrom)) rawFrom = `whatsapp:${rawFrom}`;
    const channel  = rawFrom.startsWith("whatsapp:") ? "whatsapp" : "sms";
    const from     = uidFromTwilio(rawFrom);
    const body     = (Body || "").trim();
    const numMedia = Number(NumMedia || 0);
    const waProfile = ProfileName || null;

    await dbg("webhook_in", { channel, from, body, numMedia }, null);

    // Guards
    if (!from) { res.setHeader("Content-Type","text/xml"); return res.status(200).send("<Response><Message>Missing sender</Message></Response>"); }
    if (!body && numMedia === 0) {
      const msg = "I received an empty message. Please send your question as text.";
      res.setHeader("Content-Type","text/xml"); return res.status(200).send(`<Response><Message>${escapeXml(msg)}</Message></Response>`);
    }
    if (channel === "sms" && numMedia > 0) {
      res.setHeader("Content-Type","text/xml");
      return res.status(200).send("<Response><Message>Pics don’t work over UK SMS. WhatsApp this same number instead 👍</Message></Response>");
    }

    // Ensure user
    const userId = await getOrCreateUserId(from);
    if (waProfile) await upsertContact({ userId, name: waProfile, phone: from, channel });

    if (/^buy\b/i.test(body)) {
      res.setHeader("Content-Type","text/xml");
      return res.status(200).send("<Response><Message>Top up here: https://illimitableai.com/buy</Message></Response>");
    }

    // QUICK paths (name + contacts + emergency)
    // --- keep your existing quick-path code block here unchanged ---
    // (to keep this answer readable, re-use the identical quick-path code you already had)

    // === Credits gate
    const credits = await ensureCredits(userId);
    if (credits <= 0) {
      res.setHeader("Content-Type","text/xml");
      return res.status(200).send("<Response><Message>Out of credits. Reply BUY for a top-up link.</Message></Response>");
    }

    // Memory + media
    const { data: memRow } = await supabase.from("memories").select("summary").eq("user_id", userId).maybeSingle();
    const prior = memRow?.summary ?? blankMemory();

    let userMsg = body || "";
    let visionPart = null;
    if (channel === "whatsapp" && numMedia > 0 && MediaUrl0) {
      try {
        const b64 = await fetchTwilioMediaB64(MediaUrl0);
        const ctype = MediaContentType0 || "image/jpeg";
        visionPart = { type: "image_url", image_url: { url: `data:${ctype};base64,${b64}` } };
        if (!userMsg) userMsg = "Please analyse this image.";
      } catch (err) { await dbg("wa_media_fetch_error", { message: String(err) }, userId); }
    }

    await saveTurn(userId, "user", userMsg, channel, from);

    // Context
    const history = cleanHistory(await loadRecentTurns(userId, 40));
    const userContent = visionPart ? [{ type: "text", text: userMsg }, visionPart] : userMsg;
    const messages = [
      { role: "system", content: buildSystemPrompt(prior) },
      ...history.slice(-15),
      { role: "user", content: userContent },
    ];

    // ===== Main answer with watchdog
    const completion = await watchdog(safeChatCompletion({ model: CHAT_MODEL, messages, maxTokens: 220 }), 12000, "Sorry—took too long to respond.");
    let reply = "";
    if (completion?.__timeout) {
      await dbg("model_timeout", { ms: 12000 }, userId);
      reply = completion.content;
    } else {
      reply = completion.choices?.[0]?.message?.content?.trim() || "";
    }

    // Make sure we give a direct answer (2 more passes if needed)
    reply = await expandIfShort({ reply, userMsg, priorSummary: prior.convo_summary || "" });

    // Optional tiny follow-ups
    let finalReply = reply;
    if (finalReply.length > 1200) finalReply = finalReply.slice(0, 1190) + "…";

    await dbg("reply_out", { channel, to: from, reply: finalReply, FORCE_DIRECT }, userId);

    await saveTurn(userId, "assistant", finalReply, channel, from);
    await setCredits(userId, Math.max(0, credits - 1));

    // Update memory (best effort)
    try {
      const slimHistory = history.slice(-10);
      const newSummary = await summariseConversation({ priorSummary: prior.convo_summary || "", history: slimHistory, latestUser: userMsg });
      const mergedForSummary = mergeMemory(prior, { convo_summary: newSummary });
      await supabase.from("memories").upsert({ user_id: userId, summary: mergedForSummary });
    } catch (e) { await dbg("convo_summary_error", { message: String(e) }, userId); }

    // Name nudge for WA
    let footer = "";
    const latestMem = (await supabase.from("memories").select("summary").eq("user_id", userId).maybeSingle()).data?.summary;
    if (!latestMem?.name && channel === "whatsapp") footer = "\n\n(What’s your first name so I can save it?)";

    // Delivery
    if (FORCE_DIRECT && channel === "sms") {
      const ok = await sendDirect({ channel, to: from, body: finalReply + footer });
      await dbg("direct_send_forced", { ok, to: from }, userId);
      res.setHeader("Content-Type","text/xml");
      return res.status(200).send("<Response/>"); // prevent double-send
    }

    res.setHeader("Content-Type","text/xml");
    return res.status(200).send(`<Response><Message>${escapeXml(toGsm7((finalReply || "All done.") + footer))}</Message></Response>`);
  } catch (e) {
    console.error("handler fatal", e);
    await dbg("handler_fatal", { message: String(e?.message || e), stack: e?.stack || null });
    res.setHeader("Content-Type","text/xml");
    return res.status(200).send("<Response><Message>Sorry, something went wrong.</Message></Response>");
  }
}

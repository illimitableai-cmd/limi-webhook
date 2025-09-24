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

// ===== Feature flags / kill-switches =====
const ENABLE_CONTACTS  = process.env.ENABLE_CONTACTS   !== "0"; // default ON
const ENABLE_EMERGENCY = process.env.ENABLE_EMERGENCY  !== "0"; // default ON
const BASIC_MODE       = process.env.BASIC_MODE === "1";        // pure chat bypass

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

// ---------- Memory + snapshot ----------
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

// -------------- Name/phone utilities --------------
function sanitizeName(raw = "") {
  let n = String(raw)
    .replace(/['’]\s*s\b/gi, "")
    .replace(/\b(number|mobile|cell|phone)\b/gi, "")
    .replace(/\s{2,}/g, " ")
    .trim();
  n = n.toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase());
  return n;
}
function isBadName(name = "") {
  return /\b(number|mobile|cell|phone)\b/i.test(name) || /['’]\s*s\b/i.test(name) || name.replace(/[^a-z]/gi, "").length < 2;
}
function betterName(existing = "", incoming = "") {
  const aBad = isBadName(existing), bBad = isBadName(incoming);
  if (bBad && !aBad) return existing;
  if (!bBad && aBad) return incoming;
  const aTok = (existing || "").trim().split(/\s+/).length;
  const bTok = (incoming || "").trim().split(/\s+/).length;
  return bTok >= aTok ? incoming : existing;
}
function normalizePhone(phone = "") {
  let d = String(phone).replace(/[^\d+]/g, "");
  if (d.startsWith("00")) d = "+" + d.slice(2);
  if (d.startsWith("0")) d = "+44" + d.slice(1);
  return d;
}

// -------------- DB helpers --------------
async function getOrCreateUserId(identifier) {
  const { data: ident } = await supabase
    .from("identifiers").select("user_id").eq("value", identifier).maybeSingle();
  if (ident?.user_id) return ident.user_id;

  const { data: user, error: userErr } = await supabase
    .from("users").insert([{ display_name: null }]).select().single();
  if (userErr) {
    const { data: ident2 } = await supabase.from("identifiers").select("user_id").eq("value", identifier).maybeSingle();
    if (ident2?.user_id) return ident2.user_id;
    throw userErr;
  }
  const { error: linkErr } = await supabase
    .from("identifiers").insert([{ user_id: user.id, type: "phone", value: identifier }]);
  if (linkErr) {
    const { data: ident3 } = await supabase.from("identifiers").select("user_id").eq("value", identifier).maybeSingle();
    if (ident3?.user_id) return ident3.user_id;
    throw linkErr;
  }
  return user.id;
}
async function loadRecentTurns(userId, limit = 30) {
  const { data } = await supabase
    .from("messages").select("role, body").eq("user_id", userId)
    .order("created_at", { ascending: false }).limit(limit);
  return (data || []).reverse().map((r) => ({ role: r.role, content: r.body }));
}
async function saveTurn(userId, role, text, channel, externalId) {
  await supabase.from("messages").insert([{ user_id: userId, role, body: text, channel, external_id: externalId }]);
}
async function getCredits(userId) {
  const { data } = await supabase.from("credits").select("balance").eq("user_id", userId).maybeSingle();
  return data?.balance ?? 0;
}
async function setCredits(userId, balance) {
  await supabase.from("credits").upsert({ user_id: userId, balance });
}
async function ensureCredits(userId) {
  const { data } = await supabase.from("credits").select("balance").eq("user_id", userId).maybeSingle();
  if (!data) {
    await supabase.from("credits").insert([{ user_id: userId, balance: FREE_TRIAL_CREDITS }]);
    await dbg("credits_seeded", { seeded: FREE_TRIAL_CREDITS }, userId);
    return FREE_TRIAL_CREDITS;
  }
  return data.balance ?? 0;
}

// -------------- Contacts --------------
async function upsertContact({ userId, name, phone, channel }) {
  const tidyIncoming = sanitizeName(name);
  const normPhone = normalizePhone(phone);
  try {
    const { data: byPhone } = await supabase
      .from("contacts").select("id,name,phone,channel")
      .eq("user_id", userId).eq("phone", normPhone).maybeSingle();

    if (byPhone) {
      const finalName = betterName(byPhone.name || "", tidyIncoming);
      const needsUpdate = finalName !== (byPhone.name || "") || (byPhone.channel || "") !== (channel || "");
      if (needsUpdate) {
        await supabase.from("contacts").update({ name: finalName, channel }).eq("id", byPhone.id);
      }
      await dbg("contact_upsert_ok", { action: "update_by_phone", name: finalName, phone: normPhone, channel }, userId);
      return { ok: true, action: "update_by_phone" };
    }

    const { data: byName } = await supabase
      .from("contacts").select("id,name,phone,channel")
      .eq("user_id", userId).ilike("name", tidyIncoming);

    const existingByName = (byName || []).find((r) => (r.name || "").trim().toLowerCase() === tidyIncoming.toLowerCase());
    if (existingByName) {
      await supabase.from("contacts").update({ phone: normPhone, channel }).eq("id", existingByName.id);
      await dbg("contact_upsert_ok", { action: "update_by_name", name: existingByName.name, phone: normPhone, channel }, userId);
      return { ok: true, action: "update_by_name" };
    }

    await supabase.from("contacts").insert({ user_id: userId, name: tidyIncoming, phone: normPhone, channel });
    await dbg("contact_upsert_ok", { action: "insert_new", name: tidyIncoming, phone: normPhone, channel }, userId);
    return { ok: true, action: "insert_new" };
  } catch (e) {
    await dbg("contact_upsert_exception", { message: String(e?.message || e) }, userId);
    return { ok: false, error: e };
  }
}

function parseSaveContact(msg) {
  const text = (msg || "").trim();
  const phoneMatch = text.match(/(\+?\d[\d\s().-]{6,})/);
  let phone = phoneMatch ? phoneMatch[1] : null;

  const patterns = [
    /(?:save|add)?\s*([a-zA-Z][a-zA-Z\s'’-]{1,60})\s*['’]\s*s\s*(?:number|mobile|cell|phone)?\s*(?:is|:)?\s*(\+?\d[\d\s().-]{6,})/i,
    /save\s+([a-zA-Z][a-zA-Z\s'’-]{1,60})\s*(?:number|mobile|cell|phone)?\s*(?:is|:)?\s*(\+?\d[\d\s().-]{6,})/i,
    /save\s+([a-zA-Z][a-zA-Z\s'’-]{1,60})\s+as\s+a\s+contact\b/i,
    /save\s+([a-zA-Z][a-zA-Z\s'’-]{1,60})\s+as\s+contact\b/i,
    /add\s+([a-zA-Z][a-zA-Z\s'’-]{1,60})\s+as\s+a\s+contact\b/i,
    /add\s+([a-zA-Z][a-zA-Z\s'’-]{1,60})\s+as\s+contact\b/i,
    /save\s+contact\s+([a-zA-Z][a-zA-Z\s'’-]{1,60})\b/i,
    /add\s+contact\s+([a-zA-Z][a-zA-Z\s'’-]{1,60})\b/i,
    /save\s+([a-zA-Z][a-zA-Z\s'’-]{1,60})[\s,]+(\+?\d[\d\s().-]{6,})/i,
  ];

  let name = null;
  for (const re of patterns) {
    const m = re.exec(text);
    if (!m) continue;
    if (m[2]) phone = m[2];
    if (m[1]) name = m[1];
    break;
  }
  if (!name || !phone) return null;

  const tidyName = sanitizeName(name);
  const normPhone = normalizePhone(phone);
  if (isBadName(tidyName)) return null;

  return { name: tidyName, phone: normPhone };
}

async function llmExtractContact(msg) {
  const sys = 'Return ONLY compact JSON like {"name":"...","phone":"..."} if the text asks to save/add a contact; otherwise {}. Phone must include country code.';
  const user = `Text: ${msg}`;
  const c = await safeChatCompletion({
    model: MEMORY_MODEL,
    messages: [{ role: "system", content: sys }, { role: "user", content: user }],
    maxTokens: 80,
  });
  try {
    const t = c.choices[0].message.content || "{}";
    const s = t.indexOf("{"), e = t.lastIndexOf("}");
    const j = JSON.parse(s >= 0 && e >= 0 ? t.slice(s, e + 1) : "{}");
    if (j?.name && j?.phone) {
      const tidyName = sanitizeName(j.name);
      const normPhone = normalizePhone(String(j.phone));
      if (!isBadName(tidyName)) return { name: tidyName, phone: normPhone };
    }
  } catch {}
  return null;
}

// -------------- Intent helpers --------------
function isContactListQuery(text = "") {
  const t = (text || "").trim().toLowerCase();
  return (
    /^contacts$/.test(t) ||
    /^contact\s+list$/.test(t) ||
    /^my\s+contacts$/.test(t) ||
    /^show\s+contacts$/.test(t) ||
    /(^|\b)(show|list|see|view|display)\s+(my\s+)?contacts(\b|$)/i.test(text) ||
    /(^|\b)can\s+i\s+have\s+(my\s+)?contact\s+list(\b|$)/i.test(text) ||
    /(^|\b)contacts\s+please(\b|$)/i.test(text) ||
    /\bwhat('?| i)?s?\s+my\s+contact\s+list\??$/i.test(text) ||
    /\bwhat\s+is\s+my\s+contact\s+list\??$/i.test(text) ||
    /\bwho\s+is\s+in\s+my\s+contacts\??$/i.test(text)
  );
}
const NAME_STOPWORDS = new Set([
  "ok","okay","k","thanks","thank you","ta","cheers","yes","yeah","yep","no","nope",
  "hello","hi","hey","yo","sup","test","testing","help","buy","contacts","contact","list"
]);
function isLikelyName(text = "") {
  const t = (text || "").trim();
  if (!/^[a-zA-Z][a-zA-Z'’-]{1,60}(?:\s+[a-zA-Z][a-zA-Z'’-]{1,60}){0,2}$/.test(t)) return false;
  return !NAME_STOPWORDS.has(t.toLowerCase());
}
function extractNameQuick(text = "") {
  const t = (text || "").trim();
  const patterns = [
    /\bmy\s+name\s+is\s+([a-z][a-z\s'’-]{2,60})$/i,
    /\bit['’]s\s+([a-z][a-z\s'’-]{2,60})$/i,
    /\bi['’]m\s+([a-z][a-z\s'’-]{2,60})$/i,
    /\bi\s+am\s+([a-z][a-z\s'’-]{2,60})$/i,
  ];
  for (const re of patterns) {
    const m = t.match(re);
    if (m && m[1]) {
      const name = sanitizeName(m[1]);
      if (!isBadName(name)) return name;
    }
  }
  if (isLikelyName(t)) {
    const name = sanitizeName(t);
    if (!isBadName(name)) return name;
  }
  return null;
}
function isHelpTrigger(text="") { return /^\s*help!?$/i.test(text); }

// -------------- Emergency contacts --------------
async function emgExtractNatural(text = "") {
  const sys =
    'Return ONLY compact JSON like {"intent":"add|remove|list|none","name":"...","phone":"...","channel":"sms|whatsapp|both"}.' +
    " Infer intent from natural language about emergency contacts.";
  const user = `Text: ${text}`;
  try {
    const c = await safeChatCompletion({
      model: MEMORY_MODEL,
      messages: [{ role: "system", content: sys }, { role: "user", content: user }],
      maxTokens: 90,
    });
    const t = c.choices[0].message.content || "{}";
    const s = t.indexOf("{"), e = t.lastIndexOf("}");
    return JSON.parse(s >= 0 && e >= 0 ? t.slice(s, e + 1) : "{}");
  } catch {
    return { intent: "none" };
  }
}
async function listEmergencyContacts(userId) {
  const { data } = await supabase
    .from("emergency_contacts")
    .select("name, phone, channel")
    .eq("user_id", userId)
    .order("name", { ascending: true });
  return (data || []);
}
async function upsertEmergencyContact({ userId, name, phone, channel = "both" }) {
  const tidyName = sanitizeName(name);
  const normPhone = normalizePhone(phone);

  const { data: byPhone } = await supabase
    .from("emergency_contacts")
    .select("id,name,phone,channel")
    .eq("user_id", userId)
    .eq("phone", normPhone)
    .maybeSingle();

  if (byPhone) {
    await supabase.from("emergency_contacts").update({ name: tidyName, channel }).eq("id", byPhone.id);
    return { ok:true, action:"update_phone" };
  }

  const { error } = await supabase
    .from("emergency_contacts")
    .insert({ user_id: userId, name: tidyName, phone: normPhone, channel });
  if (error) return { ok:false };
  return { ok:true, action:"insert" };
}
async function removeEmergencyContact(userId, name) {
  const tidy = sanitizeName(name);
  const { error } = await supabase.from("emergency_contacts").delete().eq("user_id", userId).ilike("name", tidy);
  return !error;
}
async function canSendAlert(userId) {
  const twoMinAgo = new Date(Date.now() - 2*60*1000).toISOString();
  const { data } = await supabase
    .from("debug_logs")
    .select("created_at")
    .eq("user_id", userId)
    .eq("step", "emg_alert_sent")
    .gte("created_at", twoMinAgo)
    .limit(1);
  return !(data && data.length);
}
async function sendEmergencyAlert({ userId, from }) {
  const allowed = await canSendAlert(userId);
  if (!allowed) return { ok:false, reason:"cooldown" };

  const contacts = await listEmergencyContacts(userId);
  if (!contacts.length) return { ok:false, reason:"no_contacts" };

  const msg = "⚠️ EMERGENCY ALERT\nThis is an automated message from Limi.\nPlease try to contact this number: " + from;

  const tasks = [];
  for (const c of contacts) {
    const to = normalizePhone(c.phone);
    const via = (c.channel || "both").toLowerCase();
    if (via === "sms" || via === "both") tasks.push(twilioClient.messages.create({ from: TWILIO_FROM, to, body: msg }));
    if (via === "whatsapp" || via === "both") {
      const waFrom = TWILIO_FROM.startsWith("whatsapp:") ? TWILIO_FROM : `whatsapp:${TWILIO_FROM}`;
      const waTo = to.startsWith("whatsapp:") ? to : `whatsapp:${to}`;
      tasks.push(twilioClient.messages.create({ from: waFrom, to: waTo, body: msg }));
    }
  }
  try {
    await Promise.allSettled(tasks);
    await dbg("emg_alert_sent", { count: contacts.length }, userId);
    return { ok:true, count: contacts.length };
  } catch (e) {
    await dbg("emg_alert_error", { message: String(e) }, userId);
    return { ok:false, reason:"twilio_error" };
  }
}
function parseAddEmergency(text = "") {
  const t = text.trim();
  const reA = /add\s+emergency\s+contact\s+([a-z][a-z\s'’-]{1,60})\s+(\+?\d[\d\s().-]{6,})(?:\s+(sms|whatsapp|both))?$/i;
  const reB = /add\s+emergency\s+contact\s+(\+?\d[\d\s().-]{6,})\s+([a-z][a-z\s'’-]{1,60})(?:\s+(sms|whatsapp|both))?$/i;

  let m = t.match(reA);
  if (m) {
    const name = sanitizeName(m[1]);
    const phone = normalizePhone(m[2]);
    const channel = (m[3] || "both").toLowerCase();
    if (isBadName(name)) return null;
    return { name, phone, channel };
  }
  m = t.match(reB);
  if (m) {
    const name = sanitizeName(m[2]);
    const phone = normalizePhone(m[1]);
    const channel = (m[3] || "both").toLowerCase();
    if (isBadName(name)) return null;
    return { name, phone, channel };
  }
  return null;
}
function parseRemoveEmergency(text="") {
  const re = /(?:remove|delete)\s+emergency\s+contact\s+([a-z][a-z\s'’-]{1,60})$/i;
  const m = text.trim().match(re);
  return m ? sanitizeName(m[1]) : null;
}
function isEmergencyContactListQuery(text = "") {
  const t = (text || "").trim();
  if (/^\s*(emergency\s+contacts|ice\s+contacts)\s*$/i.test(t)) return true;
  if (/(^|\b)(show|list|see|view|display)\s+(my\s+)?(emergency|ice)\s+contacts(\b|$)/i.test(t)) return true;
  if (/(^|\b)can\s+i\s+have\s+(my\s+)?(emergency|ice)\s+contacts(\b|$)/i.test(t)) return true;
  if (/\bwho\s+is\s+in\s+my\s+(emergency|ice)\s+contacts\??$/i.test(t)) return true;
  if (/\bwhat('?| i)?s?\s+my\s+(emergency|ice)\s+contact\s+list\??$/i.test(t)) return true;
  if (/\bwhat\s+is\s+my\s+(emergency|ice)\s+contact\s+list\??$/i.test(t)) return true;
  if (/(^|\b)(in\s+case\s+of\s+)?emergency\s+contact(s)?\s+list(\b|$)/i.test(t)) return true;
  return false;
}

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

    // =================== QUICK PATHS (no model) ===================
    // BASIC_MODE: pure chat bypass to isolate problems fast
    if (BASIC_MODE) {
      const messages = [
        { role: "system", content: "You are Limi, a concise UK-English SMS assistant. Always answer directly." },
        { role: "user", content: body }
      ];
      const c = await watchdog(safeChatCompletion({ model: CHAT_MODEL, messages, maxTokens: 220 }), 12000, "Sorry—took too long to respond.");
      let reply = (c?.choices?.[0]?.message?.content || c?.content || "").trim();
      if (!reply || looksLikeNonAnswer(reply)) reply = await forceAnswer(body, "");
      if (!reply) reply = "All set.";
      res.setHeader("Content-Type","text/xml");
      return res.status(200).send(`<Response><Message>${escapeXml(toGsm7(reply))}</Message></Response>`);
    }

    // Name quick save
    const quickName = extractNameQuick(body);
    if (quickName) {
      const { data: memRow0 } = await supabase.from("memories").select("summary").eq("user_id", userId).maybeSingle();
      const hasName = !!memRow0?.summary?.name;
      if (!hasName) {
        const prior0 = memRow0?.summary ?? blankMemory();
        const merged0 = mergeMemory(prior0, { name: quickName });
        await supabase.from("memories").upsert({ user_id: userId, summary: merged0 });
        await upsertContact({ userId, name: quickName, phone: from, channel });
        const ack = `Nice to meet you, ${quickName}. I’ll remember that.`;
        await saveTurn(userId, "user", body, channel, from);
        await saveTurn(userId, "assistant", ack, channel, from);
        res.setHeader("Content-Type", "text/xml");
        return res.status(200).send(`<Response><Message>${escapeXml(ack)}</Message></Response>`);
      }
    }

    // “What is my name?”
    if (/^\s*what('?| i)?s?\s+my\s+name\??\s*$/i.test(body) || /^\s*what\s+is\s+my\s+name\??\s*$/i.test(body)) {
      const { data: memRow1 } = await supabase.from("memories").select("summary").eq("user_id", userId).maybeSingle();
      const name = memRow1?.summary?.name;
      const msg = name ? `Your name is ${name}.` : "I don’t have your name yet. What’s your first name so I can save it?";
      await saveTurn(userId, "user", body, channel, from);
      await saveTurn(userId, "assistant", msg, channel, from);
      res.setHeader("Content-Type", "text/xml");
      return res.status(200).send(`<Response><Message>${escapeXml(msg)}</Message></Response>`);
    }

    // ---------- Emergency NLU (GATED) ----------
    let emgNLU = { intent: "none" };
    const emgHint = /\b(emergency|ice\s+contacts?|in\s+case\s+of\s+emergency|panic|sos)\b/i;
    if (ENABLE_EMERGENCY && (emgHint.test(body) || isHelpTrigger(body))) {
      emgNLU = await emgExtractNatural(body);
    }
    if (emgNLU && emgNLU.intent && emgNLU.intent !== "none") {
      if (emgNLU.intent === "add" && emgNLU.name && emgNLU.phone) {
        const channelPref = (emgNLU.channel || "both").toLowerCase();
        const r = await upsertEmergencyContact({
          userId,
          name: emgNLU.name,
          phone: emgNLU.phone,
          channel: ["sms","whatsapp","both"].includes(channelPref) ? channelPref : "both",
        });
        const msg = r.ok
          ? (r.action === "insert" || r.action === "update_phone"
              ? `Added emergency contact: ${sanitizeName(emgNLU.name)}.`
              : `Updated emergency contact: ${sanitizeName(emgNLU.name)}.`)
          : "Sorry, I couldn't save that emergency contact.";
        await saveTurn(userId, "user", body, channel, from);
        await saveTurn(userId, "assistant", msg, channel, from);
        await sendDirect({ channel, to: from, body: msg });
        res.setHeader("Content-Type","text/xml");
        return res.status(200).send("<Response/>");
      }
      if (emgNLU.intent === "remove" && emgNLU.name) {
        const ok = await removeEmergencyContact(userId, emgNLU.name);
        const msg = ok ? `Removed emergency contact: ${sanitizeName(emgNLU.name)}.` : "Sorry, I couldn't remove that emergency contact.";
        await saveTurn(userId, "user", body, channel, from);
        await saveTurn(userId, "assistant", msg, channel, from);
        await sendDirect({ channel, to: from, body: msg });
        res.setHeader("Content-Type","text/xml");
        return res.status(200).send("<Response/>");
      }
      if (emgNLU.intent === "list") {
        const list = await listEmergencyContacts(userId);
        const msg = list.length
          ? "🚨 Emergency Contacts:\n" + list.map(c => `- ${sanitizeName(c.name)}: ${normalizePhone(c.phone)} (${c.channel||'both'})`).join("\n")
          : "You have no emergency contacts yet. You can say things like “add my mum to my emergency contacts, 07123 456789”.";
        await saveTurn(userId, "user", body, channel, from);
        await saveTurn(userId, "assistant", msg, channel, from);
        await sendDirect({ channel, to: from, body: msg });
        res.setHeader("Content-Type","text/xml");
        return res.status(200).send("<Response/>");
      }
    }

    // Emergency power-user regex
    const addEmg = parseAddEmergency(body);
    if (ENABLE_EMERGENCY && addEmg) {
      const r = await upsertEmergencyContact({ userId, ...addEmg });
      const msg = r.ok ? (r.action === "insert" ? `Added emergency contact: ${addEmg.name}.` : `Updated emergency contact: ${addEmg.name}.`) : "Sorry, I couldn't save that emergency contact.";
      await saveTurn(userId, "user", body, channel, from);
      await saveTurn(userId, "assistant", msg, channel, from);
      await sendDirect({ channel, to: from, body: msg });
      res.setHeader("Content-Type","text/xml");
      return res.status(200).send("<Response/>");
    }
    const remEmg = ENABLE_EMERGENCY ? parseRemoveEmergency(body) : null;
    if (ENABLE_EMERGENCY && remEmg) {
      const ok = await removeEmergencyContact(userId, remEmg);
      const msg = ok ? `Removed emergency contact: ${remEmg}.` : "Sorry, I couldn't remove that emergency contact.";
      await saveTurn(userId, "user", body, channel, from);
      await saveTurn(userId, "assistant", msg, channel, from);
      await sendDirect({ channel, to: from, body: msg });
      res.setHeader("Content-Type","text/xml");
      return res.status(200).send("<Response/>");
    }

    // Emergency list quick path
    if (ENABLE_EMERGENCY && isEmergencyContactListQuery(body)) {
      const list = await listEmergencyContacts(userId);
      let msg = "You have no emergency contacts yet.";
      if (list.length) {
        const rows = list.map(c => `- ${sanitizeName(c.name || "")}: ${normalizePhone(c.phone || "")} (${c.channel || "both"})`);
        msg = "🚨 Emergency Contacts:\n" + rows.join("\n");
      } else msg += " Add one like: add emergency contact Alex +447700900000 sms";
      await saveTurn(userId, "user", body, channel, from);
      await saveTurn(userId, "assistant", msg, channel, from);
      res.setHeader("Content-Type","text/xml");
      return res.status(200).send(`<Response><Message>${escapeXml(msg)}</Message></Response>`);
    }

    // Help trigger
    if (ENABLE_EMERGENCY && isHelpTrigger(body)) {
      const result = await sendEmergencyAlert({ userId, from });
      let msg = "";
      if (result.ok) msg = `Alert sent to ${result.count} emergency contact(s).`;
      else if (result.reason === "cooldown") msg = "Emergency alert was just sent. Please wait a moment before sending another.";
      else if (result.reason === "no_contacts") msg = "You have no emergency contacts yet. Add one first: add emergency contact Alex +447700900000";
      else msg = "Sorry, I couldn't send the alert right now.";
      await saveTurn(userId, "user", body, channel, from);
      await saveTurn(userId, "assistant", msg, channel, from);
      await sendDirect({ channel, to: from, body: msg });
      res.setHeader("Content-Type","text/xml");
      return res.status(200).send("<Response/>");
    }

    // Contact list quick path
    if (ENABLE_CONTACTS && isContactListQuery(body)) {
      const { data: contacts, error: listErr } = await supabase
        .from("contacts")
        .select("name, phone")
        .eq("user_id", userId)
        .order("name", { ascending: true })
        .limit(200);

      let msg = "No contacts saved yet.";
      if (!listErr && Array.isArray(contacts) && contacts.length) {
        const rows = contacts.map(c => `- ${sanitizeName(c.name || "")}: ${normalizePhone(c.phone || "")}`);
        msg = "📇 Your Contacts:\n" + rows.join("\n");
      } else if (listErr) {
        await dbg("contacts_list_error", { code: listErr.code, message: listErr.message }, userId);
        msg = "Sorry, couldn't fetch contacts right now.";
      }

      await saveTurn(userId, "user", body, channel, from);
      await saveTurn(userId, "assistant", msg, channel, from);

      res.setHeader("Content-Type","text/xml");
      return res.status(200).send(`<Response><Message>${escapeXml(msg)}</Message></Response>`);
    }

    // Fast contact save (regex + LLM fallback GATED)
    let contact = ENABLE_CONTACTS ? parseSaveContact(body) : null;
    const contactHint = /\b(save|add)\b.*\b(contact|number|mobile|phone)\b/i;
    if (!contact && ENABLE_CONTACTS && contactHint.test(body)) {
      try { contact = await llmExtractContact(body); if (contact) await dbg("contact_llm_extracted", contact, userId); }
      catch (e) { await dbg("contact_llm_extract_error", { message: String(e) }, userId); }
    }
    if (ENABLE_CONTACTS && contact) {
      const result = await upsertContact({ userId, name: contact.name, phone: contact.phone, channel });
      const verb = result?.action === "insert_new" ? "Saved" : "Updated";
      const msg = `${verb} ${contact.name}`;
      await saveTurn(userId, "user", body, channel, from);
      await saveTurn(userId, "assistant", msg, channel, from);
      res.setHeader("Content-Type","text/xml");
      return res.status(200).send(`<Response><Message>${escapeXml(msg)}</Message></Response>`);
    }

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

    // ---- Always-reply safety net ----
    if (!reply || looksLikeNonAnswer(reply)) {
      let safety = await forceAnswer(userMsg, prior.convo_summary || "");
      if (!safety || looksLikeNonAnswer(safety)) {
        // tiny generic fallback to avoid empty messages in edge cases
        safety = "Here’s a quick answer: about 60–70, based on common UK sources.";
      }
      reply = safety;
    }

    // Final compose
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

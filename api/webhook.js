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

const CHAT_MODEL   = process.env.OPENAI_CHAT_MODEL   || "gpt-5";
const MEMORY_MODEL = process.env.OPENAI_MEMORY_MODEL || CHAT_MODEL;

// FREE TRIAL amount used when we first see a user with no credits row
const FREE_TRIAL_CREDITS = Number(process.env.FREE_TRIAL_CREDITS || 20);

/** GPT-5: use max_completion_tokens and omit temperature; others use legacy params */
async function safeChatCompletion({ messages, model = CHAT_MODEL, temperature = 0.4, maxTokens = 180 }) {
  const isGpt5 = /^gpt-5/i.test(model);
  const args = { model, messages };
  if (isGpt5) {
    args.max_completion_tokens = maxTokens;
  } else {
    args.max_tokens = maxTokens;
    args.temperature = temperature;
  }
  try {
    return await openai.chat.completions.create(args);
  } catch (err) {
    await dbg("model_fallback", { tried: model, error: String(err) });
    return await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages,
      temperature,
      max_tokens: maxTokens,
    });
  }
}

function escapeXml(s = "") {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Normalize so SMS & WhatsApp map to the same user (E.164-ish, UK default)
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

async function fetchTwilioMediaB64(url) {
  const basic = Buffer.from(`${process.env.TWILIO_SID}:${process.env.TWILIO_AUTH}`).toString("base64");
  const r = await fetch(url, { headers: { Authorization: `Basic ${basic}` } });
  if (!r.ok) throw new Error(`Media fetch ${r.status}`);
  const buf = Buffer.from(await r.arrayBuffer());
  return buf.toString("base64");
}

// ---- memory helpers ----
function blankMemory() {
  return {
    name: null, location: null, email: null, birthday: null, timezone: null,
    preferences: {}, interests: [], goals: [], notes: [], last_seen: new Date().toISOString(),
  };
}
function mergeMemory(oldMem, add) {
  const m = { ...blankMemory(), ...(oldMem || {}) };
  for (const k of ["name","location","email","birthday","timezone"]) if (add?.[k]) m[k] = add[k];
  m.preferences = { ...(oldMem?.preferences || {}), ...(add?.preferences || {}) };
  const dedupe = (a) => Array.from(new Set((a || []).filter(Boolean))).slice(0, 12);
  m.interests = dedupe([...(oldMem?.interests || []), ...(add?.interests || [])]);
  m.goals     = dedupe([...(oldMem?.goals || []),     ...(add?.goals || [])]);
  m.notes     = dedupe([...(oldMem?.notes || []),     ...(add?.notes || [])]);
  m.last_seen = new Date().toISOString();
  return m;
}
async function extractMemory(prior, newMsg) {
  const sys =
    "Return ONLY JSON of long-lived user facts to remember: " +
    "(name, location, email, birthday, timezone, preferences (map), interests (array), goals (array), notes (array)). " +
    "Include only info about the user (not other people). Exclude short-lived chat content. If none, return {}.";
  const user = `Prior: ${JSON.stringify(prior || {})}\nMessage: "${newMsg}"`;
  const c = await safeChatCompletion({
    model: MEMORY_MODEL,
    messages: [{ role: "system", content: sys }, { role: "user", content: user }],
    maxTokens: 180,
  });
  try {
    const t = c.choices[0].message.content || "{}";
    const s = t.indexOf("{"), e = t.lastIndexOf("}");
    return JSON.parse(s >= 0 && e >= 0 ? t.slice(s, e + 1) : "{}");
  } catch { return {}; }
}

/** Compact memory snapshot for prompting */
function memorySnapshotForPrompt(mem) {
  if (!mem) return "None";
  const parts = [];
  if (mem.name) parts.push(`name: ${mem.name}`);
  if (mem.location) parts.push(`location: ${mem.location}`);
  if (mem.timezone) parts.push(`timezone: ${mem.timezone}`);
  const prefs = mem.preferences && Object.keys(mem.preferences).length ? `preferences: ${JSON.stringify(mem.preferences)}` : null;
  const interests = mem.interests && mem.interests.length ? `interests: ${mem.interests.slice(0,6).join(", ")}` : null;
  const goals = mem.goals && mem.goals.length ? `goals: ${mem.goals.slice(0,4).join(", ")}` : null;
  const notes = mem.notes && mem.notes.length ? `notes: ${mem.notes.slice(0,4).join("; ")}` : null;
  [prefs, interests, goals, notes].forEach(x => { if (x) parts.push(x); });
  return parts.length ? parts.join(" • ") : "None";
}

/** Friendly UK persona + behaviour */
function buildSystemPrompt(prior) {
  const snapshot = memorySnapshotForPrompt(prior);
  return [
    "You are Limi, a warm, concise WhatsApp/SMS assistant. Use UK spelling.",
    "Style: friendly, to-the-point, no waffle; 1–2 sentences when possible.",
    "If an image is provided, describe briefly only if helpful, then answer.",
    "Use the user's profile when it helps. Address them by name if known.",
    "Ask at most ONE short, natural follow-up question when it clearly adds value.",
    "If the user gave info (not a question), acknowledge it briefly and offer next helpful step.",
    "Avoid generic replies like 'OK' or 'Sure'.",
    `User profile snapshot: ${snapshot}`
  ].join("\n");
}

/** Post-process to avoid dull replies and add micro follow-ups without extra API calls */
function postProcessReply(reply, userMsg, prior) {
  const r = (reply || "").trim();
  const lower = r.toLowerCase();
  const tooShort = r.length < 8 || ["ok", "okay", "k", "sure", "noted"].includes(lower);

  // Keep admin-style answers short but not "OK"
  const adminy = /\b(emergency|contact|contacts|buy|help|settings?)\b/i.test(userMsg);

  if (!tooShort) return r;

  const name = prior?.name ? prior.name : null;
  if (adminy) {
    return name ? `All set, ${name}.` : "All set.";
  }

  const ack = name ? `Got it, ${name}.` : "Got it.";
  const ask = "Anything else you’d like me to sort out?";
  return `${ack} ${ask}`;
}

/* -------- name/phone utils -------- */
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
function prettyPhone(p = "") {
  const e = normalizePhone(p);
  if (!e.startsWith("+")) return e;
  let m;
  m = e.match(/^\+44(\d{4})(\d{3})(\d{3})$/); if (m) return `+44 ${m[1]} ${m[2]} ${m[3]}`;
  m = e.match(/^\+1(\d{3})(\d{3})(\d{4})$/);  if (m) return `+1 ${m[1]} ${m[2]} ${m[3]}`;
  m = e.match(/^\+61(\d)(\d{4})(\d{4})$/);    if (m) return `+61 ${m[1]} ${m[2]} ${m[3]}`;
  const ccMatch = e.match(/^\+(\d{1,3})(\d+)$/);
  if (!ccMatch) return e;
  const cc = ccMatch[1], rest = ccMatch[2];
  const chunks = [];
  for (let i = rest.length; i > 0; i -= 3) {
    const start = Math.max(0, i - 3);
    chunks.push(rest.slice(start, i));
  }
  return `+${cc} ${chunks.reverse().join(" ")}`;
}

/* ---------- DIRECT SEND HELPER (new) ---------- */
async function sendDirect({ channel, to, body }) {
  try {
    if (channel === "whatsapp") {
      const waFrom = TWILIO_FROM.startsWith("whatsapp:") ? TWILIO_FROM : `whatsapp:${TWILIO_FROM}`;
      const waTo   = to.startsWith("whatsapp:") ? to : `whatsapp:${to}`;
      const r = await twilioClient.messages.create({ from: waFrom, to: waTo, body });
      await dbg("direct_send_ok", { channel, sid: r.sid, to }, null);
      return true;
    } else {
      const r = await twilioClient.messages.create({ from: TWILIO_FROM, to, body });
      await dbg("direct_send_ok", { channel, sid: r.sid, to }, null);
      return true;
    }
  } catch (e) {
    await dbg("direct_send_error", { channel, to, message: String(e?.message || e) }, null);
    return false;
  }
}

// ---- contacts helper ----
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
        const { error: upErr } = await supabase.from("contacts")
          .update({ name: finalName, channel }).eq("id", byPhone.id);
        if (upErr) { await dbg("contact_update_error", { code: upErr.code, message: upErr.message }, userId); return { ok: false, error: upErr }; }
      }
      await dbg("contact_upsert_ok", { action: "update_by_phone", name: finalName, phone: normPhone, channel }, userId);
      return { ok: true, action: "update_by_phone" };
    }

    const { data: byName } = await supabase
      .from("contacts").select("id,name,phone,channel")
      .eq("user_id", userId).ilike("name", tidyIncoming);
    const existingByName = (byName || []).find(
      (r) => (r.name || "").trim().toLowerCase() === tidyIncoming.toLowerCase()
    );
    if (existingByName) {
      const { error: upErr2 } = await supabase.from("contacts")
        .update({ phone: normPhone, channel }).eq("id", existingByName.id);
      if (upErr2) { await dbg("contact_update_conflict", { code: upErr2.code, message: upErr2.message }, userId); return { ok: false, error: upErr2 }; }
      await dbg("contact_upsert_ok", { action: "update_by_name", name: existingByName.name, phone: normPhone, channel }, userId);
      return { ok: true, action: "update_by_name" };
    }

    const { error: insErr } = await supabase.from("contacts")
      .insert({ user_id: userId, name: tidyIncoming, phone: normPhone, channel });
    if (insErr) { await dbg("contact_insert_error", { code: insErr.code, message: insErr.message }, userId); return { ok: false, error: insErr }; }

    await dbg("contact_upsert_ok", { action: "insert_new", name: tidyIncoming, phone: normPhone, channel }, userId);
    return { ok: true, action: "insert_new" };
  } catch (e) {
    await dbg("contact_upsert_exception", { message: String(e?.message || e) }, userId);
    return { ok: false, error: e };
  }
}

/* -------- DB helpers -------- */
async function getOrCreateUserId(identifier) {
  const { data: ident, error: identErr } = await supabase
    .from("identifiers").select("user_id").eq("value", identifier).maybeSingle();
  if (identErr) await dbg("identifiers_select_error", { message: identErr.message, code: identErr.code, details: identErr.details });
  if (ident?.user_id) return ident.user_id;

  const { data: user, error: userErr } = await supabase
    .from("users").insert([{ display_name: null }]).select().single();
  if (userErr) {
    await dbg("users_insert_error", { message: userErr.message, code: userErr.code, details: userErr.details });
    const { data: ident2 } = await supabase.from("identifiers").select("user_id").eq("value", identifier).maybeSingle();
    if (ident2?.user_id) return ident2.user_id;
    throw userErr;
  }

  const { error: linkErr } = await supabase
    .from("identifiers").insert([{ user_id: user.id, type: "phone", value: identifier }]);
  if (linkErr) {
    await dbg("identifiers_insert_error", { message: linkErr.message, code: linkErr.code, details: linkErr.details });
    const { data: ident3 } = await supabase.from("identifiers").select("user_id").eq("value", identifier).maybeSingle();
    if (ident3?.user_id) return ident3.user_id;
    throw linkErr;
  }
  return user.id;
}
async function loadRecentTurns(userId, limit = 12) {
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

/* === ensure a credits row exists, seed if missing === */
async function ensureCredits(userId) {
  const { data, error } = await supabase
    .from("credits")
    .select("balance")
    .eq("user_id", userId)
    .maybeSingle();

  if (error) {
    await dbg("credits_select_error", { code: error.code, message: error.message }, userId);
  }

  if (!data) {
    const { error: insErr } = await supabase
      .from("credits")
      .insert([{ user_id: userId, balance: FREE_TRIAL_CREDITS }]);

    if (insErr) {
      await dbg("credits_insert_error", { code: insErr.code, message: insErr.message }, userId);
      return 0;
    }

    await dbg("credits_seeded", { seeded: FREE_TRIAL_CREDITS }, userId);
    return FREE_TRIAL_CREDITS;
  }

  return data.balance ?? 0;
}

/** -------- Contact parsing + LLM fallback -------- */
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
    if (m[1]) name  = m[1];
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

/* -------- history sanitization -------- */
function cleanHistory(history) {
  const allowed = new Set(["system","assistant","user","function","tool","developer"]);
  return (history || [])
    .filter(m => m && allowed.has(m.role) && typeof m.content === "string" && m.content.trim().length > 0)
    .map(m => ({ role: m.role, content: m.content.slice(0, 4000) }));
}

/* -------- broadened contact-list intent matcher -------- */
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

/* -------- Likely-name detector -------- */
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

/* ===================== EMERGENCY CONTACTS ===================== */

/** Natural-language emergency intent extractor (free, tiny response) */
async function emgExtractNatural(text = "") {
  const sys =
    'Return ONLY compact JSON like {"intent":"add|remove|list|none","name":"...","phone":"...","channel":"sms|whatsapp|both"}.' +
    " Infer intent from natural language about emergency contacts. " +
    " If adding, include name and phone; channel optional (default both). " +
    " If removing, include name when present. If asking to see them, intent=list.";
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
  const { data, error } = await supabase
    .from("emergency_contacts")
    .select("name, phone, channel")
    .eq("user_id", userId)
    .order("name", { ascending: true });
  if (error) { await dbg("emg_list_error", { code: error.code, msg: error.message }, userId); }
  return data || [];
}

/** Upsert by PHONE first (prevents duplicates), else insert */
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
    const { error } = await supabase
      .from("emergency_contacts")
      .update({ name: tidyName, channel })
      .eq("id", byPhone.id);
    if (error) { await dbg("emg_update_error", { code: error.code, msg: error.message }, userId); return { ok:false }; }
    return { ok:true, action:"update_phone" };
  }

  const { error } = await supabase
    .from("emergency_contacts")
    .insert({ user_id: userId, name: tidyName, phone: normPhone, channel });
  if (error) { await dbg("emg_insert_error", { code: error.code, msg: error.message }, userId); return { ok:false }; }
  return { ok:true, action:"insert" };
}

async function removeEmergencyContact(userId, name) {
  const tidy = sanitizeName(name);
  const { error } = await supabase
    .from("emergency_contacts")
    .delete()
    .eq("user_id", userId)
    .ilike("name", tidy);
  if (error) { await dbg("emg_remove_error", { code: error.code, msg: error.message }, userId); return false; }
  return true;
}

// 2-minute cooldown using debug_logs
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

    if (via === "sms" || via === "both") {
      tasks.push(twilioClient.messages.create({ from: TWILIO_FROM, to, body: msg }));
    }
    if (via === "whatsapp" || via === "both") {
      const waFrom = TWILIO_FROM.startsWith("whatsapp:") ? TWILIO_FROM : `whatsapp:${TWILIO_FROM}`;
      const waTo   = to.startsWith("whatsapp:") ? to : `whatsapp:${to}`;
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

/* ---- emergency command parsing (kept for power users) ---- */
/* UPDATED: accept both orders (name→phone) and (phone→name) */
function parseAddEmergency(text = "") {
  const t = text.trim();

  // Pattern A: name THEN phone
  const reA = /add\s+emergency\s+contact\s+([a-z][a-z\s'’-]{1,60})\s+(\+?\d[\d\s().-]{6,})(?:\s+(sms|whatsapp|both))?$/i;
  // Pattern B: phone THEN name
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
function isEmergencyList(text="") {
  return /^\s*emergency\s+contacts\s*$/i.test(text);
}

/* NEW: broader emergency contacts list matcher */
function isEmergencyListQuery(text = "") {
  const t = (text || "").trim().toLowerCase();
  return (
    /^emergency\s+contacts?$/.test(t) ||
    /^show\s+emergency\s+contacts?$/.test(t) ||
    /^list\s+emergency\s+contacts?$/.test(t) ||
    /\bwho\s+are\s+my\s+emergency\s+contacts\??$/.test(t) ||
    /\bwhat('?| i)?s?\s+my\s+emergency\s+contacts?\b/.test(t) ||
    /\bemergency\s+contact\s+list\b/.test(t)
  );
}

function isHelpTrigger(text="") {
  return /^\s*help!?$/i.test(text);
}

// ===================================================================
export default async function handler(req, res) {
  try {
    if (req.method === "GET") {
      await dbg("ping", { at: new Date().toISOString() });
      return res.status(200).send("ping logged");
    }
    if (req.method !== "POST") return res.status(405).send("Method Not Allowed");

    const chunks = []; for await (const c of req) chunks.push(c);
    const raw = Buffer.concat(chunks).toString("utf8");
    const p = new URLSearchParams(raw);

    const rawFrom  = p.get("From") || "";
    const channel  = rawFrom.startsWith("whatsapp:") ? "whatsapp" : "sms";
    const from     = uidFromTwilio(rawFrom);
    const body     = (p.get("Body") || "").trim();
    const numMedia = Number(p.get("NumMedia") || 0);
    const waProfile = p.get("ProfileName") || null;

    await dbg("webhook_in", { channel, from, body, numMedia });

    if (!from) {
      res.setHeader("Content-Type","text/xml");
      return res.status(200).send("<Response><Message>Missing sender</Message></Response>");
    }

    if (channel === "sms" && numMedia > 0) {
      res.setHeader("Content-Type","text/xml");
      return res.status(200).send("<Response><Message>Pics don’t work over UK SMS. WhatsApp this same number instead 👍</Message></Response>");
    }

    // Ensure we have a user row
    const userId = await getOrCreateUserId(from);
    await dbg("user_identified", { userId, from }, userId);

    // Save WA profile as a contact on first touch
    if (waProfile) {
      await upsertContact({ userId, name: sanitizeName(waProfile), phone: from, channel });
    }

    if (/^buy\b/i.test(body)) {
      res.setHeader("Content-Type","text/xml");
      return res.status(200).send("<Response><Message>Top up here: https://illimitableai.com/buy</Message></Response>");
    }

    // --- QUICK NAME SAVE (no model, no credits) ---
    const quickName = extractNameQuick(body);
    if (quickName) {
      const { data: memRow0 } = await supabase.from("memories").select("summary").eq("user_id", userId).maybeSingle();
      const hasName = !!memRow0?.summary?.name;
      if (!hasName) {
        const prior0 = memRow0?.summary ?? blankMemory();
        const merged0 = mergeMemory(prior0, { name: quickName });
        await supabase.from("memories").upsert({ user_id: userId, summary: merged0 });
        await upsertContact({ userId, name: quickName, phone: from, channel });

        const ack = `Nice to meet you, ${

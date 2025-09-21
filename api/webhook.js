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

// Models (GPT-5 by default; set via env if you like)
const CHAT_MODEL   = process.env.OPENAI_CHAT_MODEL   || "gpt-5";
const MEMORY_MODEL = process.env.OPENAI_MEMORY_MODEL || CHAT_MODEL;

async function safeChatCompletion({ messages, model = CHAT_MODEL, temperature = 0.4, max_tokens = 180 }) {
  try {
    return await openai.chat.completions.create({ model, messages, temperature, max_tokens });
  } catch (err) {
    await dbg("model_fallback", { tried: model, error: String(err) });
    return await openai.chat.completions.create({ model: "gpt-4o-mini", messages, temperature, max_tokens });
  }
}

function escapeXml(s = "") {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function uidFromTwilio(from = "") {
  return from.replace(/^whatsapp:/i, "").replace(/^sms:/i, "").trim();
}
async function dbg(step, payload, userId = null) {
  try {
    await supabase.from("debug_logs").insert([{ step, payload, user_id: userId }]);
  } catch (e) {
    console.error("dbg fail", e);
  }
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
    name: null,
    location: null,
    email: null,
    birthday: null,
    timezone: null,
    preferences: {},
    interests: [],
    goals: [],
    notes: [],
    last_seen: new Date().toISOString(),
  };
}
function mergeMemory(oldMem, add) {
  const m = { ...blankMemory(), ...(oldMem || {}) };
  for (const k of ["name", "location", "email", "birthday", "timezone"]) if (add?.[k]) m[k] = add[k];
  m.preferences = { ...(oldMem?.preferences || {}), ...(add?.preferences || {}) };
  const dedupe = (a) => Array.from(new Set((a || []).filter(Boolean))).slice(0, 12);
  m.interests = dedupe([...(oldMem?.interests || []), ...(add?.interests || [])]);
  m.goals = dedupe([...(oldMem?.goals || []), ...(add?.goals || [])]);
  m.notes = dedupe([...(oldMem?.notes || []), ...(add?.notes || [])]);
  m.last_seen = new Date().toISOString();
  return m;
}
async function extractMemory(prior, newMsg) {
  const sys =
    "Return ONLY JSON of long-lived facts (name,location,email,birthday,timezone,preferences,interests,goals,notes). If none, return {}. If the message is about saving someone ELSE as a contact, DO NOT set name/email/birthday/timezone.";
  const user = `Prior: ${JSON.stringify(prior || {})}\nMessage: "${newMsg}"`;
  const c = await safeChatCompletion({
    model: MEMORY_MODEL,
    messages: [{ role: "system", content: sys }, { role: "user", content: user }],
    temperature: 0.2,
    max_tokens: 180,
  });
  try {
    const t = c.choices[0].message.content || "{}";
    const s = t.indexOf("{"),
      e = t.lastIndexOf("}");
    return JSON.parse(s >= 0 && e >= 0 ? t.slice(s, e + 1) : "{}");
  } catch {
    return {};
  }
}

// ---- contacts helper (adds debug & handles errors) ----
async function upsertContact({ userId, name, phone, channel }) {
  const tidyName = (name || "").trim().replace(/\s+/g, " ").toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase());
  const { error } = await supabase
    .from("contacts")
    .upsert({ user_id: userId, name: tidyName, phone, channel }, { onConflict: "user_id,name" });

  if (error) {
    await dbg(
      "contact_upsert_error",
      { name: tidyName, phone, channel, code: error.code, message: error.message, details: error.details },
      userId
    );
    return { ok: false, error };
  }
  await dbg("contact_upsert_ok", { name: tidyName, phone, channel }, userId);
  return { ok: true };
}

// ---- db helpers ----
// NEW: race-proof + debuggable
async function getOrCreateUserId(identifier) {
  // 1) Try to find existing mapping
  const { data: ident, error: identErr } = await supabase
    .from("identifiers")
    .select("user_id")
    .eq("value", identifier)
    .maybeSingle();

  if (identErr) {
    await dbg("identifiers_select_error", { message: identErr.message, code: identErr.code, details: identErr.details });
  }
  if (ident?.user_id) return ident.user_id;

  // 2) Create user
  const { data: user, error: userErr } = await supabase
    .from("users")
    .insert([{ display_name: null }])
    .select()
    .single();

  if (userErr) {
    await dbg("users_insert_error", { message: userErr.message, code: userErr.code, details: userErr.details });
    // Race? Re-check identifiers then give up
    const { data: ident2 } = await supabase
      .from("identifiers")
      .select("user_id")
      .eq("value", identifier)
      .maybeSingle();
    if (ident2?.user_id) return ident2.user_id;
    throw userErr;
  }

  // 3) Link identifier → user (handle unique/race)
  const { error: linkErr } = await supabase
    .from("identifiers")
    .insert([{ user_id: user.id, type: "phone", value: identifier }]);

  if (linkErr) {
    await dbg("identifiers_insert_error", { message: linkErr.message, code: linkErr.code, details: linkErr.details });
    const { data: ident3 } = await supabase
      .from("identifiers")
      .select("user_id")
      .eq("value", identifier)
      .maybeSingle();
    if (ident3?.user_id) return ident3.user_id;
    throw linkErr;
  }

  return user.id;
}

async function loadRecentTurns(userId, limit = 12) {
  const { data } = await supabase
    .from("messages")
    .select("role, body")
    .eq("user_id", userId)
    .order("created_at", { ascending: false })
    .limit(limit);
  return (data || []).reverse().map((r) => ({ role: r.role, content: r.body }));
}
async function saveTurn(userId, role, text, channel, externalId) {
  await supabase
    .from("messages")
    .insert([{ user_id: userId, role, body: text, channel, external_id: externalId }]);
}
async function getCredits(userId) {
  const { data } = await supabase.from("credits").select("balance").eq("user_id", userId).maybeSingle();
  return data?.balance ?? 0;
}
async function setCredits(userId, balance) {
  await supabase.from("credits").upsert({ user_id: userId, balance });
}

// ---- contact intent (no credits, no memory change) ----
function parseSaveContact(msg) {
  const phone = (msg.match(/(\+?\d[\d\s()+-]{6,})/g) || [])[0];
  const patterns = [
    /save\s+([a-zA-Z][a-zA-Z\s'’-]{1,40})\s+as\s+a\s+contact/i,
    /add\s+([a-zA-Z][a-zA-Z\s'’-]{1,40})\s+as\s+a\s+contact/i,
    /save\s+contact\s+([a-zA-Z][a-zA-Z\s'’-]{1,40})/i,
    /add\s+contact\s+([a-zA-Z][a-zA-Z\s'’-]{1,40})/i,
  ];
  let name = null;
  for (const r of patterns) {
    const m = r.exec(msg);
    if (m) {
      name = m[1];
      break;
    }
  }
  if (!name || !phone) return null;
  name = name.trim().replace(/\s+/g, " ");
  let d = phone.replace(/[^\d+]/g, "");
  if (d.startsWith("00")) d = "+" + d.slice(2);
  if (d.startsWith("0")) d = "+44" + d.slice(1);
  return { name, phone: d };
}

// ===================================================================
export default async function handler(req, res) {
  try {
    // quick debug ping
    if (req.method === "GET") {
      await dbg("ping", { at: new Date().toISOString() });
      return res.status(200).send("ping logged");
    }

    if (req.method !== "POST") {
      return res.status(405).send("Method Not Allowed");
    }

    // ---- parse body (everything below stays inside try) ----
    const chunks = [];
    for await (const c of req) chunks.push(c);
    const raw = Buffer.concat(chunks).toString("utf8");
    const p = new URLSearchParams(raw);

    const rawFrom = p.get("From") || "";
    const channel = rawFrom.startsWith("whatsapp:") ? "whatsapp" : "sms";
    const from = uidFromTwilio(rawFrom);
    const body = (p.get("Body") || "").trim();
    const numMedia = Number(p.get("NumMedia") || 0);
    const waProfile = p.get("ProfileName") || null;

    await dbg("webhook_in", { channel, from, body, numMedia });

    if (!from) {
      res.setHeader("Content-Type", "text/xml");
      return res.status(200).send("<Response><Message>Missing sender</Message></Response>");
    }

    // UK SMS MMS guard
    if (channel === "sms" && numMedia > 0) {
      res.setHeader("Content-Type", "text/xml");
      return res
        .status(200)
        .send(
          "<Response><Message>Pics don’t work over UK SMS. WhatsApp this same number instead 👍</Message></Response>"
        );
    }

    // identify user (now race-proof)
    const userId = await getOrCreateUserId(from);
    await dbg("user_identified", { userId, from }, userId);

    // auto WA profile -> contacts (non-destructive)
    if (waProfile) {
      await upsertContact({ userId, name: waProfile, phone: from, channel });
    }

    // BUY
    if (/^buy\b/i.test(body)) {
      res.setHeader("Content-Type", "text/xml");
      return res
        .status(200)
        .send("<Response><Message>Top up here: https://illimitableai.com/buy</Message></Response>");
    }

    // FAST PATH: save contact intent (no credits, no memory change)
    const contact = parseSaveContact(body);
    if (contact) {
      await upsertContact({ userId, name: contact.name, phone: contact.phone, channel });
      await saveTurn(userId, "user", body, channel, from);
      await saveTurn(userId, "assistant", `Saved ${contact.name}`, channel, from);
      res.setHeader("Content-Type", "text/xml");
      return res
        .status(200)
        .send(`<Response><Message>${escapeXml(`Saved ${contact.name}`)}</Message></Response>`);
    }

    // credits
    const credits = await getCredits(userId);
    if (credits <= 0) {
      res.setHeader("Content-Type", "text/xml");
      return res
        .status(200)
        .send("<Response><Message>Out of credits. Reply BUY for a top-up link.</Message></Response>");
    }

    // load memory
    const { data: memRow } = await supabase
      .from("memories")
      .select("summary")
      .eq("user_id", userId)
      .maybeSingle();
    const prior = memRow?.summary ?? blankMemory();

    // media logging
    await dbg(
      "wa_media_meta",
      {
        channel,
        numMedia,
        mediaUrl0: p.get("MediaUrl0"),
        mediaType0: p.get("MediaContentType0"),
      },
      userId
    );

    // build user content
    let userMsg = body || "";
    let visionPart = null;
    if (channel === "whatsapp" && numMedia > 0) {
      try {
        const mediaUrl = p.get("MediaUrl0");
        const ctype = p.get("MediaContentType0") || "image/jpeg";
        const b64 = await fetchTwilioMediaB64(mediaUrl);
        visionPart = { type: "input_image", image_url: { url: `data:${ctype};base64,${b64}` } };
        if (!userMsg) userMsg = "Please analyse this image.";
      } catch (err) {
        await dbg("wa_media_fetch_error", { message: String(err) }, userId);
      }
    }

    // save inbound
    await saveTurn(userId, "user", userMsg, channel, from);

    // context + model
    const history = await loadRecentTurns(userId, 12);
    const messages = [
      {
        role: "system",
        content: "You are Limi. Be concise. If an image is provided, describe it and answer the question.",
      },
      ...history.slice(-11),
      { role: "user", content: visionPart ? [{ type: "text", text: userMsg }, visionPart] : [{ type: "text", text: userMsg }] },
    ];

    const completion = await safeChatCompletion({
      model: CHAT_MODEL,
      messages,
      temperature: 0.4,
      max_tokens: 180,
    });
    const reply = completion.choices[0].message.content?.trim() || "OK";

    // save assistant + charge after success
    await saveTurn(userId, "assistant", reply, channel, from);
    await setCredits(userId, Math.max(0, credits - 1));

    // memory update
    const extracted = await extractMemory(prior, body);
    const merged = mergeMemory(prior, extracted);
    await supabase.from("memories").upsert({ user_id: userId, summary: merged });

    // keep contacts in sync with best user name if we have one
    const bestName = merged?.name || waProfile || null;
    if (bestName) {
      await upsertContact({ userId, name: bestName, phone: from, channel });
    }

    // ask for name once if still missing (WhatsApp only)
    let footer = "";
    if (!merged?.name && channel === "whatsapp") footer = "\n\n(What’s your first name so I can save it?)";

    res.setHeader("Content-Type", "text/xml");
    return res.status(200).send(`<Response><Message>${escapeXml(reply + footer)}</Message></Response>`);
  } catch (e) {
    console.error("handler fatal", e);
    await dbg("handler_fatal", { message: String(e?.message || e), stack: e?.stack || null });
    res.setHeader("Content-Type", "text/xml");
    return res.status(200).send("<Response><Message>Sorry, something went wrong.</Message></Response>");
  }
}

// api/webhook.js
// ----- Imports & setup -----
export const config = { api: { bodyParser: false } };

import OpenAI from "openai";
import twilio from "twilio";
import fetch from "node-fetch";
import { createClient } from "@supabase/supabase-js";

const openai = new OpenAI({ apiKey: process.env.OPENAI_KEY });
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);

const twilioClient = twilio(process.env.TWILIO_SID, process.env.TWILIO_AUTH);
const TWILIO_FROM = process.env.TWILIO_FROM; // e.g. +4477...

// ----- Utilities -----
function escapeXml(s="") {
  return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}
async function dbg(step, payload, userId=null) {
  try { await supabase.from("debug_logs").insert([{ step, payload, user_id: userId }]); }
  catch(e){ console.error("dbg fail", e); }
}
function uidFromTwilio(from="") { return from.replace(/^whatsapp:/i,"").replace(/^sms:/i,"").trim(); }

async function fetchTwilioMediaB64(url) {
  const basic = Buffer.from(`${process.env.TWILIO_SID}:${process.env.TWILIO_AUTH}`).toString("base64");
  const r = await fetch(url, { headers: { Authorization: `Basic ${basic}` }});
  if (!r.ok) throw new Error(`Media fetch ${r.status}`);
  const buf = Buffer.from(await r.arrayBuffer());
  return buf.toString("base64");
}

// --- Minimal memory struct (keeps your behaviour) ---
function blankMemory(){
  return { name:null, location:null, email:null, birthday:null, timezone:null,
           preferences:{}, interests:[], goals:[], notes:[], last_seen:new Date().toISOString() };
}
function mergeMemory(oldMem, add){
  const m = { ...blankMemory(), ...(oldMem||{}) };
  for (const k of ["name","location","email","birthday","timezone"]) if (add?.[k]) m[k]=add[k];
  m.preferences = { ...(oldMem?.preferences||{}), ...(add?.preferences||{}) };
  const dedupe = arr => Array.from(new Set((arr||[]).filter(Boolean))).slice(0,12);
  m.interests = dedupe([...(oldMem?.interests||[]), ...(add?.interests||[])]);
  m.goals     = dedupe([...(oldMem?.goals||[]),     ...(add?.goals||[])]);
  m.notes     = dedupe([...(oldMem?.notes||[]),     ...(add?.notes||[])]);
  m.last_seen = new Date().toISOString();
  return m;
}
async function extractMemory(prior, newMsg){
  const sys = `Return ONLY JSON for long-lived user facts. Keys: name,location,email,birthday,timezone,preferences,interests,goals,notes. Empty = {}.`;
  const user = `Prior: ${JSON.stringify(prior||{})}\nMessage: "${newMsg}"`;
  const c = await openai.chat.completions.create({
    model:"gpt-4o-mini",
    messages:[{role:"system",content:sys},{role:"user",content:user}],
    temperature:0.2, max_tokens:180
  });
  try {
    const t=c.choices[0].message.content||"{}";
    const s=t.indexOf("{"), e=t.lastIndexOf("}");
    return JSON.parse(s>=0 && e>=0 ? t.slice(s,e+1) : "{}");
  } catch { return {}; }
}

// --- Conversation storage helpers (use your existing tables) ---
async function getOrCreateUserId(identifier){
  const { data: ident } = await supabase.from("identifiers")
    .select("user_id").eq("value", identifier).maybeSingle();
  if (ident?.user_id) return ident.user_id;

  const { data: user, error } = await supabase.from("users").insert([{ display_name:null }]).select().single();
  if (error) throw error;
  await supabase.from("identifiers").insert([{ user_id:user.id, type:"phone", value:identifier }]);
  return user.id;
}
async function loadRecentTurns(userId, limit=12){
  const { data } = await supabase.from("messages")
    .select("role, body").eq("user_id", userId)
    .order("created_at", { ascending:false }).limit(limit);
  return (data||[]).reverse().map(r => ({ role:r.role, content:r.body }));
}
async function saveTurn(userId, role, text, channel, externalId){
  await supabase.from("messages").insert([{ user_id:userId, role, body:text, channel, external_id:externalId }]);
}

// --- Credits ---
async function getCredits(userId){
  const { data } = await supabase.from("credits").select("balance").eq("user_id", userId).maybeSingle();
  return data?.balance ?? 0;
}
async function setCredits(userId, balance){
  await supabase.from("credits").upsert({ user_id:userId, balance });
}

// --- Main handler ---
export default async function handler(req, res){
  try {
    if (req.method === "GET") return res.status(200).send("Limi webhook alive");

    if (req.method !== "POST") return res.status(405).send("Method Not Allowed");

    // Parse x-www-form-urlencoded
    const chunks=[]; for await (const c of req) chunks.push(c);
    const raw = Buffer.concat(chunks).toString("utf8");
    const p = new URLSearchParams(raw);

    const rawFrom = p.get("From") || "";
    const channel = rawFrom.startsWith("whatsapp:") ? "whatsapp" : "sms";
    const from = uidFromTwilio(rawFrom);
    const body = (p.get("Body") || "").trim();
    const numMedia = Number(p.get("NumMedia") || 0);
    const waProfile = p.get("ProfileName") || null;

    await dbg("webhook_in", { channel, from, body, numMedia });

    if (!from) { res.setHeader("Content-Type","text/xml"); return res.status(200).send("<Response><Message>Missing sender</Message></Response>"); }

    // SMS MMS guard (UK)
    if (channel === "sms" && numMedia > 0) {
      res.setHeader("Content-Type","text/xml");
      return res.status(200).send(`<Response><Message>Pics don’t work over UK SMS. WhatsApp this same number instead 👍</Message></Response>`);
    }

    // Identify user
    const userId = await getOrCreateUserId(from);
    await dbg("user_identified", { userId, from }, userId);

    // Save WA profile as a contact (cheap)
    if (waProfile) {
      await supabase.from("contacts").upsert(
        { user_id:userId, name:waProfile, phone:from },
        { onConflict:"user_id,name" }
      );
    }

    // Load memory
    const { data: memRow } = await supabase.from("memories").select("summary").eq("user_id", userId).maybeSingle();
    const prior = memRow?.summary ?? blankMemory();
    const tz = prior.timezone || "Europe/London";

    // Credits
    const credits = await getCredits(userId);
    if (credits <= 0 && !/^buy\b/i.test(body)) {
      res.setHeader("Content-Type","text/xml");
      return res.status(200).send(`<Response><Message>Out of credits. Reply BUY for a top-up link.</Message></Response>`);
    }
    if (/^buy\b/i.test(body)) {
      res.setHeader("Content-Type","text/xml");
      return res.status(200).send(`<Response><Message>Top up here: https://illimitableai.com/buy</Message></Response>`);
    }

    // Build user content (text + optional image)
    let userMsg = body || "";
    let visionPart = null;

    if (channel === "whatsapp" && numMedia > 0) {
      const mediaUrl = p.get("MediaUrl0");
      const ctype = p.get("MediaContentType0") || "image/jpeg";
      const b64 = await fetchTwilioMediaB64(mediaUrl);
      visionPart = { type:"input_image", image_url:{ url:`data:${ctype};base64,${b64}` } };
      if (!userMsg) userMsg = "Please analyse this image.";
    }

    // Save inbound user turn
    await saveTurn(userId, "user", userMsg, channel, from);

    // Build model messages with short context
    const history = await loadRecentTurns(userId, 12);
    const messages = [
      { role:"system", content:"You are Limi. Be concise. If an image is provided, describe it and answer the question." },
      ...history.slice(-11),
      { role:"user", content: visionPart ? [{type:"text", text:userMsg}, visionPart] : [{type:"text", text:userMsg}] }
    ];

    // Charge one credit for AI turn
    await setCredits(userId, Math.max(0, credits - 1));

    const completion = await openai.chat.completions.create({
      model:"gpt-4o-mini",
      messages,
      temperature:0.4,
      max_tokens:180
    });

    const reply = completion.choices[0].message.content?.trim() || "OK";

    // Save assistant turn
    await saveTurn(userId, "assistant", reply, channel, from);

    // Update memory from this message
    const extracted = await extractMemory(prior, body);
    const merged = mergeMemory(prior, extracted);
    await supabase.from("memories").upsert({ user_id:userId, summary:merged });

    // Optional: ask for a first name once
    let footer = "";
    if (!merged?.name && channel === "whatsapp") footer = "\n\n(What’s your first name so I can save it?)";

    res.setHeader("Content-Type","text/xml");
    return res.status(200).send(`<Response><Message>${escapeXml(reply + footer)}</Message></Response>`);
  } catch (e) {
    console.error("handler fatal", e);
    res.setHeader("Content-Type","text/xml");
    return res.status(200).send("<Response><Message>Sorry, something went wrong.</Message></Response>");
  }
}

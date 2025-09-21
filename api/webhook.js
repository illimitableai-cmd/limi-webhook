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

/** GPT-5: use max_completion_tokens and omit temperature; others use legacy params */
async function safeChatCompletion({ messages, model = CHAT_MODEL, temperature = 0.4, maxTokens = 180 }) {
  const isGpt5 = /^gpt-5/i.test(model);
  const args = { model, messages };
  if (isGpt5) {
    args.max_completion_tokens = maxTokens;
    // no temperature field for gpt-5
  } else {
    args.max_tokens = maxTokens;
    args.temperature = temperature;
  }

  try {
    return await openai.chat.completions.create(args);
  } catch (err) {
    await dbg("model_fallback", { tried: model, error: String(err) });
    // Fallback to 4o-mini (supports legacy params)
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

// Normalize so SMS & WhatsApp map to the same user
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
  return { name:null, location:null, email:null, birthday:null, timezone:null,
           preferences:{}, interests:[], goals:[], notes:[], last_seen:new Date().toISOString() };
}
function mergeMemory(oldMem, add){
  const m = { ...blankMemory(), ...(oldMem||{}) };
  for (const k of ["name","location","email","birthday","timezone"]) if (add?.[k]) m[k]=add[k];
  m.preferences = { ...(oldMem?.preferences||{}), ...(add?.preferences||{}) };
  const dedupe = a => Array.from(new Set((a||[]).filter(Boolean))).slice(0,12);
  m.interests = dedupe([...(oldMem?.interests||[]), ...(add?.interests||[])]);
  m.goals     = dedupe([...(oldMem?.goals||[]),     ...(add?.goals||[])]);
  m.notes     = dedupe([...(oldMem?.notes||[]),     ...(add?.notes||[])]);
  m.last_seen = new Date().toISOString();
  return m;
}
async function extractMemory(prior, newMsg){
  const sys = "Return ONLY JSON of long-lived facts (name,location,email,birthday,timezone,preferences,interests,goals,notes). If none, return {}. If the message is about saving someone ELSE as a contact, DO NOT set name/email/birthday/timezone.";
  const user = `Prior: ${JSON.stringify(prior||{})}\nMessage: "${newMsg}"`;
  const c = await safeChatCompletion({
    model: MEMORY_MODEL,
    messages: [{role:"system",content:sys},{role:"user",content:user}],
    // no custom temp for gpt-5; helper handles it
    maxTokens: 180
  });
  try {
    const t = c.choices[0].message.content || "{}";
    const s = t.indexOf("{"), e = t.lastIndexOf("}");
    return JSON.parse(s>=0 && e>=0 ? t.slice(s,e+1) : "{}");
  } catch { return {}; }
}

/* --- name/phone utils (sanitize, choose better, normalize) --- */
function sanitizeName(raw=""){
  let n = String(raw)
    .replace(/['’]\s*s\b/gi,"")
    .replace(/\b(number|mobile|cell|phone)\b/gi,"")
    .replace(/\s{2,}/g," ")
    .trim();
  n = n.toLowerCase().replace(/\b\w/g,c=>c.toUpperCase());
  return n;
}
function isBadName(name=""){
  return /\b(number|mobile|cell|phone)\b/i.test(name) || /['’]\s*s\b/i.test(name) || name.replace(/[^a-z]/gi,"").length<2;
}
function betterName(existing="", incoming=""){
  const aBad=isBadName(existing), bBad=isBadName(incoming);
  if (bBad && !aBad) return existing;
  if (!bBad && aBad) return incoming;
  const aTok=(existing||"").trim().split(/\s+/).length;
  const bTok=(incoming||"").trim().split(/\s+/).length;
  return bTok>=aTok?incoming:existing;
}
function normalizePhone(phone=""){
  let d=String(phone).replace(/[^\d+]/g,"");
  if (d.startsWith("00")) d="+"+d.slice(2);
  if (d.startsWith("0")) d="+44"+d.slice(1);
  return d;
}

// ---- contacts helper (smart dedupe) ----
async function upsertContact({ userId, name, phone, channel }) {
  const tidyIncoming = sanitizeName(name);
  const normPhone = normalizePhone(phone);

  try {
    const { data: byPhone } = await supabase
      .from("contacts").select("id,name,phone,channel")
      .eq("user_id", userId).eq("phone", normPhone).maybeSingle();

    if (byPhone) {
      const finalName = betterName(byPhone.name||"", tidyIncoming);
      const needsUpdate = finalName !== (byPhone.name||"") || (byPhone.channel||"") !== (channel||"");
      if (needsUpdate) {
        const { error: upErr } = await supabase.from("contacts")
          .update({ name: finalName, channel }).eq("id", byPhone.id);
        if (upErr) { await dbg("contact_update_error",{code:upErr.code,message:upErr.message},userId); return {ok:false,error:upErr}; }
      }
      await dbg("contact_upsert_ok",{action:"update_by_phone",name:finalName,phone:normPhone,channel},userId);
      return { ok:true, action:"update_by_phone" };
    }

    const { data: byName } = await supabase
      .from("contacts").select("id,name,phone,channel")
      .eq("user_id", userId).ilike("name", sanitizeName(name));

    const existingByName = (byName||[]).find(r => (r.name||"").trim().toLowerCase()===tidyIncoming.toLowerCase());
    if (existingByName){
      const { error: upErr2 } = await supabase.from("contacts")
        .update({ phone: normPhone, channel }).eq("id", existingByName.id);
      if (upErr2){ await dbg("contact_update_conflict",{code:upErr2.code,message:upErr2.message},userId); return {ok:false,error:upErr2}; }
      await dbg("contact_upsert_ok",{action:"update_by_name",name:existingByName.name,phone:normPhone,channel},userId);
      return { ok:true, action:"update_by_name" };
    }

    const { error: insErr } = await supabase.from("contacts")
      .insert({ user_id:userId, name: tidyIncoming, phone: normPhone, channel });
    if (insErr){ await dbg("contact_insert_error",{code:insErr.code,message:insErr.message},userId); return {ok:false,error:insErr}; }

    await dbg("contact_upsert_ok",{action:"insert_new",name:tidyIncoming,phone:normPhone,channel},userId);
    return { ok:true, action:"insert_new" };
  } catch(e){
    await dbg("contact_upsert_exception",{message:String(e?.message||e)},userId);
    return { ok:false, error:e };
  }
}

/* -------- contact parsing + LLM fallback -------- */
function parseSaveContact(msg){
  const text=(msg||"").trim();
  const phoneMatch=text.match(/(\+?\d[\d\s().-]{6,})/);
  let phone=phoneMatch?phoneMatch[1]:null;

  const patterns=[
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

  let name=null;
  for (const re of patterns){
    const m=re.exec(text); if (!m) continue;
    if (m[2]) phone=m[2];
    if (m[1]) name=m[1];
    break;
  }
  if (!name || !phone) return null;

  const tidyName=sanitizeName(name);
  const normPhone=normalizePhone(phone);
  if (isBadName(tidyName)) return null;
  return { name: tidyName, phone: normPhone };
}

async function llmExtractContact(msg){
  const sys='Return ONLY compact JSON like {"name":"...","phone":"..."} if the text asks to save/add a contact; otherwise {}. Phone must include country code.';
  const user=`Text: ${msg}`;
  const c = await safeChatCompletion({
    model: MEMORY_MODEL,
    messages: [{role:"system",content:sys},{role:"user",content:user}],
    maxTokens: 80
  });
  try{
    const t=c.choices[0].message.content||"{}";
    const s=t.indexOf("{"), e=t.lastIndexOf("}");
    const j=JSON.parse(s>=0&&e>=0?t.slice(s,e+1):"{}");
    if (j?.name && j?.phone){
      const tidyName=sanitizeName(j.name);
      const normPhone=normalizePhone(String(j.phone));
      if (!isBadName(tidyName)) return { name: tidyName, phone: normPhone };
    }
  }catch{}
  return null;
}

/* -------- history sanitization -------- */
function cleanHistory(history){
  const allowed=new Set(["system","assistant","user","function","tool","developer"]);
  return (history||[])
    .filter(m=>m && allowed.has(m.role) && typeof m.content==="string" && m.content.trim().length>0)
    .map(m=>({ role:m.role, content:m.content.slice(0,4000) }));
}

/* -------- broadened contact-list intent matcher -------- */
function isContactListQuery(text=""){
  const t=text.trim().toLowerCase();
  return (
    /^contacts$/.test(t) ||
    /^contact\s+list$/.test(t) ||
    /^my\s+contacts$/.test(t) ||
    /^show\s+contacts$/.test(t) ||
    /(^|\b)(show|list|see|view|display)\s+(my\s+)?contacts(\b|$)/i.test(text) ||
    /(^|\b)can\s+i\s+have\s+(my\s+)?contact\s+list(\b|$)/i.test(text) ||
    /(^|\b)contacts\s+please(\b|$)/i.test(text)
  );
}

// ===================================================================
export default async function handler(req, res){
  try{
    if (req.method==="GET"){
      await dbg("ping",{ at:new Date().toISOString() });
      return res.status(200).send("ping logged");
    }
    if (req.method!=="POST") return res.status(405).send("Method Not Allowed");

    const chunks=[]; for await (const c of req) chunks.push(c);
    const raw=Buffer.concat(chunks).toString("utf8");
    const p=new URLSearchParams(raw);

    const rawFrom=p.get("From")||"";
    const channel=rawFrom.startsWith("whatsapp:")?"whatsapp":"sms";
    const from=uidFromTwilio(rawFrom);
    const body=(p.get("Body")||"").trim();
    const numMedia=Number(p.get("NumMedia")||0);
    const waProfile=p.get("ProfileName")||null;

    await dbg("webhook_in",{ channel, from, body, numMedia });

    if (!from){
      res.setHeader("Content-Type","text/xml");
      return res.status(200).send("<Response><Message>Missing sender</Message></Response>");
    }

    if (channel==="sms" && numMedia>0){
      res.setHeader("Content-Type","text/xml");
      return res.status(200).send("<Response><Message>Pics don’t work over UK SMS. WhatsApp this same number instead 👍</Message></Response>");
    }

    const userId=await getOrCreateUserId(from);
    await dbg("user_identified",{ userId, from }, userId);

    if (waProfile){
      await upsertContact({ userId, name: sanitizeName(waProfile), phone: from, channel });
    }

    if (/^buy\b/i.test(body)){
      res.setHeader("Content-Type","text/xml");
      return res.status(200).send("<Response><Message>Top up here: https://illimitableai.com/buy</Message></Response>");
    }

    // ---- QUICK CONTACT LIST (broad matcher; no model, no credits)
    if (isContactListQuery(body)){
      const { data: contacts, error: listErr } = await supabase
        .from("contacts")
        .select("name, phone")
        .eq("user_id", userId)
        .order("name", { ascending: true })
        .limit(100);

      let msg="No contacts saved yet.";
      if (!listErr && Array.isArray(contacts) && contacts.length){
        msg = contacts.map(c=>`${c.name} — ${c.phone}`).join("\n");
      } else if (listErr){
        await dbg("contacts_list_error",{ code:listErr.code, message:listErr.message }, userId);
        msg="Sorry, couldn't fetch contacts right now.";
      }

      await saveTurn(userId,"user",body,channel,from);
      await saveTurn(userId,"assistant",msg,channel,from);
      res.setHeader("Content-Type","text/xml");
      return res.status(200).send(`<Response><Message>${escapeXml(msg)}</Message></Response>`);
    }

    // ---- FAST contact save (regex + LLM) ----
    let contact=parseSaveContact(body);
    if (!contact){
      try{
        contact=await llmExtractContact(body);
        if (contact) await dbg("contact_llm_extracted",contact,userId);
      }catch(e){
        await dbg("contact_llm_extract_error",{ message:String(e) },userId);
      }
    }
    if (contact){
      const result=await upsertContact({ userId, name: contact.name, phone: contact.phone, channel });
      const verb=result?.action==="insert_new"?"Saved":"Updated";
      await saveTurn(userId,"user",body,channel,from);
      await saveTurn(userId,"assistant",`${verb} ${contact.name}`,channel,from);
      res.setHeader("Content-Type","text/xml");
      return res.status(200).send(`<Response><Message>${escapeXml(`${verb} ${contact.name}`)}</Message></Response>`);
    }

    // ---- credits gate ----
    const credits=await getCredits(userId);
    if (credits<=0){
      res.setHeader("Content-Type","text/xml");
      return res.status(200).send("<Response><Message>Out of credits. Reply BUY for a top-up link.</Message></Response>");
    }

    // ---- memory + media ----
    const { data: memRow } = await supabase.from("memories").select("summary").eq("user_id", userId).maybeSingle();
    const prior = memRow?.summary ?? blankMemory();

    await dbg("wa_media_meta",{
      channel, numMedia, mediaUrl0:p.get("MediaUrl0"), mediaType0:p.get("MediaContentType0")
    }, userId);

    let userMsg=body||"";
    let visionPart=null;
    if (channel==="whatsapp" && numMedia>0){
      try{
        const mediaUrl=p.get("MediaUrl0");
        const ctype=p.get("MediaContentType0")||"image/jpeg";
        const b64=await fetchTwilioMediaB64(mediaUrl);
        visionPart={ type:"image_url", image_url:{ url:`data:${ctype};base64,${b64}` } };
        if (!userMsg) userMsg="Please analyse this image.";
      }catch(err){
        await dbg("wa_media_fetch_error",{ message:String(err) },userId);
      }
    }

    await saveTurn(userId,"user",userMsg,channel,from);

    const history=cleanHistory(await loadRecentTurns(userId,12));
    const messages=[
      { role:"system", content:"You are Limi. Be concise. If an image is provided, describe it and answer the question." },
      ...history.slice(-11),
      { role:"user", content: visionPart ? [{type:"text",text:userMsg}, visionPart] : [{type:"text",text:userMsg}] }
    ];

    const completion=await safeChatCompletion({ model: CHAT_MODEL, messages, maxTokens: 180 });
    const reply=completion.choices[0].message.content?.trim() || "OK";

    await saveTurn(userId,"assistant",reply,channel,from);
    await setCredits(userId, Math.max(0, credits-1));

    const extracted=await extractMemory(prior, body);
    const merged=mergeMemory(prior, extracted);
    await supabase.from("memories").upsert({ user_id:userId, summary: merged });

    const bestName=merged?.name || (waProfile ? sanitizeName(waProfile) : null) || null;
    if (bestName){
      await upsertContact({ userId, name: bestName, phone: from, channel });
    }

    let footer="";
    if (!merged?.name && channel==="whatsapp") footer="\n\n(What’s your first name so I can save it?)";

    res.setHeader("Content-Type","text/xml");
    return res.status(200).send(`<Response><Message>${escapeXml(reply+footer)}</Message></Response>`);
  } catch(e){
    console.error("handler fatal",e);
    await dbg("handler_fatal",{ message:String(e?.message||e), stack:e?.stack||null });
    res.setHeader("Content-Type","text/xml");
    return res.status(200).send("<Response><Message>Sorry, something went wrong.</Message></Response>");
  }
}

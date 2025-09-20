  // Outbound SMS setup
import twilio from 'twilio';
const twilioClient = twilio(process.env.TWILIO_SID, process.env.TWILIO_AUTH);
const TWILIO_FROM = process.env.TWILIO_FROM; // e.g. +4477...

// api/webhook.js
import { createClient } from '@supabase/supabase-js';
import OpenAI from 'openai';

export const config = { api: { bodyParser: false } };

const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);
const openai = new OpenAI({ apiKey: process.env.OPENAI_KEY });

// ---------- Memory helpers ----------
function blankMemory() {
  return {
    name: null,
    location: null,
    email: null,
    birthday: null,   // YYYY-MM-DD
    timezone: null,   // IANA e.g. Europe/London
    preferences: {},
    interests: [],
    goals: [],
    notes: [],
    last_seen: new Date().toISOString()
  };
}

function mergeMemory(oldMem, newMem) {
  const m = { ...blankMemory(), ...oldMem };
  for (const k of ['name','location','email','birthday','timezone']) {
    if (newMem?.[k]) m[k] = newMem[k];
  }
  m.preferences = { ...(oldMem?.preferences||{}), ...(newMem?.preferences||{}) };

  const dedupe = arr => Array.from(new Set((arr||[]).filter(Boolean))).slice(0, 12);
  m.interests = dedupe([...(oldMem?.interests||[]), ...(newMem?.interests||[])]);
  m.goals     = dedupe([...(oldMem?.goals||[]),     ...(newMem?.goals||[])]);
  m.notes     = dedupe([...(oldMem?.notes||[]),     ...(newMem?.notes||[])]);
  m.last_seen = new Date().toISOString();
  return m;
}

async function extractMemory(openaiClient, priorMemory, newMessage) {
  const sys = `You are a memory extractor for an assistant.
Return ONLY compact JSON matching:

type Memory = {
  name?: string|null;
  location?: string|null;
  email?: string|null;
  birthday?: string|null;   // YYYY-MM-DD
  timezone?: string|null;   // IANA like "Europe/London"
  preferences?: Record<string,string>;
  interests?: string[];
  goals?: string[];
  notes?: string[];
};

Rules:
- Extract only STABLE, long-lived facts/preferences.
- Normalise birthday to YYYY-MM-DD if present; else omit.
- For timezone, return a valid IANA if clearly stated.
- If prior conflicts, prefer the NEW statement.
- Keep values <= 60 chars. No secrets.
- If nothing new, return {}.`;

  const user = `Prior memory: ${JSON.stringify(priorMemory || {})}
New user message: "${newMessage}"`;

  const comp = await openaiClient.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [{ role:'system', content: sys }, { role:'user', content: user }],
    temperature: 0.2,
    max_tokens: 200
  });

  try {
    const text = comp.choices[0].message.content || '{}';
    const s = text.indexOf('{'), e = text.lastIndexOf('}');
    return JSON.parse(s >= 0 && e >= 0 ? text.slice(s, e + 1) : '{}');
  } catch {
    return {};
  }
}
async function routeIntent(openaiClient, prior, text) {
  const sys = `You are an intent router for an SMS assistant (Limi).
Return ONLY compact JSON (no prose).

type Output = {
  action: "add_contact" | "send_text" | "set_reminder" | "link_email" | "none",
  params?: {
    name?: string,     // contact full name if any words look like a human name
    phone?: string,    // any phone-like string found anywhere in the message
    message?: string,  // message text for send_text
    when?: string,     // natural time for set_reminder (e.g. "tomorrow 8am", "in 2 hours")
    text?: string,     // the reminder note for set_reminder (e.g. "call John", "buy milk")
    email?: string
  }
};

Rules:
- Understand messy, natural language in ANY order.
  Examples:
    "add 07755... as a contact Ashley Leggett"
    "save this number under Ashley: 07755..."
    "text Ashley: running late"
    "remind me tomorrow 8am to call John"
- If a phone is present anywhere, put it in params.phone exactly as seen (no formatting required).
- For set_reminder, capture BOTH:
    params.when = the time expression
    params.text = what to be reminded about
- If you can’t infer enough for a confident action, return {"action":"none"}.
- NO explanations.`;

  const user = `Prior memory (may help with names): ${JSON.stringify(prior || {})}
Message: ${text}`;

  const comp = await openaiClient.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [{ role: "system", content: sys }, { role: "user", content: user }],
    temperature: 0,
    max_tokens: 180
  });

  try {
    const t = comp.choices[0].message.content || "{}";
    const s = t.indexOf("{"), e = t.lastIndexOf("}");
    return JSON.parse(s >= 0 && e >= 0 ? t.slice(s, e + 1) : "{}");
  } catch {
    return { action: "none" };
  }
}

  // Pull a phone from any messy text; returns digits with leading +
function findPhone(str='') {
  const m = str.match(/(\+?\d[\d\s()+-]{6,})/);
  if (!m) return null;
  return m[1].replace(/\s+/g, '');
}

// Normalize UK phones to +44 E.164-ish format
function normalizeUkPhone(p='') {
  if (!p) return p;
  let d = p.replace(/[^\d+]/g, '');
  if (d.startsWith('00')) d = '+' + d.slice(2);
  if (d.startsWith('0')) d = '+44' + d.slice(1);
  return d;
}

function titleCaseName(n='') {
  const s = n.trim().replace(/\s+/g,' ').toLowerCase();
  return s.replace(/\b\w/g, c => c.toUpperCase());
}

// quick iso-ish parser for common natural "when" phrases
function parseWhen(str, tzGuess = "Europe/London") {
  // very lightweight: “tomorrow 08:00”, “in 2 hours”, “2025-12-24 17:00”
  const now = new Date();
  const lower = (str || "").toLowerCase().trim();

  // ISO-like
  const iso = lower.match(/\d{4}-\d{2}-\d{2}(\s+|\s*at\s*)(\d{1,2}:\d{2})/);
  if (iso) return new Date(`${iso[0].replace(" at ", " ") }:00Z`);

  // tomorrow HH:MM
  const tom = lower.match(/tomorrow\s+(\d{1,2}:\d{2})/);
  if (tom) {
    const [h,m] = tom[1].split(":").map(Number);
    const d = new Date();
    d.setDate(d.getDate()+1); d.setHours(h,m,0,0);
    return d;
  }

  // in N hours / minutes
  const inH = lower.match(/in\s+(\d+)\s*hours?/);
  if (inH) { const d = new Date(now); d.setHours(d.getHours()+Number(inH[1])); return d; }
  const inM = lower.match(/in\s+(\d+)\s*mins?|minutes?/);
  if (inM) { const d = new Date(now); d.setMinutes(d.getMinutes()+Number(inM[1])); return d; }

  // HH:MM today/tomorrow guess
  const hm = lower.match(/(\d{1,2}:\d{2})/);
  if (hm) {
    const [h,m] = hm[1].split(":").map(Number);
    const d = new Date(); d.setHours(h,m,0,0);
    if (d < now) d.setDate(d.getDate()+1); // if passed, schedule for tomorrow
    return d;
  }

  return null;
}

async function dbg(step, payload, userId = null) {
  try {
    await supabase.from('debug_logs').insert([{ step, payload, user_id: userId }]);
  } catch (e) {
    // last resort
    console.error('dbg insert fail', e);
  }
}

// ---------- Handler ----------
export default async function handler(req, res) {
  try {
    // ---- TEST HOOK (so you can ping it in a browser) ----
    if (req.method === 'GET') {
      if (req.query?.testLog) {
        await dbg('manual_ping', { from: 'GET test' });
        return res.status(200).send('logged');
      }
      return res.status(200).send('Limi webhook is alive');
    }

    
    // ------------------------------------------------------

    if (req.method !== 'POST') {
      return res.status(405).send('Method Not Allowed');
    }

    
    // ------------------------------------------------------

    if (req.method !== 'POST') return res.status(405).send('Method Not Allowed');

    // Parse Twilio x-www-form-urlencoded body
    const chunks = [];
    for await (const c of req) chunks.push(c);
    const raw = Buffer.concat(chunks).toString('utf8');
    const params = new URLSearchParams(raw);

    // Normalise From so SMS + WhatsApp map to same ID
    const rawFrom = params.get('From') || '';
    const from = rawFrom.replace(/^whatsapp:/i, '').replace(/^sms:/i, '').trim();
    const body = (params.get('Body') || '').trim();
    // Normalise for command matching (strip polite openers, collapse spaces)
    const cmd = body
      .replace(/^(please|plz|hey|hi|hello|yo|limi|hey limi|hi limi)[,!\s:-]*/i, '')
      .replace(/\?+$/, '')
      .replace(/\s+/g, ' ')
      .trim();
    // DEBUG: confirm env + inbound payload (use error so it always shows)
    console.error('env_ok', {
      openai: !!process.env.OPENAI_KEY,
      supabase_url: !!process.env.SUPABASE_URL,
      supabase_key: !!process.env.SUPABASE_KEY,
      twilio_sid: !!process.env.TWILIO_SID,
      twilio_auth: !!process.env.TWILIO_AUTH,
      twilio_from: !!process.env.TWILIO_FROM,
});
    
    console.error('webhook_in', { from, body, cmd });
    console.error('DEBUG_CMD', { body, cmd });
    console.log('webhook START');
    console.log({ from, body });
    await dbg('webhook_in', { from, body });

    if (!from || !body) return res.status(400).send('Missing From/Body');

    // Find or create user by identifier
    const { data: ident } = await supabase
      .from('identifiers').select('user_id').eq('value', from).maybeSingle();

    let userId = ident?.user_id;
    if (!userId) {
      const { data: user, error: userErr } = await supabase
        .from('users').insert([{ display_name: null }]).select().single();
      if (userErr) { console.error('user insert err', userErr); throw userErr; }
      userId = user.id;
      await supabase
        .from('identifiers').insert([{ user_id: userId, type: 'phone', value: from }]);
    }
    await dbg('user_identified', { userId, from });
      
    // --- Load prior memory EARLY (some commands use it) ---
    const { data: mem } = await supabase
      .from('memories').select('summary').eq('user_id', userId).maybeSingle();
const prior = mem?.summary ?? blankMemory();
const tz = prior.timezone || 'Europe/London';

// ---------- EARLY FALLBACK: add contact from messy phrasing ----------
{
  const phoneInBody = (body.match(/(\+?\d[\d\s()+-]{6,})/g) || [])[0];
  const directSave = /save\s+([a-zA-Z][a-zA-Z\s'’-]{1,40})\s+as\s+a\s+contact\b/i.exec(body);

  const patterns = [
    /(?:^|\b)(?:add|save)\b[\s\S]*?(\+?\d[\d\s()+-]{6,})[\s\S]*?\bcontact\b[\s:,-]*([a-zA-Z][a-zA-Z\s'’-]{1,40})\b/i,
    /(?:^|\b)(?:add|save)\b[\s\S]*?\bcontact\b[\s:,-]*([a-zA-Z][a-zA-Z\s'’-]{1,40})[\s\S]*?(\+?\d[\d\s()+-]{6,})/i,
    /(?:^|\b)save\s+([a-zA-Z][a-zA-Z\s'’-]{1,40})\s+as\s+a\s+contact[\s\S]*?(\+?\d[\d\s()+-]{6,})/i,
    /(?:^|\b)(?:add|save)\b[\s\S]*?(?:number|no\.?)\s*(\+?\d[\d\s()+-]{6,})[\s\S]*?\b(?:under|for)\b\s+([a-zA-Z][a-zA-Z\s'’-]{1,40})/i,
    /(?:^|\b)(?:add|save)\b[\s\S]*?([a-zA-Z][a-zA-Z\s'’-]{1,40})[\s\S]*?(?:number|no\.?)\s*(\+?\d[\d\s()+-]{6,})/i,
  ];

  let rawName = null, rawPhone = null;

  if (directSave && phoneInBody) {
    rawName  = directSave[1];
    rawPhone = phoneInBody;
  } else {
    for (const r of patterns) {
      const m = r.exec(body);
      if (m) {
        if (/\d/.test(m[1])) { rawPhone = m[1]; rawName = m[2]; }
        else { rawName = m[1]; rawPhone = m[2]; }
        break;
      }
    }
    const nameSaveMatch1 =
      /(?:^| )(?:add|save)\s+(?:a\s+)?contact\s+([a-zA-Z][a-zA-Z\s'’-]{1,40})(?:\b|$)/i.exec(cmd);
    const nameSaveMatch2 =
      /(?:^| )(?:can you|please)?\s*save\s+([a-zA-Z][a-zA-Z\s'’-]{1,40})\s+as\s+a\s+contact\b/i.exec(cmd);
    if (!rawName) rawName = (nameSaveMatch1?.[1] || nameSaveMatch2?.[1]) || null;
    if (!rawPhone && phoneInBody) rawPhone = phoneInBody;
  }

  // Log AFTER parsing so we can see what was found
  await dbg('contact_fallback_parsed', { rawName, rawPhone, body }, userId);

  if (rawName && rawPhone) {
    const contactName = titleCaseName(rawName);
    const phoneClean  = normalizeUkPhone(rawPhone);

    await dbg('contact_upsert_try', { source: 'fallback', name: contactName, phone: phoneClean }, userId);

    const { error: cErr } = await supabase
      .from('contacts')
      .upsert(
        { user_id: userId, name: contactName, phone: phoneClean },
        { onConflict: 'user_id,name' }
      );

    if (cErr) {
      await dbg('contact_upsert_error', {
        source: 'fallback',
        code: cErr.code, message: cErr.message, details: cErr.details, hint: cErr.hint
      }, userId);

      res.setHeader('Content-Type','text/xml');
      return res.status(200).send('<Response><Message>Could not save contact right now.</Message></Response>');
    }

    res.setHeader('Content-Type','text/xml');
    return res.status(200).send(`<Response><Message>Saved ${contactName}</Message></Response>`);
  }
}
// ---------- END EARLY FALLBACK ----------

    
// Now try the LLM router only if fallback didn’t handle it
const intent = await routeIntent(openai, prior, body);

    console.error('intent_json', JSON.stringify(intent));
    console.error('intent_out', intent);

if (intent?.action && intent.action !== 'none') {
  const a = intent.action;
  const p = intent.params || {};

  // ADD CONTACT: accept phone or try to find one anywhere in the raw body
if (a === 'add_contact') {
  const nameRaw  = (p.name || '').trim();
  const phoneRaw = (p.phone || findPhone(body));

  const name  = titleCaseName(nameRaw);
  const phone = phoneRaw ? normalizeUkPhone(phoneRaw) : null;

  if (!name || !phone) {
    res.setHeader('Content-Type','text/xml');
    return res.status(200).send('<Response><Message>I need a name and a phone to save a contact.</Message></Response>');
  }

  // NEW: log what we’ll upsert via the LLM route
  await dbg('contact_upsert_try', { source: 'llm', name, phone }, userId);

  const { error: cErr } = await supabase
    .from('contacts')
    .upsert({ user_id: userId, name, phone }, { onConflict: 'user_id,name' });

  if (cErr) {
    // NEW: capture DB error
    await dbg('contact_upsert_error', {
      source: 'llm',
      code: cErr.code,
      message: cErr.message,
      details: cErr.details,
      hint: cErr.hint
    }, userId);

    res.setHeader('Content-Type','text/xml');
    return res.status(200).send('<Response><Message>Could not save contact right now.</Message></Response>');
  }

  // SEND TEXT
  if (a === 'send_text') {
    const name = (p.name || '').trim();
    const msg  = (p.message || '').trim();

    if (!name || !msg) {
      res.setHeader('Content-Type','text/xml');
      return res.status(200).send('<Response><Message>I need who to text and the message.</Message></Response>');
    }

    // check credits
    const { data: bal } = await supabase.from('credits').select('balance').eq('user_id', userId).maybeSingle();
    if (!bal || bal.balance <= 0) {
      res.setHeader('Content-Type','text/xml');
      return res.status(200).send('<Response><Message>Out of credits. Reply BUY for a top-up link.</Message></Response>');
    }

    const { data: list } = await supabase.from('contacts').select('name,phone').eq('user_id', userId);
    const contact = (list || []).find(c => c.name.toLowerCase() === name.toLowerCase());

    res.setHeader('Content-Type','text/xml');
    if (!contact) {
      return res.status(200).send(`<Response><Message>No contact "${name}". Try: add contact ${name} +44...</Message></Response>`);
    }

    await twilioClient.messages.create({ to: contact.phone, from: TWILIO_FROM, body: msg.slice(0,320) });
    await supabase.from('messages').insert([{ user_id: userId, channel: 'sms', external_id: contact.phone, body: `(outbound) ${msg.slice(0,320)}` }]);
    await supabase.from('credits').upsert({ user_id: userId, balance: (bal.balance - 1) });

    return res.status(200).send(`<Response><Message>Sent to ${contact.name}</Message></Response>`);
  }

  // LINK EMAIL
  if (a === 'link_email' && p.email) {
    await supabase.from('identifiers').upsert({ user_id: userId, type: 'email', value: p.email.toLowerCase() });
    res.setHeader('Content-Type','text/xml');
    return res.status(200).send(`<Response><Message>Linked email: ${p.email}</Message></Response>`);
  }

  // SET REMINDER
  if (a === 'set_reminder' && p.when && p.text) {
    const when = parseWhen(p.when, tz) || new Date(p.when);
    if (!when || Number.isNaN(+when)) {
      res.setHeader('Content-Type','text/xml');
      return res.status(200).send('<Response><Message>I couldn’t parse the time.</Message></Response>');
    }
    await supabase.from('reminders').insert({ user_id: userId, text: p.text, run_at: when.toISOString(), tz, status: 'scheduled' });
    res.setHeader('Content-Type','text/xml');
    return res.status(200).send('<Response><Message>Reminder set.</Message></Response>');
  }

  // If we had an action but missing bits, fall through to regex fallback next.
}

    
if (/^buy\b/i.test(cmd)) {
  res.setHeader('Content-Type', 'text/xml');
  return res.status(200)
    .send('<Response><Message>Top up here: https://illimitableai.com/buy</Message></Response>');
}
    // --- Credits (simple) ---
    const { data: bal } = await supabase
      .from('credits').select('balance').eq('user_id', userId).maybeSingle();

    if (!bal || bal.balance <= 0) {
      res.setHeader('Content-Type', 'text/xml');
      return res
        .status(200)
        .send('<Response><Message>Out of credits. Reply BUY for a top-up link.</Message></Response>');
    }


    
    // ---- AI flow (also charges a credit) ----

    // Extract & merge memory
    const extracted = await extractMemory(openai, prior, body);
    const summary = mergeMemory(prior, extracted);

    // Persist memory immediately
    await supabase.from('memories').upsert({ user_id: userId, summary });

    // Log inbound message
    await supabase
      .from('messages').insert([{ user_id: userId, channel: 'sms', external_id: from, body }]);

    // Decrement credits for this AI turn
    const newBal = (bal?.balance ?? 1) - 1;
    await supabase.from('credits').upsert({ user_id: userId, balance: newBal });

    // Generate reply (keep short for Twilio trial)
    const replyComp = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        { role: 'system',
          content:
            'You are Limi. Use memory (name, location, email, birthday, timezone, preferences, interests, goals, notes). Keep replies under 120 characters unless asked for detail.' },
        { role: 'user', content: `Memory: ${JSON.stringify(summary)}` },
        { role: 'user', content: `User: ${body}` }
      ],
      temperature: 0.5,
      max_tokens: 80
    });
    const reply = replyComp.choices[0].message.content?.trim() || 'OK';

    // Save last reply
    summary.lastReply = reply;
    await supabase.from('memories').upsert({ user_id: userId, summary });

    // Respond as TwiML
    res.setHeader('Content-Type', 'text/xml');
    return res.status(200).send(`<Response><Message>${reply}</Message></Response>`);

  } catch (e) {
    console.error('handler fatal', e);
    res.setHeader('Content-Type', 'text/xml');
    return res.status(200).send('<Response><Message>Sorry, something went wrong.</Message></Response>');
  }
}


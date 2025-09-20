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
Return ONLY compact JSON.

type Output = {
  action: "add_contact" | "send_text" | "set_reminder" | "link_email" | "none",
  params?: {
    name?: string,
    phone?: string,
    message?: string,  // for send_text
    when?: string,     // natural time
    text?: string,     // for set_reminder (- For set_reminder, capture both:
  - params.when = time expression ("tomorrow 8am", "in 2 hours")
  - params.text = what to be reminded about ("call John", "buy milk")
)
    email?: string
  }
};


Rules:
- Understand messy, natural language (any order)...
- If you can’t infer enough, return {"action":"none"}.
`;

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

// ---------- Handler ----------
export default async function handler(req, res) {
  try {
    if (req.method === 'GET') return res.status(200).send('Limi webhook is alive');
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

    // --- Load prior memory EARLY (some commands use it) ---
    const { data: mem } = await supabase
      .from('memories').select('summary').eq('user_id', userId).maybeSingle();
    const prior = mem?.summary ?? blankMemory();
    const tz = prior.timezone || 'Europe/London';

// --- LLM FIRST: try to understand any natural phrasing ---
const intent = await routeIntent(openai, prior, body);
console.error('intent_out', intent);

if (intent?.action && intent.action !== 'none') {
  const a = intent.action;
  const p = intent.params || {};

  // ADD CONTACT: accept phone or try to find one anywhere in the raw body
  if (a === 'add_contact') {
    const name  = (p.name || '').trim();
    const phone = (p.phone || findPhone(body));

    if (!name || !phone) {
      res.setHeader('Content-Type','text/xml');
      return res.status(200).send('<Response><Message>I need a name and a phone to save a contact.</Message></Response>');
    }

    const { error: cErr } = await supabase
      .from('contacts')
      .upsert({ user_id: userId, name, phone: phone.replace(/\s+/g,'') }, { onConflict: 'user_id,name' });

    res.setHeader('Content-Type','text/xml');
    if (cErr) {
      console.error('contacts upsert error', cErr);
      return res.status(200).send('<Response><Message>Could not save contact right now.</Message></Response>');
    }
    return res.status(200).send(`<Response><Message>Saved ${name}</Message></Response>`);
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



// ----- TEMP FALLBACK: contacts + send text (robust) -----
/**
 * We parse name and phone separately so it also works when there's no space
 * between "contact?" and the phone (e.g. "contact?+4477...") or when the
 * phone is on a new line.
 */
let matched = false;

// 1) Find a phone number anywhere in the original body
const phoneFromBody = (body.match(/(\+?\d[\d\s()+-]{6,})/g) || []).pop();

// 2) Find a “save contact” intent + name in the normalized text
const nameSaveMatch =
  /(?:^| )(?:add|save)\s+(?:a\s+)?contact\s+([a-zA-Z][a-zA-Z\s'’-]{1,40})(?:\b|$)/i.exec(cmd) ||
  /(?:^| )(?:can you|please)?\s*save\s+([a-zA-Z][a-zA-Z\s'’-]{1,40})\s+as\s+a\s+contact\b/i.exec(cmd);

// If we have both a name and a phone, save the contact now
if (nameSaveMatch && phoneFromBody) {
  matched = true;
  const contactName = nameSaveMatch[1].trim();
  const phoneClean  = phoneFromBody.replace(/\s+/g, '');

  const { error: cErr } = await supabase
    .from('contacts')
    .upsert(
      { user_id: userId, name: contactName, phone: phoneClean },
      { onConflict: 'user_id,name' } // requires unique index on (user_id, name)
    );

  res.setHeader('Content-Type','text/xml');
  if (cErr) {
    console.error('contacts upsert error', cErr);
    return res.status(200).send('<Response><Message>Could not save contact right now.</Message></Response>');
  }
  return res.status(200).send(`<Response><Message>Saved ${contactName}</Message></Response>`);
}

// 3) Send a text: "send a text to Louise Hart saying I love you"
const sendTextMatch =
  /(?:^| )(?:send|text|message)\s+(?:a\s+)?(?:text\s+)?to\s+([a-zA-Z][a-zA-Z\s'’-]{1,40})\s+(?:that|saying|to)?\s*(.+)$/i.exec(cmd);

if (sendTextMatch) {
  matched = true;
  const [, nameRaw, msg] = sendTextMatch;
  const name = nameRaw.trim().toLowerCase();

  const { data: list, error: listErr } = await supabase
    .from('contacts').select('name,phone').eq('user_id', userId);
  if (listErr) console.error('contacts select err', listErr);

  const contact = (list || []).find(c => c.name.toLowerCase() === name);
  res.setHeader('Content-Type','text/xml');

  if (!contact) {
    return res.status(200).send(
      `<Response><Message>No contact "${nameRaw}". Try: add contact ${nameRaw} +44...</Message></Response>`
    );
  }

  await twilioClient.messages.create({
    to: contact.phone,
    from: TWILIO_FROM,
    body: msg.slice(0, 320)
  });

  await supabase.from('messages').insert([
    { user_id: userId, channel: 'sms', external_id: contact.phone, body: `(outbound) ${msg.slice(0,320)}` }
  ]);

  return res.status(200).send(`<Response><Message>Sent to ${contact.name}</Message></Response>`);
}
// ----- END TEMP FALLBACK -----

  // If action had missing params fall through to chat
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

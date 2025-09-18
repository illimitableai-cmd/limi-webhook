import twilio from 'twilio';
const twilioClient = twilio(process.env.TWILIO_SID, process.env.TWILIO_AUTH);
const TWILIO_FROM = process.env.TWILIO_FROM; // your Twilio number e.g. +4477...

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
    birthday: null,   // ISO date: YYYY-MM-DD
    timezone: null,   // IANA tz: e.g., "Europe/London"
    preferences: {},
    interests: [],
    goals: [],
    notes: [],
    last_seen: new Date().toISOString()
  };
}

function mergeMemory(oldMem, newMem) {
  const m = { ...blankMemory(), ...oldMem };

  for (const k of ['name', 'location', 'email', 'birthday', 'timezone']) {
    if (newMem?.[k]) m[k] = newMem[k];
  }
  m.preferences = { ...(oldMem?.preferences || {}), ...(newMem?.preferences || {}) };

  const dedupe = arr => Array.from(new Set((arr || []).filter(Boolean))).slice(0, 12);
  m.interests = dedupe([...(oldMem?.interests || []), ...(newMem?.interests || [])]);
  m.goals     = dedupe([...(oldMem?.goals || []),     ...(newMem?.goals || [])]);
  m.notes     = dedupe([...(oldMem?.notes || []),     ...(newMem?.notes || [])]);

  m.last_seen = new Date().toISOString();
  return m;
}

async function extractMemory(openaiClient, priorMemory, newMessage) {
  const sys = `You are a memory extractor for an assistant.
Return ONLY compact JSON matching:

type Memory = {
  name?: string | null;
  location?: string | null;
  email?: string | null;
  birthday?: string | null;   // ISO YYYY-MM-DD
  timezone?: string | null;   // IANA like "Europe/London"
  preferences?: Record<string,string>;
  interests?: string[];
  goals?: string[];
  notes?: string[];
};

Rules:
- Extract STABLE facts or long-lived preferences only.
- Normalise birthday to ISO YYYY-MM-DD if a date is present; else omit.
- For timezone, return a valid IANA name if clearly stated (e.g., "Europe/London").
- If prior memory conflicts, prefer the NEW statement.
- Keep values short (<= 60 chars). Never include secrets.
- If nothing new, return {}.`;

  const user = `Prior memory: ${JSON.stringify(priorMemory || {})}
New user message: "${newMessage}"`;

  const comp = await openaiClient.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [{ role: 'system', content: sys }, { role: 'user', content: user }],
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

    if (!from || !body) return res.status(400).send('Missing From/Body');

    // Find or create user by identifier
    const { data: ident, error: identErr } = await supabase
      .from('identifiers').select('user_id').eq('value', from).maybeSingle();
    if (identErr) console.error('ident select err', identErr);

    let userId = ident?.user_id;
    if (!userId) {
      const { data: user, error: userErr } = await supabase
        .from('users').insert([{ display_name: null }]).select().single();
      if (userErr) { console.error('user insert err', userErr); throw userErr; }
      userId = user.id;
      const { error: linkErr } = await supabase
        .from('identifiers').insert([{ user_id: userId, type: 'phone', value: from }]);
      if (linkErr) console.error('identifier insert err', linkErr);
    }

    // Commands (processed before AI)
    // BUY -> send top-up link
    if (/^buy\b/i.test(body)) {
      res.setHeader('Content-Type', 'text/xml');
      return res
        .status(200)
        .send('<Response><Message>Top up here: https://illimitableai.com/buy</Message></Response>');
    }

    // LINK EMAIL command
    const linkMatch = /^link email\s+(.+)$/i.exec(body);
    if (linkMatch) {
      const email = linkMatch[1].trim().toLowerCase();
      await supabase.from('identifiers').upsert({ user_id: userId, type: 'email', value: email });
      res.setHeader('Content-Type', 'text/xml');
      return res.status(200).send(`<Response><Message>Linked email: ${email}</Message></Response>`);
    }
// ===== START: CONTACTS (free commands) =====

// ADD CONTACT Jon +447700000000
const addC = /^add contact\s+([a-zA-Z][a-zA-Z\s'-]{1,40})\s+(\+?\d[\d\s()+-]{6,})$/i.exec(body);
if (addC) {
  const [, name, phone] = addC;
  await supabase.from('contacts').upsert(
    { user_id: userId, name: name.trim(), phone: phone.replace(/\s+/g,'') },
    { onConflict: 'user_id,name' }
  );
  res.setHeader('Content-Type','text/xml');
  return res.status(200).send(`<Response><Message>Saved ${name}</Message></Response>`);
}

// REMINDERS (schedule only; no send now)

// “remind me tomorrow at 08:00 to bring coffee”
let m = /^remind me (?:tomorrow|tmrw) at (\d{1,2}:\d{2})\s+to\s+(.+)$/i.exec(body);
if (m) {
  const [, hhmm, text] = m;
  const d = new Date(); d.setDate(d.getDate()+1);
  const [h, mi] = hhmm.split(':').map(Number);
  d.setHours(h, mi, 0, 0);
  await supabase.from('reminders').insert({
    user_id: userId, text, run_at: d.toISOString(),
    tz: (summary?.timezone || 'Europe/London'), status: 'scheduled'
  });
  res.setHeader('Content-Type','text/xml');
  return res.status(200).send('<Response><Message>Reminder set.</Message></Response>');
}

// “remind me on 2025-02-14 at 19:00 to book dinner”
m = /^remind me on (\d{4}-\d{2}-\d{2}) (?:at )?(\d{1,2}:\d{2})\s+to\s+(.+)$/i.exec(body);
if (m) {
  const [, ymd, hhmm, text] = m;
  const d = new Date(`${ymd}T${hhmm}:00Z`); // simple UTC
  await supabase.from('reminders').insert({
    user_id: userId, text, run_at: d.toISOString(),
    tz: (summary?.timezone || 'Europe/London'), status: 'scheduled'
  });
  res.setHeader('Content-Type','text/xml');
  return res.status(200).send('<Response><Message>Reminder set.</Message></Response>');
}

// “send me a text on my birthday”
m = /^send me a text (?:on|at)? my birthday\b/i.exec(body);
if (m) {
  const bday = (summary?.birthday);
  if (!bday) {
    res.setHeader('Content-Type','text/xml');
    return res.status(200).send('<Response><Message>I don’t know your birthday yet.</Message></Response>');
  }
  const now = new Date();
  const [Y,M,D] = bday.split('-').map(Number);
  let when = new Date(Date.UTC(now.getUTCFullYear(), M-1, D, 9, 0, 0));
  if (when < now) when = new Date(Date.UTC(now.getUTCFullYear()+1, M-1, D, 9, 0, 0));
  await supabase.from('reminders').insert({
    user_id: userId, text: 'Happy birthday! 🎉', run_at: when.toISOString(),
    tz: (summary?.timezone || 'Europe/London'), status: 'scheduled'
  });
  res.setHeader('Content-Type','text/xml');
  return res.status(200).send('<Response><Message>Birthday text scheduled.</Message></Response>');
}

// ===== END: CONTACTS / REMINDERS (free commands) =====
    
// ===== START: SEND TEXT TO CONTACT (uses a credit) =====
const sendC = /^send (?:a )?text to\s+([a-zA-Z][a-zA-Z\s'-]{1,40})\s+(?:that|saying|to)?\s*(.+)$/i.exec(body);
if (sendC) {
  const [, nameRaw, msg] = sendC;
  const name = nameRaw.trim().toLowerCase();

  const { data: list, error: listErr } = await supabase
    .from('contacts').select('name,phone').eq('user_id', userId);
  if (listErr) console.error('contacts select err', listErr);

  const contact = (list || []).find(c => c.name.toLowerCase() === name);
  if (!contact) {
    res.setHeader('Content-Type','text/xml');
    return res.status(200).send(
      `<Response><Message>No contact '${nameRaw}'. Try: ADD CONTACT ${nameRaw} +44...</Message></Response>`
    );
  }

  // Twilio trial can only send to verified numbers
  await twilioClient.messages.create({
    to: contact.phone,
    from: TWILIO_FROM,
    body: msg.slice(0, 320)
  });

  // (Optional) log an outbound message
  await supabase.from('messages').insert([
    { user_id: userId, channel: 'sms', external_id: contact.phone, body: `(outbound) ${msg.slice(0,320)}` }
  ]);

  // You already decremented a credit above; if you prefer to charge ONLY here,
  // move your credits decrement into this block and the AI block.

  res.setHeader('Content-Type','text/xml');
  return res.status(200).send(`<Response><Message>Sent to ${contact.name}</Message></Response>`);
}
// ===== END: SEND TEXT =====

    // Load prior memory
    const { data: mem, error: memSelErr } = await supabase
      .from('memories').select('summary').eq('user_id', userId).maybeSingle();
    if (memSelErr) console.error('mem select err', memSelErr);
    const prior = mem?.summary ?? blankMemory();

    // --- Credits (simple) ---
    // Seed at least one row in "credits" first; see instructions below.
    const { data: bal } = await supabase
      .from('credits').select('balance').eq('user_id', userId).maybeSingle();

    if (!bal || bal.balance <= 0) {
      res.setHeader('Content-Type', 'text/xml');
      return res
        .status(200)
        .send('<Response><Message>Out of credits. Reply BUY for a top-up link.</Message></Response>');
    }

    // Extract & merge memory
    const extracted = await extractMemory(openai, prior, body);
    const summary = mergeMemory(prior, extracted);

    // Persist memory immediately
    const { error: up1 } = await supabase.from('memories').upsert({ user_id: userId, summary });
    if (up1) console.error('mem upsert err (pre)', up1);

    // Log inbound message
    const { error: msgErr } = await supabase
      .from('messages').insert([{ user_id: userId, channel: 'sms', external_id: from, body }]);
    if (msgErr) console.error('message insert err', msgErr);

    // Decrement credits
    const newBal = (bal?.balance ?? 1) - 1;
    const { error: credErr } = await supabase
      .from('credits').upsert({ user_id: userId, balance: newBal });
    if (credErr) console.error('credits upsert err', credErr);

    // Generate reply (short for Twilio trial)
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
    const { error: up2 } = await supabase.from('memories').upsert({ user_id: userId, summary });
    if (up2) console.error('mem upsert err (post)', up2);

    // Respond as TwiML
    res.setHeader('Content-Type', 'text/xml');
    return res.status(200).send(`<Response><Message>${reply}</Message></Response>`);
  } catch (e) {
    console.error('handler fatal', e);
    res.setHeader('Content-Type', 'text/xml');
    return res
      .status(200)
      .send('<Response><Message>Sorry, something went wrong.</Message></Response>');
  }
}

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

    // --- Commands (processed before AI) ---

    // BUY -> send top-up link (free)
    if (/^buy\b/i.test(body)) {
      res.setHeader('Content-Type', 'text/xml');
      return res
        .status(200)
        .send('<Response><Message>Top up here: https://illimitableai.com/buy</Message></Response>');
    }

    // LINK EMAIL me@domain.com (free)
    const linkMatch = /^link email\s+(.+)$/i.exec(body);
    if (linkMatch) {
      const email = linkMatch[1].trim().toLowerCase();
      await supabase.from('identifiers').upsert({ user_id: userId, type: 'email', value: email });
      res.setHeader('Content-Type', 'text/xml');
      return res.status(200).send(`<Response><Message>Linked email: ${email}</Message></Response>`);
    }

    // ADD CONTACT Jon +447700000000 (free)
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

    // “remind me tomorrow at 08:00 to bring coffee” (free)
    let m = /^remind me (?:tomorrow|tmrw) at (\d{1,2}:\d{2})\s+to\s+(.+)$/i.exec(body);
    if (m) {
      const [, hhmm, text] = m;
      const d = new Date(); d.setDate(d.getDate()+1);
      const [h, mi] = hhmm.split(':').map(Number);
      d.setHours(h, mi, 0, 0);
      await supabase.from('reminders').insert({
        user_id: userId, text, run_at: d.toISOString(), tz, status: 'scheduled'
      });
      res.setHeader('Content-Type','text/xml');
      return res.status(200).send('<Response><Message>Reminder set.</Message></Response>');
    }

    // “remind me on 2025-02-14 at 19:00 to book dinner” (free)
    m = /^remind me on (\d{4}-\d{2}-\d{2}) (?:at )?(\d{1,2}:\d{2})\s+to\s+(.+)$/i.exec(body);
    if (m) {
      const [, ymd, hhmm, text] = m;
      const d = new Date(`${ymd}T${hhmm}:00Z`);
      await supabase.from('reminders').insert({
        user_id: userId, text, run_at: d.toISOString(), tz, status: 'scheduled'
      });
      res.setHeader('Content-Type','text/xml');
      return res.status(200).send('<Response><Message>Reminder set.</Message></Response>');
    }

    // “send me a text on my birthday” (free)
    m = /^send me a text (?:on|at)? my birthday\b/i.exec(body);
    if (m) {
      const bday = prior.birthday;
      if (!bday) {
        res.setHeader('Content-Type','text/xml');
        return res.status(200).send('<Response><Message>I don’t know your birthday yet.</Message></Response>');
      }
      const now = new Date();
      const [Y,M,D] = bday.split('-').map(Number);
      let when = new Date(Date.UTC(now.getUTCFullYear(), M-1, D, 9, 0, 0));
      if (when < now) when = new Date(Date.UTC(now.getUTCFullYear()+1, M-1, D, 9, 0, 0));
      await supabase.from('reminders').insert({
        user_id: userId, text: 'Happy birthday! 🎉', run_at: when.toISOString(), tz, status: 'scheduled'
      });
      res.setHeader('Content-Type','text/xml');
      return res.status(200).send('<Response><Message>Birthday text scheduled.</Message></Response>');
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

    // SEND TEXT TO CONTACT (charge a credit)
    const sendC = /^send (?:a )?text to\s+([a-zA-Z][a-zA-Z\s'-]{1,40})\s+(?:that|saying|to)?\s*(.+)$/i.exec(body);
    if (sendC) {
      const [, nameRaw, msg] = sendC;
      const name = nameRaw.trim().toLowerCase();

      const { data: list } = await supabase
        .from('contacts').select('name,phone').eq('user_id', userId);
      const contact = (list || []).find(c => c.name.toLowerCase() === name);

      if (!contact) {
        res.setHeader('Content-Type','text/xml');
        return res.status(200).send(
          `<Response><Message>No contact '${nameRaw}'. Try: ADD CONTACT ${nameRaw} +44...</Message></Response>`
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

      // Charge a credit for this request
      const newBalNow = (bal?.balance ?? 1) - 1;
      await supabase.from('credits').upsert({ user_id: userId, balance: newBalNow });

      res.setHeader('Content-Type','text/xml');
      return res.status(200).send(`<Response><Message>Sent to ${contact.name}</Message></Response>`);
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

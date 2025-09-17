import { createClient } from '@supabase/supabase-js';
import OpenAI from 'openai';

const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);
const openai = new OpenAI({ apiKey: process.env.OPENAI_KEY });

/** Utility */
const nowISO = () => new Date().toISOString();
const isStr = (v) => typeof v === 'string' && v.trim().length > 0;

/** Merge helper: prefer existing stable facts; only fill gaps or update if explicitly set */
function mergeMemory(oldMem = {}, patch = {}) {
  const out = structuredClone(oldMem);

  // top-level stable facts
  const fields = [
    'name', 'nickname', 'pronouns',
    'email', 'location', 'timezone',
    'interests', 'tone', 'household'
  ];
  for (const f of fields) {
    if (patch?.[f] !== undefined) {
      if (out[f] == null || (patch.__force && isStr(patch[f]))) out[f] = patch[f];
    }
  }

  // projects (unique by title)
  if (Array.isArray(patch?.projects)) {
    out.projects = Array.isArray(out.projects) ? out.projects : [];
    const seen = new Set(out.projects.map((p) => p.title?.toLowerCase()));
    for (const p of patch.projects) {
      const key = (p.title || '').toLowerCase();
      if (key && !seen.has(key)) out.projects.push(p);
    }
  }

  // reminders (append)
  if (Array.isArray(patch?.reminders) && patch.reminders.length) {
    out.reminders = Array.isArray(out.reminders) ? out.reminders : [];
    out.reminders.push(...patch.reminders);
  }

  // short rolling relationship summary
  if (isStr(patch.summary)) out.summary = patch.summary;

  // housekeeping
  out.last_seen = nowISO();
  return out;
}

/** Parse Twilio body (form or json) */
function getFromBody(req) {
  const ct = (req.headers['content-type'] || '').toLowerCase();
  if (ct.includes('application/x-www-form-urlencoded')) {
    const parsed = typeof req.body === 'string'
      ? Object.fromEntries(new URLSearchParams(req.body))
      : req.body;
    return { from: parsed?.From, body: parsed?.Body, raw: parsed };
  }
  if (ct.includes('application/json')) {
    return { from: req.body?.From, body: req.body?.Body, raw: req.body };
  }
  return { from: null, body: null, raw: null };
}

/** Commands (LINK/SET/GET/FORGET/RESET) */
async function handleCommands(text, userId) {
  const t = text.trim();

  // LINK EMAIL foo@bar.com
  let m = /^link\s+email\s+(.+)$/i.exec(t);
  if (m) {
    const email = m[1].trim().toLowerCase();
    await supabase.from('identifiers').upsert([{ user_id: userId, type: 'email', value: email }]);
    await supabase.from('memories').upsert([{ user_id: userId, summary: { email }, last_updated: nowISO() }]);
    return `Linked email: ${email}`;
  }

  // SET NAME Ashley
  m = /^set\s+name\s+(.+)$/i.exec(t);
  if (m) {
    const name = m[1].trim();
    await supabase.from('memories').upsert([{ user_id: userId, summary: { name, __force: true }, last_updated: nowISO() }]);
    return `Saved your name as ${name}.`;
  }

  // GET MEMORY
  if (/^get\s+memory$/i.test(t)) {
    const { data } = await supabase.from('memories').select('summary').eq('user_id', userId).maybeSingle();
    return `Memory:\n${JSON.stringify(data?.summary ?? {}, null, 2)}`;
  }

  // FORGET NAME
  if (/^forget\s+name$/i.test(t)) {
    const { data } = await supabase.from('memories').select('summary').eq('user_id', userId).maybeSingle();
    const s = data?.summary ?? {};
    delete s.name;
    await supabase.from('memories').upsert([{ user_id: userId, summary: s, last_updated: nowISO() }]);
    return `Okay, I’ve removed your saved name.`;
  }

  // RESET MEMORY
  if (/^reset\s+memory$/i.test(t)) {
    await supabase.from('memories').upsert([{ user_id: userId, summary: {}, last_updated: nowISO() }]);
    return `Memory reset for this account.`;
  }

  return null; // not a command
}

/** Extract structured facts + refreshed short summary via LLM (JSON) */
async function extractFacts(oldSummary, userMessage) {
  const sys = `
You maintain compact, stable profile memory about a person.
Only extract facts that are clearly stated or strongly implied by the user's message.
Return strict JSON with keys below. Keep it minimal. Do NOT invent data.

Schema:
{
  "name": string|undefined,
  "nickname": string|undefined,
  "pronouns": string|undefined,
  "email": string|undefined,
  "location": string|undefined,
  "timezone": string|undefined,
  "interests": string[]|undefined,      // short tags like ["golf","cars"]
  "tone": string|undefined,              // e.g., "friendly", "concise"
  "household": { "children": number|undefined }|undefined,
  "projects": [{"title": string, "note": string}]|undefined,
  "reminders": [{"text": string}]|undefined,
  "summary": string|undefined,           // <= 240 chars, rolling relationship summary mentioning only durable info
  "__force": boolean|undefined           // if user explicitly set a field (e.g., "My name is X"), allow overwrite
}`;
  const res = await openai.chat.completions.create({
    model: 'gpt-4o-mini',
    response_format: { type: 'json_object' },
    messages: [
      { role: 'system', content: sys },
      { role: 'user', content: `Existing memory (may be empty): ${JSON.stringify(oldSummary ?? {})}` },
      { role: 'user', content: `New message: ${userMessage}` }
    ],
    temperature: 0
  });
  const json = res.choices?.[0]?.message?.content || '{}';
  try { return JSON.parse(json); } catch { return {}; }
}

/** Create Twilio-compatible XML response */
const twiml = (msg) => `<Response><Message>${escapeXml(msg)}</Message></Response>`;
function escapeXml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(200).send('Limi webhook is alive.');

  // 1) Parse inbound
  const { from, body } = getFromBody(req);
  if (!from || !isStr(body)) return res.status(400).send('Missing From/Body');

  // 2) Resolve user by identifier
  let { data: identifier } = await supabase
    .from('identifiers').select('user_id').eq('value', from).maybeSingle();

  let userId;
  if (!identifier) {
    const { data: user } = await supabase.from('users')
      .insert([{ display_name: null }]).select().single();
    await supabase.from('identifiers').insert([{ user_id: user.id, type: 'phone', value: from }]);
    userId = user.id;
  } else userId = identifier.user_id;

  // 3) Commands (short-circuit)
  const cmdReply = await handleCommands(body, userId);
  if (cmdReply) {
    res.setHeader('Content-Type', 'text/xml');
    return res.status(200).send(twiml(cmdReply));
  }

  // 4) Load memory
  const { data: memRow } = await supabase.from('memories')
    .select('summary').eq('user_id', userId).maybeSingle();
  let summary = memRow?.summary || {};

  // 5) Lightweight explicit patterns (fast path)
  const nameMatch = /\bmy name is\s+([a-z][a-z'-]{1,30})\b/i.exec(body);
  if (nameMatch) summary = mergeMemory(summary, { name: nameMatch[1], __force: true });

  // 6) LLM extraction to enrich memory
  const extracted = await extractFacts(summary, body);
  summary = mergeMemory(summary, extracted);

  // 7) Compose assistant response using memory
  const systemPrompt = `You are Limi, a warm, concise assistant. 
You have structured memory about the user. 
Use it when helpful (e.g., greet by name, recall past facts), but don't hallucinate.
If unsure, ask a short clarifying question.
Prefer British English.`;

  const memoryForModel = {
    name: summary.name, nickname: summary.nickname, pronouns: summary.pronouns,
    location: summary.location, timezone: summary.timezone, interests: summary.interests,
    tone: summary.tone, household: summary.household, projects: summary.projects,
    summary: summary.summary
  };

  const chat = await openai.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: `Memory: ${JSON.stringify(memoryForModel)}` },
      { role: 'user', content: `User said: ${body}` }
    ]
  });
  const reply = chat.choices[0].message.content;

  // 8) Persist message + memory
  summary.lastMessage = body;
  summary.lastReply = reply;
  summary.last_seen = nowISO();

  await supabase.from('messages').insert([
    { user_id: userId, channel: 'sms', external_id: from, body }
  ]);
  await supabase.from('memories').upsert([
    { user_id: userId, summary, last_updated: nowISO() }
  ]);

  // 9) Respond
  res.setHeader('Content-Type', 'text/xml');
  return res.status(200).send(twiml(reply));
}

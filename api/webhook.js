// api/webhook.js
import { createClient } from '@supabase/supabase-js';
import OpenAI from 'openai';

// Let us read raw form data from Twilio
export const config = { api: { bodyParser: false } };

const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);
const openai = new OpenAI({ apiKey: process.env.OPENAI_KEY });

export default async function handler(req, res) {
  try {
    if (req.method === 'GET') return res.status(200).send('Limi webhook is alive');

    if (req.method !== 'POST') return res.status(405).send('Method Not Allowed');

    // --- Parse Twilio x-www-form-urlencoded body ---
    const chunks = [];
    for await (const c of req) chunks.push(c);
    const raw = Buffer.concat(chunks).toString('utf8');
    const params = new URLSearchParams(raw);
    const rawFrom = params.get('From') || '';
    const from = rawFrom
    .replace(/^whatsapp:/i, '')   // remove "whatsapp:"
    .replace(/^sms:/i, '')        // just in case Twilio adds "sms:"
    .trim();
    const body = (params.get('Body') || '').trim();

    if (!from || !body) return res.status(400).send('Missing From/Body');

    // --- Find or create user by phone ---
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

    // --- Load existing memory ---
    const { data: mem, error: memSelErr } = await supabase
      .from('memories').select('summary').eq('user_id', userId).maybeSingle();
    if (memSelErr) console.error('mem select err', memSelErr);

    const summary = mem?.summary ?? {};

    // --- Extract facts from this message ---
    const nameMatch = /my name is\s+([a-zA-Z'-]+)/i.exec(body);
    if (nameMatch) summary.name = nameMatch[1];

    const locMatch = /i live in\s+(.+)/i.exec(body);
    if (locMatch) summary.location = locMatch[1].trim();

    summary.lastMessage = body;

    // --- Persist memory immediately (so next turn has it even if GPT fails) ---
    const { error: upErr1 } = await supabase
      .from('memories').upsert({ user_id: userId, summary });
    if (upErr1) console.error('mem upsert err (pre)', upErr1);

    // --- Log inbound message ---
    const { error: msgErr } = await supabase
      .from('messages').insert([{ user_id: userId, channel: 'sms', external_id: from, body }]);
    if (msgErr) console.error('message insert err', msgErr);

    // --- Ask GPT (keep replies <120 chars for Twilio trial) ---
    const completion = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        { role: 'system', content: 'You are Limi. Use facts in memory (name, location). Keep replies under 120 characters.' },
        { role: 'user', content: `Memory: ${JSON.stringify(summary)}` },
        { role: 'user', content: `New message: ${body}` }
      ],
      max_tokens: 80
    });
    const reply = completion.choices[0].message.content?.trim() || 'OK';

    summary.lastReply = reply;

    // --- Save updated memory again (captures lastReply) ---
    const { error: upErr2 } = await supabase
      .from('memories').upsert({ user_id: userId, summary });
    if (upErr2) console.error('mem upsert err (post)', upErr2);

    // --- Respond to Twilio (TwiML) ---
    res.setHeader('Content-Type', 'text/xml');
    return res.status(200).send(`<Response><Message>${reply}</Message></Response>`);
  } catch (e) {
    console.error('handler fatal', e);
    res.setHeader('Content-Type', 'text/xml');
    return res.status(200).send(`<Response><Message>Sorry, something went wrong.</Message></Response>`);
  }
}

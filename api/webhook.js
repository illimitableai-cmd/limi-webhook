import { createClient } from '@supabase/supabase-js';
import OpenAI from 'openai';

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_KEY
);

const openai = new OpenAI({ apiKey: process.env.OPENAI_KEY });

export default async function handler(req, res) {
  if (req.method === 'POST') {
    const ct = req.headers['content-type'] || '';
    let from, body;

    // Twilio sends form-urlencoded, but sometimes JSON if tested manually
    if (ct.includes('application/x-www-form-urlencoded')) {
      const parsed =
        typeof req.body === 'string'
          ? Object.fromEntries(new URLSearchParams(req.body))
          : req.body;
      from = parsed.From;
      body = parsed.Body;
    } else if (ct.includes('application/json')) {
      from = req.body?.From;
      body = req.body?.Body;
    }

    if (!from || !body) {
      return res.status(400).send('Missing From/Body');
    }

    // --- Find or create user ---
    let { data: identifier } = await supabase
      .from('identifiers')
      .select('user_id')
      .eq('value', from)
      .maybeSingle();

    let userId;
    if (!identifier) {
      const { data: user } = await supabase
        .from('users')
        .insert([{ display_name: null }])
        .select()
        .single();

      await supabase.from('identifiers').insert([
        { user_id: user.id, type: 'phone', value: from }
      ]);

      userId = user.id;
    } else {
      userId = identifier.user_id;
    }

    // --- Get memory ---
    let { data: mem } = await supabase
      .from('memories')
      .select('summary')
      .eq('user_id', userId)
      .maybeSingle();

    let summary = mem ? mem.summary : {};

    // --- Simple name extractor ---
    const nameMatch = body.match(/my name is (\w+)/i);
    if (nameMatch) {
      summary.name = nameMatch[1];
    }

    // --- Call GPT with memory ---
    const completion = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        {
          role: 'system',
          content:
            'You are Limi, the helpful AI with memory. Use the stored facts when replying.'
        },
        { role: 'user', content: `Memory: ${JSON.stringify(summary)}` },
        { role: 'user', content: `New message: ${body}` }
      ]
    });

    const reply = completion.choices[0].message.content;

    // --- Save message + update memory ---
    summary.lastMessage = body;
    summary.lastReply = reply;

    await supabase.from('messages').insert([
      { user_id: userId, channel: 'sms', external_id: from, body }
    ]);
    await supabase
      .from('memories')
      .upsert([{ user_id: userId, summary, last_updated: new Date() }]);

    // --- Reply back to Twilio ---
    res.setHeader('Content-Type', 'text/xml');
    res.status(200).send(`<Response><Message>${reply}</Message></Response>`);
  } else {
    res.status(200).send('Limi webhook is alive.');
  }
}

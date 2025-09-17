import { createClient } from '@supabase/supabase-js';
import OpenAI from 'openai';

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_KEY
);

const openai = new OpenAI({ apiKey: process.env.OPENAI_KEY });

export default async function handler(req, res) {
  if (req.method === 'POST') {
    const from = req.body.From;
    const body = req.body.Body;

    // find or create user
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
      await supabase
        .from('identifiers')
        .insert([{ user_id: user.id, type: 'phone', value: from }]);
      userId = user.id;
    } else {
      userId = identifier.user_id;
    }

    // fetch memory
    let { data: mem } = await supabase
      .from('memories')
      .select('summary')
      .eq('user_id', userId)
      .maybeSingle();
    const memory = mem ? JSON.stringify(mem.summary) : 'No memory yet.';

    // ask GPT
    const completion = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        { role: 'system', content: 'You are Limi, the helpful AI with memory.' },
        { role: 'user', content: `Past memory: ${memory}` },
        { role: 'user', content: `New message: ${body}` }
      ]
    });
    const reply = completion.choices[0].message.content;

    // save conversation + memory
    await supabase.from('messages').insert([
      { user_id: userId, channel: 'sms', external_id: from, body }
    ]);
    await supabase.from('memories').upsert([
      { user_id: userId, summary: { lastMessage: body, lastReply: reply } }
    ]);

    // reply to Twilio
    res.setHeader('Content-Type', 'text/xml');
    res.status(200).send(`<Response><Message>${reply}</Message></Response>`);
  } else {
    res.status(200).send('Limi webhook is alive.');
  }
}

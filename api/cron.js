// api/cron.js
import { createClient } from '@supabase/supabase-js';
import twilio from 'twilio';

const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);
const twilioClient = twilio(process.env.TWILIO_SID, process.env.TWILIO_AUTH);
const TWILIO_FROM = process.env.TWILIO_FROM;

// (Optional) protect the endpoint if you added CRON_SECRET in Vercel env
// export default async function handler(req, res) {
//   if (req.headers.get('authorization') !== `Bearer ${process.env.CRON_SECRET}`) {
//     return res.status(401).send('Unauthorized');
//   }
//   ...rest...

export default async function handler(req, res) {
  const now = new Date().toISOString();

  const { data: due, error } = await supabase
    .from('reminders')
    .select('id,user_id,text,target_contact_id')
    .lte('run_at', now)
    .eq('status', 'scheduled')
    .limit(50);

  if (error) return res.status(500).json({ error });

  for (const r of (due || [])) {
    let to = null;

    if (r.target_contact_id) {
      const { data: c } = await supabase
        .from('contacts')
        .select('phone')
        .eq('id', r.target_contact_id)
        .single();
      to = c?.phone || null;
    }

    if (!to) {
      const { data: id } = await supabase
        .from('identifiers')
        .select('value')
        .eq('user_id', r.user_id)
        .eq('type', 'phone')
        .maybeSingle();
      to = id?.value || null;
    }

    try {
      if (!to) throw new Error('no destination');
      await twilioClient.messages.create({
        to,
        from: TWILIO_FROM,
        body: r.text.slice(0, 320)
      });
      await supabase.from('reminders').update({ status: 'sent' }).eq('id', r.id);
    } catch (e) {
      await supabase
        .from('reminders')
        .update({ status: 'failed', meta: { error: String(e) } })
        .eq('id', r.id);
    }
  }

  return res.status(200).json({ sent: due?.length || 0 });
}

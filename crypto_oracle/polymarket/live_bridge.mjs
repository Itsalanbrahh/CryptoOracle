import { PolymarketUS } from 'polymarket-us';

function fail(message, extra = {}) {
  process.stdout.write(JSON.stringify({ ok: false, error: message, ...extra }, null, 2));
  process.exit(1);
}

const raw = process.argv[2];
if (!raw) fail('missing JSON payload');

let payload;
try {
  payload = JSON.parse(raw);
} catch (err) {
  fail('invalid JSON payload', { detail: String(err) });
}

const {
  marketSlug,
  intent,
  price,
  quantity,
  tif = 'TIME_IN_FORCE_GOOD_TILL_CANCEL',
  type = 'ORDER_TYPE_LIMIT',
  dryRun = true,
} = payload;

if (!marketSlug) fail('marketSlug is required');
if (!intent) fail('intent is required');
if (!(price > 0 && price < 1)) fail('price must be between 0 and 1');
if (!(quantity > 0)) fail('quantity must be positive');

if (dryRun) {
  process.stdout.write(JSON.stringify({
    ok: true,
    mode: 'live',
    dry_run: true,
    order: {
      marketSlug,
      intent,
      type,
      tif,
      price: { value: String(price), currency: 'USD' },
      quantity,
    },
    note: 'Dry run only. No live Polymarket order was submitted.'
  }, null, 2));
  process.exit(0);
}

const keyId = process.env.POLYMARKET_KEY_ID;
const secretKey = process.env.POLYMARKET_SECRET_KEY;
if (!keyId || !secretKey) fail('missing POLYMARKET_KEY_ID or POLYMARKET_SECRET_KEY');
if (process.env.POLYMARKET_LIVE_ENABLED !== '1') {
  fail('POLYMARKET_LIVE_ENABLED is not set to 1; refusing live order submission');
}

const client = new PolymarketUS({ keyId, secretKey });

let created;
try {
  created = await client.orders.create({
    marketSlug,
    intent,
    type,
    price: { value: String(price), currency: 'USD' },
    quantity,
    tif,
  });
} catch (err) {
  fail('order submission failed', { detail: String(err?.message || err) });
}

// Reject if the API returned an error body
if (created && (created.error || created.errorCode || created.message?.toLowerCase().includes('error'))) {
  fail('order rejected by Polymarket', { response: created });
}

process.stdout.write(JSON.stringify({
  ok: true,
  mode: 'live',
  dry_run: false,
  submitted_at: new Date().toISOString(),
  order_id: created?.id || created?.orderId || null,
  response: created,
}, null, 2));

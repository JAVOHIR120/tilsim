'use strict';
/* Tilsim landing page. One shared <audio>; a Web Audio analyser makes the playing orb breathe with the voice. */

const SAMPLES = [
  { id: 'hikoya', orb: 'orb-oltin', title: 'Hikoya', desc: 'Tinch va iliq ovoz: audiokitob va podkastlar uchun.',
    text: 'O‘zbekiston Markaziy Osiyoning yuragida joylashgan bo‘lib, uning boy tarixi va mehmondo‘st xalqi butun dunyoga mashhur.' },
  { id: 'raqam', orb: 'orb-feruza', title: 'Raqam va sana', desc: '«1991-yil 31-avgust» ni o‘zbekcha so‘z bilan o‘qiydi.',
    text: '1991-yil 31-avgustda O‘zbekiston mustaqilligini e’lon qildi.' },
  { id: 'savol', orb: 'orb-lola', title: 'Savol ohangi', desc: 'So‘roq gaplarda ohang tabiiy ko‘tariladi.',
    text: 'Siz ertaga kelasizmi?' },
  { id: 'narx', orb: 'orb-anor', title: 'Reklama', desc: 'Narx va foizlarni aniq, ishonchli ovozda aytadi.',
    text: 'Mahsulot narxi 1 250 000 so‘m, chegirma esa 15%.',
    parts: ['mahsulot', 'narxi', 'bir million', 'ikki yuz ellik', 'ming', 'so‘m', 'chegirma', 'esa', 'o‘n besh foiz'] },
  { id: 'suhbat', orb: 'orb-yalpiz', title: 'Suhbat', desc: 'Kundalik, samimiy muloqot uchun yengil ovoz.',
    text: 'Assalomu alaykum, ahvollaringiz yaxshimi?' },
  { id: 'talaffuz', orb: 'orb-lapis', title: 'Talaffuz', desc: 'Qo‘l va ko‘l, o‘ va o: har bir harf o‘z o‘rnida.',
    text: 'Qo‘lingizni ko‘lga solmang, suv juda sovuq.' },
  { id: 'tutuq', orb: 'orb-sadaf', title: 'Tutuq belgisi', desc: 'Ta’lim, san’at, ma’rifat: tutuq aniq eshitiladi.',
    text: 'Ta’lim va tarbiya, san’at va ma’rifat bir-biriga bog‘liq.' },
  { id: 'qiyin', orb: 'orb-tun', title: 'Qiyin tovushlar', desc: 'Q, g‘, sh va ch bir gapda ham adashmaydi.',
    text: 'Qo‘shni qishloqdagi qo‘g‘irchoqboz qiziqarli tomosha ko‘rsatdi.' },
];
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const src = (id) => `assets/audio/${id}.mp3`;
const PLAY = '<svg viewBox="0 0 24 24" aria-hidden="true"><path class="i-play" d="M8 5v14l11-7z"/><path class="i-pause" d="M7 5h4v14H7zM13 5h4v14h-4z"/></svg>';

/* ------------------------------------------------ audio */
const player = $('#player');
let ctx = null, analyser = null, buf = null;
let current = null; // { id, onStop(), onFrame(level, progress) }

function ensureAnalyser() {
  if (ctx) return;
  try {
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    const node = ctx.createMediaElementSource(player);
    analyser = ctx.createAnalyser();
    analyser.fftSize = 256;
    analyser.smoothingTimeConstant = 0.8;
    node.connect(analyser).connect(ctx.destination);
    buf = new Uint8Array(analyser.fftSize);
  } catch { ctx = null; }
}

function stop() {
  player.pause();
  const c = current;
  current = null;
  if (c) c.onStop();
}

async function play(id, handlers, url = src(id)) {
  stop();
  ensureAnalyser();
  if (ctx && ctx.state === 'suspended') await ctx.resume();
  const abs = new URL(url, location.href).href;
  if (player.src !== abs) player.src = abs;
  player.currentTime = 0;
  try {
    await player.play();
    current = { id, ...handlers };
    tick();
  } catch { handlers.onStop(); }
}
player.addEventListener('ended', stop);

let raf = 0;
function tick() {
  cancelAnimationFrame(raf);
  const step = () => {
    if (!current) return;
    let level = 0;
    if (analyser) {
      analyser.getByteTimeDomainData(buf);
      let sum = 0;
      for (let i = 0; i < buf.length; i++) { const v = (buf[i] - 128) / 128; sum += v * v; }
      level = Math.min(1, Math.sqrt(sum / buf.length) * 4);
    }
    current.onFrame?.(level, player.duration ? player.currentTime / player.duration : 0);
    raf = requestAnimationFrame(step);
  };
  raf = requestAnimationFrame(step);
}

/* ------------------------------------------------ voice orbs */
const orbsEl = $('#orbs');
const pillsEl = $('#orb-pills');
let center = 0;
const items = SAMPLES.map((s, i) => {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'orb-item';
  b.setAttribute('aria-label', `${s.title}: tinglash`);
  b.innerHTML = `<span class="orb ${s.orb}"></span><span class="play">${PLAY}</span>`;
  b.addEventListener('click', () => (i === center ? toggleOrb() : setCenter(i)));
  orbsEl.append(b);
  const p = document.createElement('button');
  p.type = 'button';
  p.className = 'pill';
  p.setAttribute('role', 'tab');
  p.textContent = s.title;
  p.addEventListener('click', () => setCenter(i));
  pillsEl.append(p);
  return { b, p, s };
});

function spacing() { return innerWidth < 720 ? 168 : 300; }

function layout(level = 0) {
  const n = items.length, gap = spacing();
  items.forEach(({ b }, i) => {
    let off = i - center;
    if (off > n / 2) off -= n;
    if (off < -n / 2) off += n;
    const a = Math.abs(off);
    const scale = a === 0 ? 1 + level * 0.07 : a === 1 ? 0.78 : 0.56;
    const x = off * gap * (a > 1 ? 0.86 : 1);
    b.style.transform = `translateX(${x}px) scale(${scale})`;
    b.style.opacity = a === 0 ? 1 : a === 1 ? 0.95 : a === 2 ? 0.55 : 0;
    b.style.zIndex = String(10 - a);
    b.style.pointerEvents = a > 2 ? 'none' : 'auto';
    b.tabIndex = a === 0 ? 0 : -1;
    b.classList.toggle('center', a === 0);
  });
}

let onSample = null; // set by the live composer below: a picked sample fills the text box
function setCenter(i, instant = false) {
  const n = items.length;
  center = ((i % n) + n) % n;
  if (current && current.id !== items[center].s.id && current.orb) stop();
  const s = items[center].s;
  if (onSample) onSample(s);
  $('#orb-title').textContent = s.title;
  $('#orb-desc').textContent = s.desc;
  items.forEach(({ p }, k) => p.setAttribute('aria-selected', String(k === center)));
  // scroll only the pill row (scrollIntoView would also scroll the page)
  const pill = items[center].p;
  // instant on first load: a smooth scroll here would cancel the browser's jump to a #section link
  const left = pill.offsetLeft - (pillsEl.clientWidth - pill.offsetWidth) / 2;
  if (instant || reduce) pillsEl.scrollLeft = left;
  else pillsEl.scrollTo({ left, behavior: 'smooth' });
  layout();
}

function toggleOrb() {
  const it = items[center];
  if (current && current.orb === it) { stop(); return; }
  if (genReady()) { playGenerated(lastUrl); return; } // the box holds text just voiced: the orb plays that
  play(it.s.id, {
    orb: it,
    onFrame: (level) => layout(level),
    onStop: () => { it.b.classList.remove('playing'); it.b.setAttribute('aria-label', `${it.s.title}: tinglash`); layout(); },
  });
  it.b.classList.add('playing');
  it.b.setAttribute('aria-label', `${it.s.title}: to‘xtatish`);
}

$('#orb-prev').addEventListener('click', () => setCenter(center - 1));
$('#orb-next').addEventListener('click', () => setCenter(center + 1));
orbsEl.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowLeft') { e.preventDefault(); setCenter(center - 1); items[center].b.focus(); }
  if (e.key === 'ArrowRight') { e.preventDefault(); setCenter(center + 1); items[center].b.focus(); }
});
// swipe on touch screens
let touchX = null;
orbsEl.addEventListener('touchstart', (e) => { touchX = e.touches[0].clientX; }, { passive: true });
orbsEl.addEventListener('touchend', (e) => {
  if (touchX === null) return;
  const dx = e.changedTouches[0].clientX - touchX;
  if (Math.abs(dx) > 40) setCenter(center + (dx < 0 ? 1 : -1));
  touchX = null;
});
addEventListener('resize', () => layout());
setCenter(0, true);
// first layout lands without animation; later moves glide
requestAnimationFrame(() => requestAnimationFrame(() => orbsEl.classList.add('ready')));

/* ------------------------------------------------ product tabs */
const tabs = $$('.seg-btn');
tabs.forEach((t) => t.addEventListener('click', () => {
  tabs.forEach((x) => {
    const on = x === t;
    x.setAttribute('aria-selected', String(on));
    $(`#${x.getAttribute('aria-controls')}`).hidden = !on;
  });
  stop();
}));

/* ------------------------------------------------ live voice: engine/serve.py on the owner's GPU, behind a tunnel */
// The tunnel address changes on every start, so start-tilsim.ps1 writes it to this gist; the page reads it here.
const GIST = 'https://api.github.com/gists/5801b83645819ab8d22340b5b60798f2';
const MAX = 300;
const ta = $('#tts-text'), goBtn = $('#tts-go'), dl = $('#tts-download'), msg = $('#tts-msg');
const statusEl = $('#tts-status'), statusText = $('#tts-status-text'), countEl = $('#tts-count');
let api = null, online = false, busy = false, recheck = 0, userEdited = false;
let lastUrl = null, lastText = '';

const setStatus = (state, text) => { statusEl.dataset.state = state; statusText.textContent = text; };
const note = (text, err = false) => { msg.textContent = text; msg.classList.toggle('err', err); };
const isSample = (t) => SAMPLES.some((s) => s.text === t);
const genReady = () => !!lastUrl && ta.value.trim() === lastText;
const LABELS = { busy: 'Yaratilmoqda…', stop: 'To‘xtatish', play: 'Tinglash', gen: 'Ovoz yaratish' };

function syncBtn() {
  const mode = busy ? 'busy' : current && current.id === 'gen' ? 'stop' : genReady() ? 'play' : 'gen';
  // toggleAttribute, not .hidden: SVG elements have no `hidden` property
  $('.ic-gen', goBtn).toggleAttribute('hidden', mode !== 'gen');
  $('.ic-play', goBtn).toggleAttribute('hidden', mode !== 'play');
  $('.ic-stop', goBtn).toggleAttribute('hidden', mode !== 'stop');
  $('.spin', goBtn).toggleAttribute('hidden', mode !== 'busy');
  $('span', goBtn).textContent = LABELS[mode];
  goBtn.disabled = busy;
  goBtn.setAttribute('aria-busy', String(busy));
}

function count() {
  ta.style.height = 'auto'; // grow with the text (up to the CSS max-height), so long texts need no inner scrollbar
  ta.style.height = `${ta.scrollHeight}px`;
  countEl.textContent = `${ta.value.length} / ${MAX}`;
  countEl.classList.toggle('over', ta.value.length >= MAX);
  dl.hidden = !genReady();
  syncBtn();
}

// Phones (iOS) only let audio start inside a tap. The voice arrives seconds later, so the shared player is
// started (muted, silent) during the tap; later play() calls on the same element are then allowed.
const SILENT = 'data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=';
let unlocked = false;
function unlockAudio() {
  ensureAnalyser();
  if (ctx && ctx.state === 'suspended') ctx.resume();
  if (unlocked || current) return;
  unlocked = true;
  player.muted = true;
  player.src = SILENT;
  player.play().then(() => player.pause()).catch(() => {}).finally(() => { player.muted = false; });
}

// Two possible servers: the owner's laptop (free, fast, only while it runs start-tilsim.bat) and the cloud copy
// on Modal (always there, but it sleeps and every minute costs credit). The laptop wins when it answers.
// The cloud gets no health check, because any request would wake (and bill) its GPU; it is only called on «Ovoz yaratish».
let cloud = false;
const trim = (u) => String(u).replace(/\/$/, '');

async function alive(url) {
  try {
    const r = await fetch(`${url}/v1/health`, { cache: 'no-store', signal: AbortSignal.timeout(6000) });
    return r.ok;
  } catch { return false; }
}

async function findApi() {
  cloud = false;
  const q = new URLSearchParams(location.search).get('api');
  if (q) return trim(q);
  if (/^(localhost|127\.0\.0\.1)$/.test(location.hostname)) return 'http://127.0.0.1:8790';
  try {
    const r = await fetch(GIST, { cache: 'no-store', headers: { Accept: 'application/vnd.github+json' } });
    const cfg = JSON.parse((await r.json()).files['tilsim-api.json'].content);
    if (cfg.laptop && await alive(trim(cfg.laptop))) return trim(cfg.laptop);
    if (cfg.cloud) { cloud = true; return trim(cfg.cloud); }
    return cfg.api ? trim(cfg.api) : null; // older single-address entry
  } catch { return null; }
}

function offline() {
  online = false;
  setStatus('off', 'Server hozir o‘chiq: tayyor namunalarni tinglang');
  recheck = setTimeout(checkHealth, 90000); // the gist API allows 60 reads an hour per visitor
}

async function checkHealth() {
  clearTimeout(recheck);
  if (document.hidden) { recheck = setTimeout(checkHealth, 30000); return; }
  if (!api) api = await findApi();
  if (!api) { offline(); return; }
  if (cloud) {
    online = true;
    setStatus('on', 'Bulutda: birinchi ovoz 1 daqiqagacha');
    return;
  }
  try {
    const r = await fetch(`${api}/v1/health`, { cache: 'no-store', signal: AbortSignal.timeout(7000) });
    const h = await r.json();
    if (!h.ready) {
      online = false;
      setStatus('wait', 'Ovoz modeli yuklanmoqda…');
      recheck = setTimeout(checkHealth, 8000);
      return;
    }
    online = true;
    const waiting = (h.busy || 0) + (h.queue || 0);
    setStatus('on', waiting ? `Jonli · navbatda ${waiting} ta so‘rov` : 'Jonli: asoschi ovozida yaratadi');
    recheck = setTimeout(checkHealth, 30000);
  } catch {
    api = null; // the tunnel may have restarted with a new address: read the gist again next time
    offline();
  }
}

function setBusy(on) {
  busy = on;
  items.forEach(({ b }, i) => b.classList.toggle('thinking', on && i === center));
  syncBtn();
}

function playGenerated(url) {
  const it = items[center];
  it.b.classList.add('playing');
  play('gen', {
    orb: it,
    onFrame: (level) => layout(level),
    onStop: () => { it.b.classList.remove('playing'); layout(); syncBtn(); },
  }, url).then(syncBtn);
}

async function generate(e) {
  e?.preventDefault();
  if (busy) return;
  if (current && current.id === 'gen') { stop(); return; }
  const text = ta.value.trim();
  if (!text) { note('Avval matn yozing.', true); ta.focus(); return; }
  unlockAudio();
  if (genReady()) { playGenerated(lastUrl); return; }
  if (!online) {
    const i = SAMPLES.findIndex((s) => s.text === text);
    if (i >= 0) {
      setCenter(i);
      toggleOrb();
      note('Server o‘chiq, shu matnning tayyor namunasi ijro etilmoqda.');
    } else {
      note('Ovoz serveri hozir o‘chiq. Yuqoridagi namunalarni tinglang yoki keyinroq urinib ko‘ring.', true);
      checkHealth();
    }
    return;
  }
  stop();
  setBusy(true);
  const t0 = performance.now();
  const tick = () => {
    const s = Math.round((performance.now() - t0) / 1000);
    note(cloud && s >= 12
      ? `Bulut serveri uyg‘onmoqda… ${s} s (birinchi ovoz 1 daqiqagacha, keyingilari tez)`
      : `Ovoz yaratilmoqda… ${s} s (har bir gap 5–8 soniya oladi)`);
  };
  tick();
  const timer = setInterval(tick, 1000);
  try {
    const r = await fetch(`${api}/v1/tts`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }),
    });
    if (!r.ok) {
      const j = await r.json().catch(() => ({}));
      note(j.error || `Xato (${r.status}). Qayta urinib ko‘ring.`, true);
      return;
    }
    const blob = await r.blob();
    if (lastUrl) URL.revokeObjectURL(lastUrl);
    lastUrl = URL.createObjectURL(blob);
    lastText = text;
    dl.href = lastUrl;
    count();
    const secs = parseFloat(r.headers.get('X-Tilsim-Seconds') || '0');
    note(secs ? `Tayyor: ${secs.toFixed(1).replace('.', ',')} soniyalik ovoz.` : 'Tayyor.');
    playGenerated(lastUrl);
  } catch {
    note('Server bilan aloqa uzildi. Birozdan keyin qayta urinib ko‘ring.', true);
    api = null;
    checkHealth();
  } finally {
    clearInterval(timer);
    setBusy(false);
  }
}

onSample = (s) => {
  if (userEdited) return;
  ta.value = s.text;
  count();
};
ta.addEventListener('input', () => {
  const t = ta.value.trim();
  userEdited = t !== '' && !isSample(t);
  count();
  if (msg.classList.contains('err')) note('');
});
ta.addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) generate(e); });
$('#composer').addEventListener('submit', generate);
onSample(items[center].s);
checkHealth();
document.addEventListener('visibilitychange', () => { if (!document.hidden && !online) checkHealth(); });

/* buttons that just play a sample (clone card) */
$$('[data-play]').forEach((btn) => btn.addEventListener('click', () => {
  if (current && current.btn === btn) { stop(); return; }
  play(btn.dataset.play, {
    btn,
    onStop: () => { btn.classList.remove('playing'); btn.querySelector('span').textContent = 'Tinglash'; },
  });
  btn.classList.add('playing');
  btn.querySelector('span').textContent = 'To‘xtatish';
}));

/* ------------------------------------------------ studio editor: karaoke over the written text */
const ed = $('#ed-text');
const edBtn = $('#ed-play');
const edSample = SAMPLES.find((s) => s.id === 'narx');
const edWords = edSample.text.split(' ');
ed.replaceChildren(...edWords.flatMap((w, i) => {
  const span = document.createElement('span');
  span.className = /\d/.test(w) ? 'w num' : 'w';
  span.textContent = w;
  return i ? [document.createTextNode(' '), span] : [span];
}));
const said = edSample.parts && edSample.parts.length === edWords.length ? edSample.parts : edWords;
const weights = said.map((w) => w.replace(/[^\p{L}\p{N}]/gu, '').length + 1.5 * w.split(' ').length);
const total = weights.reduce((a, b) => a + b, 0);
let acc = 0;
const ends = weights.map((w) => (acc += w) / total);
const edSpans = $$('.w', ed);
function markEditor(p) {
  const lead = 0.06 / (player.duration || 5), tail = 0.12 / (player.duration || 5);
  const q = Math.min(1, Math.max(0, (p - lead) / (1 - lead - tail)));
  const idx = ends.findIndex((e) => q <= e);
  edSpans.forEach((s, i) => { s.classList.toggle('done', i < idx || idx === -1); s.classList.toggle('now', i === idx); });
}
edBtn.addEventListener('click', () => {
  if (current && current.btn === edBtn) { stop(); return; }
  play('narx', {
    btn: edBtn,
    onFrame: (_, p) => markEditor(p),
    onStop: () => { edBtn.classList.remove('playing'); edBtn.querySelector('span').textContent = 'Tinglash'; ed.classList.remove('playing'); },
  });
  ed.classList.add('playing');
  edBtn.classList.add('playing');
  edBtn.querySelector('span').textContent = 'To‘xtatish';
});

/* ------------------------------------------------ entrance + charts */
const io = new IntersectionObserver((entries) => {
  for (const e of entries) if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
}, { threshold: 0.15, rootMargin: '0px 0px -40px 0px' });
$$('.sec-head, .media, .tile, .plan, .faq, .stats, .logos, .final-box').forEach((el) => {
  if (reduce) return;
  el.classList.add('reveal');
  io.observe(el);
});
$$('.bars').forEach((el) => (reduce ? el.classList.add('in') : io.observe(el)));

/* ------------------------------------------------ nav */
const nav = $('#nav');
const onScroll = () => nav.classList.toggle('scrolled', scrollY > 40);
addEventListener('scroll', onScroll, { passive: true });
onScroll();
const menuBtn = $('#menu-btn'), menu = $('#menu');
menuBtn.addEventListener('click', () => {
  const open = menuBtn.getAttribute('aria-expanded') !== 'true';
  menuBtn.setAttribute('aria-expanded', String(open));
  menuBtn.setAttribute('aria-label', open ? 'Menyuni yopish' : 'Menyuni ochish');
  menu.classList.toggle('open', open);
});
menu.querySelectorAll('a').forEach((a) => a.addEventListener('click', () => { menu.classList.remove('open'); menuBtn.setAttribute('aria-expanded', 'false'); }));

/* placeholder contact (owner fills in) */
$$('[data-todo]').forEach((a) => a.addEventListener('click', (e) => { if (a.getAttribute('href') === '#') e.preventDefault(); }));

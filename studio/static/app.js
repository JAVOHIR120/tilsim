'use strict';
// Ovoz Studiyasi: record one sentence at a time, check the take, save it as WAV through the local server.

const $ = (s) => document.querySelector(s);
const PREROLL_SEC = 0.3;   // audio kept from just before you press record, so the first syllable is never cut
const MAX_SEC = 30;
const TARGET_SEC = 90 * 60;
const WORKLET = `
class Tap extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) this.port.postMessage(ch.slice(0));
    return true;
  }
}
registerProcessor('tap', Tap);`;

const S = {
  items: [], idx: 0, totalSeconds: 0,
  ctx: null, playCtx: null, stream: null, src: null, node: null,
  recording: false, chunks: [], recSamples: 0, preroll: [], prerollSamples: 0,
  measuring: null, noiseDb: null, take: null, player: null,
  framePeak: 0, level: -100, peakHold: -100, peakHoldAt: 0,
};

const dB = (x) => 20 * Math.log10(Math.max(x, 1e-9));
const fmtMin = (sec) => `${Math.floor(sec / 60)} daq ${String(Math.round(sec % 60)).padStart(2, '0')} s`;

async function api(path, opts) {
  const r = await fetch(path, opts);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

function toast(msg, kind = '') {
  const t = $('#toast');
  t.textContent = msg; t.className = `toast ${kind}`; t.hidden = false;
  clearTimeout(toast.timer); toast.timer = setTimeout(() => { t.hidden = true; }, 2800);
}

// ------------------------------------------------------------ script + list

async function load() {
  const st = await api('/api/state');
  S.items = st.items; S.totalSeconds = st.totalSeconds;
  $('#voiceName').textContent = st.voice;
  const first = S.items.findIndex((i) => !i.recorded);
  S.idx = first >= 0 ? first : 0;
  renderList(); render();
}

const current = () => S.items[S.idx];

function render() {
  const it = current();
  if (!it) return;
  $('#itemNo').textContent = `${S.idx + 1} / ${S.items.length} · ${it.id}${it.edited ? ' · tuzatilgan' : ''}`;
  $('#itemStatus').textContent = it.recorded ? `yozilgan ✓ ${it.duration.toFixed(1)} s` : '';
  $('#itemStatus').className = `small ${it.recorded ? 'ok' : ''}`;
  $('#spoken').textContent = it.spoken;
  $('#original').textContent = it.text !== it.spoken ? `Yozma shakli: ${it.text}` : '';
  const box = $('#issues');
  box.replaceChildren(...it.issues.map((msg) => Object.assign(document.createElement('div'), { className: 'warn', textContent: `⚠ ${msg}` })));
  if (!S.take && !S.recording) { clearWave(); $('#stats').replaceChildren(); }
  updateProgress(); updateButtons(); highlightList();
}

function updateProgress() {
  const done = S.items.filter((i) => i.recorded).length;
  $('#progressFill').style.width = `${Math.min(100, (S.totalSeconds / TARGET_SEC) * 100)}%`;
  $('#progressText').textContent = `${done}/${S.items.length} gap · ${fmtMin(S.totalSeconds)} / 90 daq`;
}

function updateButtons() {
  const it = current();
  $('#recBtn').disabled = !S.ctx;
  $('#recBtn').classList.toggle('on', S.recording);
  $('#recLabel').textContent = S.recording ? `■ To'xtatish ${(S.recSamples / S.ctx.sampleRate).toFixed(1)} s` : '● Yozish';
  $('#playBtn').disabled = S.recording || !(S.take || (it && it.recorded));
  $('#saveBtn').disabled = S.recording || !S.take;
}

function renderList() {
  const ol = $('#list');
  ol.replaceChildren(...S.items.map((it, i) => {
    const li = document.createElement('li');
    li.innerHTML = '<span class="dot"></span><span class="n"></span><span class="t"></span>';
    li.querySelector('.n').textContent = i + 1;
    li.querySelector('.t').textContent = it.spoken;
    li.title = it.spoken;
    li.addEventListener('click', () => go(i));
    return li;
  }));
  S.items.forEach((_, i) => renderListItem(i));
}

function renderListItem(i) {
  const li = $('#list').children[i];
  const it = S.items[i];
  if (!li) return;
  li.classList.toggle('done', it.recorded);
  li.classList.toggle('flag', !it.recorded && it.issues.length > 0);
  li.querySelector('.t').textContent = it.spoken;
}

function highlightList() {
  const ol = $('#list');
  ol.querySelector('.current')?.classList.remove('current');
  const li = ol.children[S.idx];
  if (li) { li.classList.add('current'); li.scrollIntoView({ block: 'nearest' }); }
}

function nextIndex() {
  for (let k = 1; k <= S.items.length; k++) {
    const i = (S.idx + k) % S.items.length;
    if (!S.items[i].recorded) return i;
  }
  return Math.min(S.idx + 1, S.items.length - 1);
}

function go(i) {
  if (S.recording || i < 0 || i >= S.items.length || i === S.idx) return;
  if (S.take && !confirm("Saqlanmagan yozuv o'chadi. Davom etilsinmi?")) return;
  S.take = null; stopPlayback(); closeEdit();
  S.idx = i; render();
}

// ------------------------------------------------------------ microphone

async function enableMic(deviceId) {
  try {
    if (S.stream) S.stream.getTracks().forEach((t) => t.stop());
    S.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: deviceId ? { exact: deviceId } : undefined, channelCount: 1, sampleRate: 48000,
        echoCancellation: false, noiseSuppression: false, autoGainControl: false,  // raw voice: a TTS must learn YOUR sound, not a filter's
      },
    });
  } catch (e) { toast(`Mikrofonga ruxsat yo'q: ${e.message}`, 'bad'); return; }
  if (!S.ctx) {
    S.ctx = new AudioContext({ sampleRate: 48000 });
    await S.ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET], { type: 'text/javascript' })));
    S.node = new AudioWorkletNode(S.ctx, 'tap', { channelCount: 1, channelCountMode: 'explicit' });
    S.node.port.onmessage = (e) => onFrame(e.data);
    const mute = S.ctx.createGain(); mute.gain.value = 0;
    S.node.connect(mute).connect(S.ctx.destination);  // keeps the tap running; nothing is heard
    requestAnimationFrame(drawMeter);
  }
  await S.ctx.resume();
  S.src?.disconnect();
  S.src = S.ctx.createMediaStreamSource(S.stream);
  S.src.connect(S.node);
  S.prerollSamples = Math.round(PREROLL_SEC * S.ctx.sampleRate);
  const track = S.stream.getAudioTracks()[0];
  const set = track.getSettings();
  const flag = (on, name) => (on ? `${name} YONIQ ✗` : `${name} o'chiq ✓`);
  $('#micInfo').textContent = [track.label || 'Mikrofon', `${S.ctx.sampleRate} Hz`,
    flag(set.echoCancellation, 'aks-sado tozalash'), flag(set.noiseSuppression, 'shovqin bostirish'),
    flag(set.autoGainControl, 'avto-balandlik')].join(' · ');
  await listDevices(set.deviceId);
  $('#micBtn').textContent = 'Mikrofon yoniq';
  $('#noiseBtn').disabled = false;
  updateButtons();
}

async function listDevices(activeId) {
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'audioinput');
  const sel = $('#deviceSelect');
  sel.replaceChildren(...devices.map((d, i) => new Option(d.label || `Mikrofon ${i + 1}`, d.deviceId, false, d.deviceId === activeId)));
  sel.hidden = devices.length < 2;
}

function onFrame(f) {
  let peak = 0;
  for (let i = 0; i < f.length; i++) { const a = Math.abs(f[i]); if (a > peak) peak = a; }
  if (peak > S.framePeak) S.framePeak = peak;
  if (S.recording) {
    S.chunks.push(f); S.recSamples += f.length;
    if (S.recSamples / S.ctx.sampleRate > MAX_SEC) stopRec();
  } else {
    S.preroll.push(f);
    let n = S.preroll.reduce((a, c) => a + c.length, 0);
    while (n - S.preroll[0].length >= S.prerollSamples) n -= S.preroll.shift().length;
  }
  if (S.measuring) {
    S.measuring.push(f);
    if (S.measuring.length * f.length >= 3 * S.ctx.sampleRate) finishNoise();
  }
}

function drawMeter(t) {
  const d = dB(S.framePeak); S.framePeak = 0;
  S.level = Math.max(d, S.level - 1.2);
  if (d >= S.peakHold || t - S.peakHoldAt > 1500) { S.peakHold = d; S.peakHoldAt = t; }
  const pct = (x) => Math.min(100, Math.max(0, ((x + 60) / 60) * 100));  // scale: −60 … 0 dBFS
  const fill = $('#meterFill');
  fill.style.width = `${pct(S.level)}%`;
  fill.className = S.level > -3 ? 'hot' : S.level > -18 ? 'good' : S.level > -40 ? 'ok' : '';
  $('#meterPeak').style.left = `${pct(S.peakHold)}%`;
  $('#meterDb').textContent = S.peakHold > -90 ? `${S.peakHold.toFixed(0)} dB` : '—';
  if (S.recording) $('#recLabel').textContent = `■ To'xtatish ${(S.recSamples / S.ctx.sampleRate).toFixed(1)} s`;
  requestAnimationFrame(drawMeter);
}

function frameDb(x, sr) {
  const hop = Math.round(sr * 0.02);
  const out = [];
  for (let i = 0; i + hop <= x.length; i += hop) {
    let s = 0;
    for (let j = i; j < i + hop; j++) s += x[j] * x[j];
    out.push(10 * Math.log10(s / hop + 1e-12));
  }
  return { frames: out, hop };
}

function concat(chunks) {
  const all = new Float32Array(chunks.reduce((a, c) => a + c.length, 0));
  let o = 0;
  for (const c of chunks) { all.set(c, o); o += c.length; }
  return all;
}

function measureNoise() {
  if (!S.ctx || S.recording) return;
  S.measuring = [];
  $('#noiseBtn').disabled = true;
  $('#noiseInfo').textContent = 'Jim turing… 3 soniya';
}

function finishNoise() {
  const { frames } = frameDb(concat(S.measuring), S.ctx.sampleRate);
  S.measuring = null;
  const sorted = [...frames].sort((a, b) => a - b);
  S.noiseDb = sorted[Math.floor(sorted.length / 2)];
  const n = S.noiseDb;
  const verdict = n <= -60 ? "a'lo" : n <= -50 ? 'yaxshi' : n <= -42 ? "o'rtacha, jimroq bo'lsa yaxshi" : 'shovqinli, boshqa joy toping';
  $('#noiseInfo').textContent = `Xona shovqini: ${n.toFixed(0)} dB, ${verdict}`;
  $('#noiseBtn').disabled = false;
}

// ------------------------------------------------------------ recording + checks

function startRec() {
  if (!S.ctx || S.recording) return;
  stopPlayback();
  S.take = null;
  S.chunks = S.preroll.slice();
  S.recSamples = S.chunks.reduce((a, c) => a + c.length, 0);
  S.recording = true;
  clearWave(); $('#stats').replaceChildren(); updateButtons();
}

function stopRec() {
  if (!S.recording) return;
  S.recording = false;
  const all = concat(S.chunks);
  S.chunks = [];
  const sr = S.ctx.sampleRate;
  const stats = analyze(all, sr, S.noiseDb, current().spoken);
  S.take = { samples: all.subarray(stats.trimStart, stats.trimEnd), sr, stats };
  drawWave(all, stats); showStats(stats); updateButtons();
}

function analyze(x, sr, noiseDb, text) {
  const { frames, hop } = frameDb(x, sr);
  let peak = 0, clipped = 0;
  for (let i = 0; i < x.length; i++) { const a = Math.abs(x[i]); if (a > peak) peak = a; if (a >= 0.999) clipped++; }
  const sorted = [...frames].sort((a, b) => a - b);
  const floor = noiseDb ?? (sorted.length ? sorted[Math.floor(sorted.length * 0.1)] : -90);
  const thr = Math.max(floor + 12, -55);
  const first = frames.findIndex((d) => d > thr);
  let last = -1;
  for (let i = frames.length - 1; i >= 0; i--) if (frames[i] > thr) { last = i; break; }
  const has = first >= 0;
  const voiced = frames.filter((d) => d > thr);
  const speechDb = voiced.length ? 10 * Math.log10(voiced.reduce((a, d) => a + 10 ** (d / 10), 0) / voiced.length) : -90;
  const duration = x.length / sr;
  const speech = has ? (last - first + 1) * 0.02 : 0;
  const pad = Math.round(0.3 * sr);
  const letters = (text.match(/\p{L}/gu) || []).length;
  const st = {
    duration: +duration.toFixed(2), speech: +speech.toFixed(2), peakDb: +dB(peak).toFixed(1),
    speechDb: +speechDb.toFixed(1), noiseDb: +floor.toFixed(1), snr: +(speechDb - floor).toFixed(1), clipped,
    lead: +(has ? first * 0.02 : duration).toFixed(2), trail: +(has ? (frames.length - 1 - last) * 0.02 : duration).toFixed(2),
    rate: +(speech > 0 ? letters / speech : 0).toFixed(1),
    trimStart: has ? Math.max(0, first * hop - pad) : 0, trimEnd: has ? Math.min(x.length, (last + 1) * hop + pad) : x.length,
    warnings: [], grade: 'good',
  };
  const W = (level, msg) => {
    st.warnings.push({ level, msg });
    if (level === 'bad') st.grade = 'bad'; else if (st.grade === 'good') st.grade = 'warn';
  };
  if (!has) { W('bad', 'Ovoz eshitilmadi.'); return st; }
  if (clipped > 0) W('bad', `Juda baland: ${clipped} nuqtada kesildi (clipping). Mikrofondan uzoqlashing yoki kuchaytirishni kamaytiring.`);
  else if (st.peakDb > -1) W('warn', "Cho'qqi −1 dB dan baland, kesilishga yaqin.");
  if (st.peakDb < -20) W('warn', 'Ovoz past. Mikrofonga yaqinroq keling yoki kuchaytirishni oshiring.');
  if (st.snr < 30) W('bad', `Shovqin ko'p (SNR ${st.snr} dB). Jimroq joy toping.`);
  else if (st.snr < 40) W('warn', `SNR ${st.snr} dB: ishlaydi, lekin 40 dB dan yuqori bo'lsa a'lo.`);
  if (st.lead < 0.1) W('warn', "Boshi kesilgan bo'lishi mumkin: yozishni bosgach, bir lahza kutib gapiring.");
  if (st.trail < 0.1) W('warn', "Oxiri kesilgan bo'lishi mumkin: gapni tugatgach, bir lahza kutib to'xtating.");
  if (speech < 0.8) W('bad', 'Juda qisqa.');
  if (st.rate && (st.rate < 6 || st.rate > 22)) W('warn', `Tezlik g'alati (${st.rate} harf/s): aytilgan gap matnga mos kelmasligi mumkin.`);
  return st;
}

function showStats(st) {
  const box = $('#stats');
  const grade = { good: "A'lo", warn: 'Tekshiring', bad: 'Qayta yozing' }[st.grade];
  const num = (label, value) => `<span class="num">${label} <b>${value}</b></span>`;
  box.innerHTML = `<span class="grade ${st.grade}">${grade}</span>`
    + num('Davomiyligi', `${st.speech.toFixed(1)} s`) + num("Cho'qqi", `${st.peakDb} dB`)
    + num('SNR', `${st.snr} dB`) + num('Shovqin', `${st.noiseDb} dB`) + num('Tezlik', `${st.rate} harf/s`);
  if (st.warnings.length) {
    const ul = document.createElement('ul');
    ul.replaceChildren(...st.warnings.map((w) => Object.assign(document.createElement('li'), { className: w.level, textContent: w.msg })));
    box.append(ul);
  }
}

function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function clearWave() {
  const c = $('#wave');
  c.getContext('2d').clearRect(0, 0, c.width, c.height);
}

function drawWave(x, st) {
  const c = $('#wave');
  const dpr = window.devicePixelRatio || 1;
  c.width = Math.round(c.clientWidth * dpr); c.height = Math.round(96 * dpr);
  const g = c.getContext('2d');
  const W = c.width, H = c.height, mid = H / 2, step = x.length / W;
  g.clearRect(0, 0, W, H);
  for (let px = 0; px < W; px++) {
    const a = Math.floor(px * step), b = Math.min(x.length, Math.floor((px + 1) * step) + 1);
    let lo = 0, hi = 0;
    for (let i = a; i < b; i++) { if (x[i] < lo) lo = x[i]; if (x[i] > hi) hi = x[i]; }
    const kept = a >= st.trimStart && a < st.trimEnd;
    g.fillStyle = kept ? css('--wave') : css('--wave-dim');
    g.fillRect(px, mid - hi * mid, 1, Math.max(1, (hi - lo) * mid));
  }
}

// ------------------------------------------------------------ playback + save

function stopPlayback() {
  if (S.player) { try { S.player.stop ? S.player.stop() : S.player.pause(); } catch { /* already stopped */ } S.player = null; }
}

async function play() {
  if (S.recording) return;
  stopPlayback();
  if (S.take) {
    const ctx = S.ctx || (S.playCtx ||= new AudioContext());
    const buf = ctx.createBuffer(1, S.take.samples.length, S.take.sr);
    buf.copyToChannel(S.take.samples, 0);
    const node = ctx.createBufferSource();
    node.buffer = buf; node.connect(ctx.destination); node.start();
    S.player = node;
  } else if (current()?.recorded) {
    const audio = new Audio(`/api/audio?id=${encodeURIComponent(current().id)}&t=${Date.now()}`);
    S.player = audio;
    await audio.play().catch((e) => toast(`Ijro bo'lmadi: ${e.message}`, 'bad'));
  }
}

function encodeWav(x, sr) {
  const buf = new ArrayBuffer(44 + x.length * 2);
  const v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); v.setUint32(4, 36 + x.length * 2, true); str(8, 'WAVE'); str(12, 'fmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, sr, true);
  v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); str(36, 'data');
  v.setUint32(40, x.length * 2, true);
  for (let i = 0, o = 44; i < x.length; i++, o += 2) {
    const s = Math.max(-1, Math.min(1, x[i]));
    v.setInt16(o, Math.round(s < 0 ? s * 0x8000 : s * 0x7fff), true);
  }
  return new Blob([buf], { type: 'audio/wav' });
}

async function save() {
  if (!S.take || S.recording) return;
  const it = current();
  if (S.take.stats.grade === 'bad' && !confirm('Bu yozuvda jiddiy muammo bor. Baribir saqlansinmi?')) return;
  const { trimStart, trimEnd, warnings, ...rest } = S.take.stats;
  const stats = { ...rest, warnings: warnings.map((w) => w.msg) };
  try {
    const r = await api(`/api/save?id=${encodeURIComponent(it.id)}`, {
      method: 'POST', body: encodeWav(S.take.samples, S.take.sr),
      headers: { 'Content-Type': 'audio/wav', 'X-Text': encodeURIComponent(it.text), 'X-Stats': encodeURIComponent(JSON.stringify(stats)) },
    });
    Object.assign(it, { recorded: true, duration: r.duration, spoken: r.spoken });
    S.totalSeconds = r.totalSeconds;
    S.take = null;
    toast(`Saqlandi: ${it.id} · ${r.duration.toFixed(1)} s`, 'good');
    renderListItem(S.idx);
    const next = nextIndex();
    if (next === S.idx) render(); else go(next);
  } catch (e) { toast(`Saqlanmadi: ${e.message}`, 'bad'); }
}

// ------------------------------------------------------------ editing the sentence

function startEdit() {
  if (S.recording) return;
  $('#editText').value = current().text;
  $('#editBox').hidden = false;
  $('#editText').focus();
}

function closeEdit() { $('#editBox').hidden = true; }

async function applyEdit() {
  const it = current();
  const text = $('#editText').value.trim();
  if (!text) return;
  try {
    const r = await api('/api/normalize', { method: 'POST', body: JSON.stringify({ text }), headers: { 'Content-Type': 'application/json' } });
    Object.assign(it, { text, spoken: r.spoken, issues: r.issues, edited: true });
    closeEdit(); renderListItem(S.idx); render();
  } catch (e) { toast(e.message, 'bad'); }
}

// ------------------------------------------------------------ wiring

function on(id, fn) {
  $(id).addEventListener('click', (e) => { e.currentTarget.blur(); fn(); });
}
on('#micBtn', () => enableMic());
on('#noiseBtn', measureNoise);
on('#recBtn', () => (S.recording ? stopRec() : startRec()));
on('#playBtn', play);
on('#saveBtn', save);
on('#prevBtn', () => go(S.idx - 1));
on('#nextBtn', () => go(S.idx + 1));
on('#editSave', applyEdit);
on('#editCancel', closeEdit);
$('#deviceSelect').addEventListener('change', (e) => enableMic(e.target.value));
$('#onlyTodo').addEventListener('change', (e) => $('.list').classList.toggle('todo', e.target.checked));
$('#editText').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); applyEdit(); }
  if (e.key === 'Escape') closeEdit();
});

document.addEventListener('keydown', (e) => {
  if (e.target.closest('textarea, input, select') || e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === ' ') { e.preventDefault(); S.recording ? stopRec() : startRec(); }
  else if (k === 'enter') { e.preventDefault(); save(); }
  else if (k === 'r') { e.preventDefault(); if (!S.recording) { S.take = null; startRec(); } }
  else if (k === 'p') { e.preventDefault(); play(); }
  else if (k === 'e') { e.preventDefault(); startEdit(); }
  else if (k === 'arrowright') go(S.idx + 1);
  else if (k === 'arrowleft') go(S.idx - 1);
});
window.addEventListener('beforeunload', (e) => { if (S.take || S.recording) e.preventDefault(); });

window.studio = { S, analyze, encodeWav, go };  // handy in the browser console
load().catch((e) => toast(`Server bilan aloqa yo'q: ${e.message}`, 'bad'));

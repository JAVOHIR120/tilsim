/**
 * Tilsim live speech in the browser: WebSocket in, gap-free playback through an AudioWorklet, instant cancel.
 * No dependencies. Works in current Chrome, Edge, Firefox and Safari.
 *
 *   import { TilsimStream } from './tilsim-stream.js';
 *   const tts = new TilsimStream({
 *     url: 'wss://javohir120--tilsim-api.us-east.modal.run/v1/stream',
 *     getToken: async () => (await fetch('/api/tilsim-token')).json().then(r => r.token), // your backend
 *     onMark: (text) => showSubtitle(text),
 *     onEnd: (reason) => console.log('turn over:', reason),
 *   });
 *   await tts.connect();          // call from a click/tap: browsers only start audio after a user gesture
 *   tts.say('Endi Part 2 ga o‘tamiz. ');   // pieces from your LLM, as they arrive
 *   tts.say('Describe a place you like to visit.');
 *   tts.flush();                  // end of the examiner's turn
 *   tts.cancel();                 // the student starts talking: silence now
 *
 * Keys: never put your API key in a page. Your backend calls POST /v1/stream/token with the key and hands the
 * browser a short-lived token (getToken). `key` works too, for tests on your own machine.
 *
 * Server -> client: binary PCM16 little-endian, mono, 24 kHz; JSON {"type": "ready" | "start" | "mark" | "end" | "error"}.
 */

const WORKLET = `
class TilsimPlayer extends AudioWorkletProcessor {
  constructor() {
    super();
    this.chunks = []; this.offset = 0; this.buffered = 0;
    this.playing = false; this.prebuffer = 0; this.underruns = 0; this.idleSent = true;
    this.port.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'pcm') { this.chunks.push(m.data); this.buffered += m.data.length; this.idleSent = false; }
      else if (m.type === 'clear') { this.chunks = []; this.offset = 0; this.buffered = 0; this.playing = false; }
      else if (m.type === 'prebuffer') { this.prebuffer = m.samples; }
    };
  }
  process(_inputs, outputs) {
    const out = outputs[0][0];
    if (!this.playing && this.buffered >= Math.max(this.prebuffer, 1)) this.playing = true;
    let i = 0;
    while (this.playing && i < out.length && this.chunks.length) {
      const c = this.chunks[0];
      const n = Math.min(out.length - i, c.length - this.offset);
      out.set(c.subarray(this.offset, this.offset + n), i);
      i += n; this.offset += n; this.buffered -= n;
      if (this.offset >= c.length) { this.chunks.shift(); this.offset = 0; }
    }
    if (i < out.length) {
      out.fill(0, i);
      if (this.playing && !this.chunks.length) {   // ran dry: wait for the prebuffer again
        this.playing = false;
        if (!this.idleSent) { this.port.postMessage({ type: 'idle' }); this.idleSent = true; }
      }
    }
    return true;
  }
}
registerProcessor('tilsim-player', TilsimPlayer);
`;

export class TilsimStream {
  /**
   * @param {object} o
   * @param {string} o.url          wss://.../v1/stream
   * @param {string} [o.key]        API key (tests only; prefer a token)
   * @param {string} [o.token]      token from POST /v1/stream/token
   * @param {() => Promise<string>} [o.getToken]  fetches a fresh token for each connect()
   * @param {number} [o.prebufferMs=100]  audio kept before playback starts (absorbs network jitter and the ~0.1 s
   *   worst gap measured on the laptop GPU)
   * @param {(text: string) => void} [o.onMark]   the sentence being spoken now
   * @param {() => void} [o.onStart]  first audio of a turn is on its way
   * @param {(reason: string) => void} [o.onEnd]  "done" | "cancelled"
   * @param {() => void} [o.onIdle]   the speaker went quiet (buffer played out)
   * @param {(err: {code: string, message?: string}) => void} [o.onError]
   */
  constructor(o) {
    this.o = { prebufferMs: 100, ...o };
    this.ws = null; this.ctx = null; this.node = null; this.sr = 24000;
    this.turnLive = true;   // false between cancel() and the server's "end": late audio is dropped
  }

  async connect() {
    if (!this.ctx) {
      this.ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: this.sr, latencyHint: 'interactive' });
      const url = URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' }));
      await this.ctx.audioWorklet.addModule(url);
      URL.revokeObjectURL(url);
      this.node = new AudioWorkletNode(this.ctx, 'tilsim-player', { outputChannelCount: [1] });
      this.node.connect(this.ctx.destination);
      this.node.port.onmessage = (e) => { if (e.data.type === 'idle') this.o.onIdle?.(); };
      this.node.port.postMessage({ type: 'prebuffer', samples: Math.round(this.ctx.sampleRate * this.o.prebufferMs / 1000) });
    }
    if (this.ctx.state === 'suspended') await this.ctx.resume();
    const token = this.o.getToken ? await this.o.getToken() : this.o.token;
    const q = token ? `token=${encodeURIComponent(token)}` : this.o.key ? `key=${encodeURIComponent(this.o.key)}` : '';
    const ws = new WebSocket(this.o.url + (q ? (this.o.url.includes('?') ? '&' : '?') + q : ''));
    ws.binaryType = 'arraybuffer';
    this.ws = ws;
    await new Promise((resolve, reject) => {
      ws.onmessage = (e) => this._message(e, resolve);
      ws.onerror = () => reject(new Error('Tilsim: WebSocket error'));
      ws.onclose = (e) => { if (this.ws === ws) this.ws = null; reject(new Error(`Tilsim: closed (${e.code})`)); };
    });
    ws.onclose = () => { if (this.ws === ws) this.ws = null; };
    return this;
  }

  _message(e, ready) {
    if (typeof e.data !== 'string') {           // audio
      if (!this.turnLive) return;
      const pcm = new Int16Array(e.data);
      let f = new Float32Array(pcm.length);
      for (let i = 0; i < pcm.length; i++) f[i] = pcm[i] / 32768;
      if (this.ctx.sampleRate !== this.sr) f = resample(f, this.sr, this.ctx.sampleRate);
      this.node.port.postMessage({ type: 'pcm', data: f }, [f.buffer]);
      return;
    }
    const m = JSON.parse(e.data);
    if (m.type === 'ready') ready?.();
    else if (m.type === 'start') { this.turnLive = true; this.o.onStart?.(); }
    else if (m.type === 'mark') this.o.onMark?.(m.text);
    else if (m.type === 'end') { this.turnLive = true; this.o.onEnd?.(m.reason); }
    else if (m.type === 'error') this.o.onError?.(m);
  }

  _send(obj) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) throw new Error('Tilsim: not connected');
    this.ws.send(JSON.stringify(obj));
  }

  /** Text to speak; pieces may be any size. flush=true ends the turn. */
  say(text, { flush = false } = {}) { this.turnLive = true; this._send({ type: 'text', text, flush }); }

  /** End of the turn: the rest of the text is spoken, then onEnd("done"). */
  flush() { this._send({ type: 'flush' }); }

  /** Stop at once: local audio is dropped this instant, the server stops making more. */
  cancel() {
    this.turnLive = false;
    this.node?.port.postMessage({ type: 'clear' });
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify({ type: 'cancel' }));
  }

  close() { this.cancel(); this.ws?.close(); this.ws = null; }
}

/** POST /v1/warm: call when the exam screen opens, so the first answer does not wait for a cold start. */
export async function warm(httpBase, { key, token } = {}) {
  const q = token ? `?token=${encodeURIComponent(token)}` : '';
  const r = await fetch(httpBase.replace(/\/$/, '') + '/v1/warm' + q, {
    method: 'POST', headers: { 'Content-Type': 'application/json', ...(key ? { 'X-API-Key': key } : {}) }, body: '{}',
  });
  return r.json();
}

function resample(x, from, to) {   // linear: only used when the browser refused a 24 kHz AudioContext
  const n = Math.round(x.length * to / from);
  const y = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const p = i * from / to, k = Math.floor(p), a = p - k;
    y[i] = (x[k] ?? 0) * (1 - a) + (x[k + 1] ?? x[k] ?? 0) * a;
  }
  return y;
}

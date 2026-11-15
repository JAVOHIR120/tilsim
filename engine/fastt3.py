"""CUDA-graph decoding for T3 (Chatterbox's 520M Llama): one graph replay per speech token.

Why: measured 2026-11-15 on the RTX 3050 laptop, T3 takes ~33 ms per token in fp32 AND in bf16. Precision did
not matter, so the time goes into launching ~600 small kernels per step from Python, not into the math.
A CUDA graph records the whole decode step once (static shapes, static KV cache) and replays it with one call.

    dec = GraphDecoder(t3, max_len=1400)        # weights already on the GPU in bf16/fp16
    logits = dec.prefill(inputs_embeds)         # HF forward over cond+text+prefix, KV copied into static buffers
    logits = dec.step(next_embedding)           # (2, vocab) for the CFG pair, one graph replay

The step is plain PyTorch over the model's own modules (RMSNorm, projections, MLP, rotary table from the
model's rotary_emb), so it computes what transformers' LlamaModel computes. check_parity() compares the two.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _cache_kv(past, i: int):
    if hasattr(past, "layers"):          # transformers >= 4.56 / 5.x
        return past.layers[i].keys, past.layers[i].values
    return past.key_cache[i], past.value_cache[i]


class GraphDecoder:
    def __init__(self, t3, max_len: int = 1400, batch: int = 2):
        self.t3 = t3
        self.llama = llama = t3.tfmr
        cfg = llama.config
        p = next(llama.parameters())
        self.device, self.dtype = p.device, p.dtype
        self.B, self.max_len = batch, max_len
        self.H, self.KV = cfg.num_attention_heads, cfg.num_key_value_heads
        self.Dh = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        hidden = cfg.hidden_size
        shape = (batch, self.KV, max_len, self.Dh)
        self.k = [torch.zeros(shape, dtype=self.dtype, device=self.device) for _ in llama.layers]
        self.v = [torch.zeros(shape, dtype=self.dtype, device=self.device) for _ in llama.layers]
        with torch.inference_mode():
            pos_ids = torch.arange(max_len, device=self.device)[None]
            cos, sin = llama.rotary_emb(torch.zeros(1, 1, hidden, dtype=self.dtype, device=self.device), pos_ids)
        self.cos, self.sin = cos[0].to(self.dtype), sin[0].to(self.dtype)
        self.arange = torch.arange(max_len, device=self.device)
        self.x_in = torch.zeros(batch, 1, hidden, dtype=self.dtype, device=self.device)
        self.pos = torch.zeros(1, dtype=torch.long, device=self.device)
        self.logits = None
        self.graph = None
        self.n = 0  # next position to write

    # ------------------------------------------------------------ the static step
    def _step(self) -> None:
        B, H, KV, Dh = self.B, self.H, self.KV, self.Dh
        x = self.x_in
        cos = self.cos.index_select(0, self.pos).view(1, 1, 1, Dh)
        sin = self.sin.index_select(0, self.pos).view(1, 1, 1, Dh)
        mask = (self.arange <= self.pos).view(1, 1, 1, -1)
        for i, layer in enumerate(self.llama.layers):
            attn = layer.self_attn
            h = layer.input_layernorm(x)
            q = attn.q_proj(h).view(B, 1, H, Dh).transpose(1, 2)
            k = attn.k_proj(h).view(B, 1, KV, Dh).transpose(1, 2)
            v = attn.v_proj(h).view(B, 1, KV, Dh).transpose(1, 2)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
            self.k[i].index_copy_(2, self.pos, k)
            self.v[i].index_copy_(2, self.pos, v)
            kk, vv = self.k[i], self.v[i]
            if KV != H:  # grouped-query attention (not used by Llama_520M, kept for safety)
                kk = kk.repeat_interleave(H // KV, dim=1)
                vv = vv.repeat_interleave(H // KV, dim=1)
            o = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask, scale=attn.scaling)
            x = x + attn.o_proj(o.transpose(1, 2).reshape(B, 1, H * Dh))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        self.logits = self.t3.speech_head(self.llama.norm(x))[:, -1, :]

    @torch.inference_mode()
    def capture(self) -> None:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step()

    # ------------------------------------------------------------ public
    @torch.inference_mode()
    def prefill(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """HF forward over the whole prompt; its KV goes into the static buffers. Returns (B, vocab) logits."""
        out = self.llama(inputs_embeds=inputs_embeds, use_cache=True, return_dict=True)
        past = out.past_key_values
        S = inputs_embeds.shape[1]
        if S >= self.max_len:
            raise ValueError(f"prompt of {S} positions does not fit max_len {self.max_len}")
        for i in range(len(self.k)):
            kk, vv = _cache_kv(past, i)
            self.k[i][:, :, :S].copy_(kk)
            self.v[i][:, :, :S].copy_(vv)
        self.n = S
        return self.t3.speech_head(out.last_hidden_state[:, -1:, :])[:, -1, :]

    @torch.inference_mode()
    def step(self, emb: torch.Tensor) -> torch.Tensor:
        """emb: (B, 1, hidden) input embedding of the newest token. Returns (B, vocab); valid until the next step."""
        if self.n >= self.max_len:
            raise ValueError("speech longer than max_len")
        self.x_in.copy_(emb)
        self.pos.fill_(self.n)
        if self.graph is None:
            self._step()
        else:
            self.graph.replay()
        self.n += 1
        return self.logits

    def room(self) -> int:
        return self.max_len - self.n


@torch.inference_mode()
def check_parity(t3, embeds: torch.Tensor, steps: int = 24, max_len: int = 1400) -> dict:
    """Feed the same tokens through transformers and through the graph; compare logits."""
    from chatterbox.models.t3.inference.t3_hf_backend import T3HuggingfaceBackend
    model = T3HuggingfaceBackend(config=t3.cfg, llama=t3.tfmr, speech_enc=t3.speech_emb, speech_head=t3.speech_head,
                                 alignment_stream_analyzer=None)
    out = model(inputs_embeds=embeds, past_key_values=None, use_cache=True, output_attentions=False,
                output_hidden_states=True, return_dict=True)
    past, ref = out.past_key_values, [out.logits[:, -1, :].float()]
    dec = GraphDecoder(t3, max_len=max_len)
    dec.capture()
    got = [dec.prefill(embeds).float()]
    n0 = embeds.shape[1]
    torch.manual_seed(0)
    for i in range(steps):
        tok = torch.randint(0, 6000, (1, 1), device=embeds.device)
        emb = t3.speech_emb(tok) + t3.speech_pos_emb.get_fixed_embedding(200 + i)
        emb = torch.cat([emb, emb])
        o = model(inputs_embeds=emb, past_key_values=past, output_attentions=False, output_hidden_states=True,
                  return_dict=True)
        past = o.past_key_values
        ref.append(o.logits[:, -1, :].float())
        got.append(dec.step(emb).float().clone())
    diffs = [float((a - b).abs().max()) for a, b in zip(ref, got)]
    agree = sum(int(a[0].argmax() == b[0].argmax()) for a, b in zip(ref, got))
    scale = float(torch.stack([r.abs().max() for r in ref]).max())
    return {"max_abs_diff": max(diffs), "logit_scale": scale, "argmax_agree": f"{agree}/{len(ref)}", "positions": n0}

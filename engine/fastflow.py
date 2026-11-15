"""CUDA graphs for S3Gen's flow estimator: the 56-transformer-block UNet that runs once per Euler step.

Why: measured 2026-11-15 on the RTX 3050 laptop, one flow call cost ~300-500 ms whether it had 22 or 274 new
frames, i.e. ~50 ms per Euler step of kernel-launch overhead. A CUDA graph per input-length bucket replays the
whole UNet with one call.

Lengths vary, graphs need fixed shapes. Inputs are padded at the end to the next bucket with mask = 0.
The UNet's convolutions are causal (they look only left), its attention ignores masked keys, and every
down/up-sampling multiplies by the mask first. So the valid frames come out as without padding;
check_parity() measures it.

    graphs = FlowGraphs(s3gen, buckets=(256, 320, 384, 448, 512)); graphs.capture()
    # from then on s3gen.flow.inference(...) uses the graphs automatically for lengths that fit a bucket
"""
from __future__ import annotations

import threading

import torch


def _graph_safe_masks(xs, masks, use_dynamic_chunk, use_dynamic_left_chunk, decoding_chunk_size,
                      static_chunk_size, num_decoding_left_chunks, enable_full_context=True):
    """add_optional_chunk_mask for the only case S3Gen uses (no chunking): the masks unchanged.
    The original also checks for rows with no valid key via .item(), a CPU sync that CUDA graphs forbid;
    with a (B, 1, T) padding mask such a row would need an all-padding input, which never happens."""
    if use_dynamic_chunk or static_chunk_size > 0:
        raise NotImplementedError("chunked flow masks are not graph-safe here")
    return masks


class FlowGraphs:
    def __init__(self, s3gen, buckets=(256, 320, 384, 448, 512, 640)):
        from chatterbox.models.s3gen import decoder as decoder_module
        decoder_module.add_optional_chunk_mask = _graph_safe_masks
        self.est = s3gen.flow.decoder.estimator
        p = next(self.est.parameters())
        self.device, self.dtype = p.device, p.dtype
        self.buckets = sorted(buckets)
        self.bufs: dict[int, dict] = {}
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.pool = None
        self.lock = threading.Lock()   # one set of static buffers: one flow at a time
        self.eager = self.est.forward
        self.est.forward = self.forward  # solve_euler calls estimator.forward(...) explicitly
        self.hits = self.misses = 0

    def _buffers(self, T: int) -> dict:
        z = lambda *s: torch.zeros(*s, device=self.device, dtype=self.dtype)  # noqa: E731
        return {"x": z(2, 80, T), "mask": z(2, 1, T), "mu": z(2, 80, T), "t": z(2), "spks": z(2, 80),
                "cond": z(2, 80, T), "out": None}

    def _run(self, b: dict) -> None:
        b["out"] = self.eager(x=b["x"], mask=b["mask"], mu=b["mu"], t=b["t"], spks=b["spks"], cond=b["cond"])

    @torch.inference_mode()
    def capture(self) -> None:
        for T in self.buckets:
            b = self._buffers(T)
            b["mask"][:, :, : T // 2] = 1  # realistic warm-up input
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._run(b)
            torch.cuda.current_stream().wait_stream(stream)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                self._run(b)
            self.pool = g.pool()  # all buckets share one memory pool (they never run at the same time)
            self.bufs[T], self.graphs[T] = b, g

    def forward(self, x, mask, mu, t, spks=None, cond=None, r=None):
        T = x.shape[2]
        bucket = next((s for s in self.buckets if s >= T), None)
        if bucket is None or r is not None or spks is None or cond is None or x.shape[0] != 2:
            self.misses += 1
            return self.eager(x=x, mask=mask, mu=mu, t=t, spks=spks, cond=cond, r=r)
        self.hits += 1
        with self.lock:
            b = self.bufs[bucket]
            for name, val in (("x", x), ("mu", mu), ("cond", cond), ("mask", mask)):
                b[name][:, :, :T].copy_(val)
                b[name][:, :, T:].zero_()
            b["t"].copy_(t)
            b["spks"].copy_(spks)
            self.graphs[bucket].replay()
            return b["out"][:, :, :T].clone()


@torch.inference_mode()
def check_parity(s3gen, ref_dict: dict, lengths=(30, 80, 140), steps: int = 6) -> dict:
    """Same tokens and noise through the eager flow and through the graphs; max difference of the mel."""
    flow = s3gen.flow
    est = flow.decoder.estimator
    graphs_forward = est.forward
    out = {}
    for n in lengths:
        g = torch.Generator(device=s3gen.device).manual_seed(n)
        toks = torch.randint(0, 6000, (1, n), generator=g, device=s3gen.device)
        # noise for prompt + new frames: the decoder's own randn_like would differ between the two runs
        noise = torch.randn(1, 80, ref_dict["prompt_feat"].shape[1] + 2 * n, generator=g, device=s3gen.device)
        kw = dict(token=toks, token_len=torch.tensor([n], device=s3gen.device), prompt_token=ref_dict["prompt_token"],
                  prompt_token_len=ref_dict["prompt_token_len"], prompt_feat=ref_dict["prompt_feat"],
                  prompt_feat_len=ref_dict.get("prompt_feat_len"), embedding=ref_dict["embedding"], finalize=True,
                  n_timesteps=steps, noised_mels=noise)
        est.forward = est.__class__.forward.__get__(est)   # eager
        a, _ = flow.inference(**kw)
        est.forward = graphs_forward                        # graphs
        b, _ = flow.inference(**kw)
        out[n] = {"max_abs_diff": round(float((a - b).abs().max()), 6), "mel_scale": round(float(a.abs().max()), 3),
                  "frames": int(a.shape[2])}
    return out

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim,
        eps=1e-6,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.ones(dim)
        )

        self.eps = eps

    def forward(self, x):
        variance = (
            x.float()
            .pow(2)
            .mean(
                dim=-1,
                keepdim=True,
            )
        )

        x = x * torch.rsqrt(
            variance + self.eps
        )

        return (
            self.weight * x
        )


def rotate_half(x):
    half = x.shape[-1] // 2

    return torch.cat(
        [
            -x[..., half:],
            x[..., :half],
        ],
        dim=-1,
    )


def apply_rope(q, k):
    seq_len = q.shape[-2]
    head_dim = q.shape[-1]
    device = q.device

    inv_freq = 1.0 / (
        10000 ** (
            torch.arange(
                0,
                head_dim,
                2,
                device=device,
                dtype=torch.float32,
            )
            / head_dim
        )
    )

    positions = torch.arange(
        seq_len,
        device=device,
        dtype=torch.float32,
    )

    freqs = torch.outer(
        positions,
        inv_freq,
    )

    emb = torch.cat(
        [freqs, freqs],
        dim=-1,
    )

    cos = emb.cos()[
        None,
        None,
        :,
        :
    ]

    sin = emb.sin()[
        None,
        None,
        :,
        :
    ]

    return (
        q * cos + rotate_half(q) * sin,
        k * cos + rotate_half(k) * sin,
    )


class SwiGLU(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        dropout,
    ):
        super().__init__()

        self.gate = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )

        self.up = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )

        self.down = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(self, x):
        x = (
            F.silu(
                self.gate(x)
            )
            * self.up(x)
        )

        return self.dropout(
            self.down(x)
        )


class GQAAttention(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        n_kv_heads,
        dropout,
    ):
        super().__init__()

        if n_heads % n_kv_heads:
            raise ValueError(
                "n_heads must be divisible "
                "by n_kv_heads"
            )

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads

        self.head_dim = (
            dim // n_heads
        )

        self.repeat = (
            n_heads // n_kv_heads
        )

        self.q_proj = nn.Linear(
            dim,
            n_heads * self.head_dim,
            bias=False,
        )

        self.k_proj = nn.Linear(
            dim,
            n_kv_heads * self.head_dim,
            bias=False,
        )

        self.v_proj = nn.Linear(
            dim,
            n_kv_heads * self.head_dim,
            bias=False,
        )

        self.o_proj = nn.Linear(
            dim,
            dim,
            bias=False,
        )

        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape

        q = self.q_proj(x).view(
            B,
            T,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        k = self.k_proj(x).view(
            B,
            T,
            self.n_kv_heads,
            self.head_dim,
        ).transpose(1, 2)

        v = self.v_proj(x).view(
            B,
            T,
            self.n_kv_heads,
            self.head_dim,
        ).transpose(1, 2)

        q, k = apply_rope(q, k)

        k = k.repeat_interleave(
            self.repeat,
            dim=1,
        )

        v = v.repeat_interleave(
            self.repeat,
            dim=1,
        )

        # PyTorch SDPA uses an optimized attention
        # implementation when available.
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=(
                self.dropout
                if self.training
                else 0.0
            ),
            is_causal=True,
        )

        output = (
            output
            .transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        return self.o_proj(output)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        n_kv_heads,
        ffn_dim,
        dropout,
    ):
        super().__init__()

        self.norm1 = RMSNorm(dim)

        self.attention = GQAAttention(
            dim,
            n_heads,
            n_kv_heads,
            dropout,
        )

        self.norm2 = RMSNorm(dim)

        self.ffn = SwiGLU(
            dim,
            ffn_dim,
            dropout,
        )

    def forward(self, x):
        x = (
            x
            + self.attention(
                self.norm1(x)
            )
        )

        x = (
            x
            + self.ffn(
                self.norm2(x)
            )
        )

        return x


class ModernGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        dim,
        n_layers,
        n_heads,
        n_kv_heads,
        ffn_dim,
        dropout=0.0,
    ):
        super().__init__()

        self.block_size = block_size

        self.embedding = nn.Embedding(
            vocab_size,
            dim,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    n_heads,
                    n_kv_heads,
                    ffn_dim,
                    dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm = RMSNorm(dim)

        self.lm_head = nn.Linear(
            dim,
            vocab_size,
            bias=False,
        )

        # Weight tying:
        self.lm_head.weight = (
            self.embedding.weight
        )

        self.apply(
            self._init_weights
        )

    def _init_weights(self, module):
        if isinstance(
            module,
            nn.Linear,
        ):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

        elif isinstance(
            module,
            nn.Embedding,
        ):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(
        self,
        idx,
        targets=None,
    ):
        B, T = idx.shape

        if T > self.block_size:
            raise ValueError(
                "Sequence exceeds context length."
            )

        x = self.embedding(idx)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=0.8,
        top_k=40,
    ):
        self.eval()

        for _ in range(
            max_new_tokens
        ):
            idx_cond = idx[
                :,
                -self.block_size:
            ]

            logits, _ = self(
                idx_cond
            )

            logits = logits[
                :,
                -1,
                :
            ]

            logits /= max(
                temperature,
                1e-5,
            )

            if top_k:
                k = min(
                    top_k,
                    logits.size(-1),
                )

                values, _ = torch.topk(
                    logits,
                    k,
                )

                threshold = values[
                    :,
                    [-1],
                ]

                logits = torch.where(
                    logits < threshold,
                    torch.full_like(
                        logits,
                        float("-inf"),
                    ),
                    logits,
                )

            probs = F.softmax(
                logits,
                dim=-1,
            )

            next_token = torch.multinomial(
                probs,
                1,
            )

            idx = torch.cat(
                [idx, next_token],
                dim=1,
            )

        return idx

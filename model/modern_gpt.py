"""Core Transformer architecture used by Small-GPT.

This module implements the decoder-only Transformer architecture used by
Small-GPT.

The model combines several modern Transformer components:

    - RMSNorm
    - Rotary Positional Embeddings (RoPE)
    - Grouped-Query Attention (GQA)
    - SwiGLU feed-forward networks
    - Pre-normalization
    - Residual connections
    - Weight tying between token embeddings and the language-model head
    - Causal self-attention
    - Autoregressive text generation

The implementation is intentionally kept compact and educational while
using PyTorch's optimized scaled dot-product attention where available.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    RMSNorm normalizes the magnitude of hidden states without subtracting
    their mean. Compared with traditional LayerNorm, it removes the mean
    and variance centering operation and uses only the root mean square
    magnitude of the activations.

    Args:
        dim: Size of the final dimension being normalized.
        eps: Small value added for numerical stability.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        # Learnable scaling parameter applied after normalization.
        self.weight = nn.Parameter(
            torch.ones(dim)
        )

        # Prevent division by zero when computing the inverse RMS.
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize the hidden states using their root mean square.

        Args:
            x: Input tensor whose final dimension has size ``dim``.

        Returns:
            RMS-normalized tensor with the same shape as ``x``.
        """
        # Compute mean squared activation magnitude.
        #
        # Float32 is used for this calculation to improve numerical
        # stability when the model itself is running in lower precision.
        variance = (
            x.float()
            .pow(2)
            .mean(
                dim=-1,
                keepdim=True,
            )
        )

        # Scale each hidden state by the inverse root mean square.
        x = x * torch.rsqrt(
            variance + self.eps
        )

        # Apply the learned RMSNorm scaling parameter.
        return self.weight * x


def rotate_half(
    x: torch.Tensor,
) -> torch.Tensor:
    """Rotate the final dimension by splitting it into two halves.

    This operation is a helper used by Rotary Positional Embeddings (RoPE).
    The first and second halves are exchanged with a sign change on the
    second half.

    Args:
        x: Tensor whose final dimension contains the attention head features.

    Returns:
        Tensor with the rotated feature representation.
    """
    half = x.shape[-1] // 2

    return torch.cat(
        [
            -x[..., half:],
            x[..., :half],
        ],
        dim=-1,
    )


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Positional Embeddings to queries and keys.

    RoPE encodes token position by rotating query and key representations
    according to position-dependent sinusoidal frequencies.

    Applying RoPE directly to queries and keys allows attention scores to
    incorporate relative positional information without requiring a
    separate learned positional embedding table.

    Args:
        q: Query tensor with shape
            ``[batch, heads, sequence, head_dim]``.
        k: Key tensor with shape
            ``[batch, heads, sequence, head_dim]``.

    Returns:
        A tuple containing the rotated query and key tensors.
    """
    seq_len = q.shape[-2]
    head_dim = q.shape[-1]
    device = q.device

    # Calculate the inverse frequencies used by the rotary embedding.
    #
    # Frequencies are distributed across the dimensions of each attention
    # head, allowing different dimensions to rotate at different rates.
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

    # Generate a position index for every token in the sequence.
    positions = torch.arange(
        seq_len,
        device=device,
        dtype=torch.float32,
    )

    # Combine token positions and frequencies to obtain rotation angles.
    freqs = torch.outer(
        positions,
        inv_freq,
    )

    # Duplicate the frequency representation so that it matches the
    # complete attention-head dimension.
    emb = torch.cat(
        [freqs, freqs],
        dim=-1,
    )

    # Add batch and head dimensions for broadcasting over q and k.
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

    # Apply the rotary transformation to both queries and keys.
    return (
        q * cos + rotate_half(q) * sin,
        k * cos + rotate_half(k) * sin,
    )


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network used inside each Transformer block.

    SwiGLU combines a SiLU-activated gating projection with a second
    projection and then projects the result back to the model dimension.

    This provides a gated alternative to the traditional Transformer
    feed-forward network.

    Args:
        dim: Transformer hidden dimension.
        hidden_dim: Intermediate feed-forward dimension.
        dropout: Dropout probability applied to the FFN output.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        # Gating projection. SiLU is applied to this branch before it is
        # multiplied with the up-projection.
        self.gate = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )

        # Second projection providing the values controlled by the gate.
        self.up = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )

        # Project the expanded representation back to the model dimension.
        self.down = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the SwiGLU feed-forward transformation.

        Args:
            x: Transformer hidden states.

        Returns:
            Transformed hidden states with the original model dimension.
        """
        # SwiGLU:
        #
        #   SiLU(gate(x)) * up(x)
        #
        # The multiplicative gate allows the network to control which
        # intermediate features are passed through the FFN.
        x = (
            F.silu(
                self.gate(x)
            )
            * self.up(x)
        )

        # Project back to the model dimension and apply dropout.
        return self.dropout(
            self.down(x)
        )


class GQAAttention(nn.Module):
    """Grouped-Query Self-Attention.

    GQA uses more query heads than key/value heads. Multiple query heads
    share the same key/value representations, reducing the amount of
    key/value memory required compared with standard multi-head attention.

    Args:
        dim: Transformer hidden dimension.
        n_heads: Number of query attention heads.
        n_kv_heads: Number of key/value heads.
        dropout: Attention dropout probability.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        # Each key/value head is shared by an equal number of query heads.
        if n_heads % n_kv_heads:
            raise ValueError(
                "n_heads must be divisible by n_kv_heads"
            )

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads

        # Determine the dimensionality of each attention head.
        self.head_dim = dim // n_heads

        # Number of query heads assigned to each key/value head.
        self.repeat = n_heads // n_kv_heads

        # Query projection produces one representation for every query head.
        self.q_proj = nn.Linear(
            dim,
            n_heads * self.head_dim,
            bias=False,
        )

        # Key and value projections use fewer heads because of GQA.
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

        # Final projection combines all attention-head outputs.
        self.o_proj = nn.Linear(
            dim,
            dim,
            bias=False,
        )

        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Compute causal grouped-query self-attention.

        Args:
            x: Hidden states with shape
                ``[batch, sequence, model_dimension]``.

        Returns:
            Attention output with the same shape as ``x``.
        """
        B, T, C = x.shape

        # Project hidden states into query representations and rearrange
        # them into:
        #
        # [batch, heads, sequence, head_dimension]
        q = self.q_proj(x).view(
            B,
            T,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        # Key and value projections use fewer heads than queries because
        # this module implements Grouped-Query Attention.
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

        # Inject positional information into queries and keys using RoPE.
        q, k = apply_rope(q, k)

        # Repeat each key/value head so that the number of key/value heads
        # matches the number of query heads.
        k = k.repeat_interleave(
            self.repeat,
            dim=1,
        )

        v = v.repeat_interleave(
            self.repeat,
            dim=1,
        )

        # PyTorch's scaled dot-product attention provides an optimized
        # implementation when supported by the current hardware and
        # PyTorch version.
        #
        # is_causal=True prevents each token from attending to future tokens,
        # which is required for autoregressive language modeling.
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

        # Restore the standard Transformer shape:
        #
        # [batch, heads, sequence, head_dim]
        #             ↓
        # [batch, sequence, model_dimension]
        output = (
            output
            .transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        return self.o_proj(output)


class TransformerBlock(nn.Module):
    """Single pre-normalized decoder Transformer block.

    Each block consists of:

        RMSNorm → GQA → residual connection
        RMSNorm → SwiGLU → residual connection

    Args:
        dim: Transformer hidden dimension.
        n_heads: Number of query attention heads.
        n_kv_heads: Number of key/value heads.
        ffn_dim: Intermediate SwiGLU dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
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

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply attention and feed-forward residual transformations.

        Args:
            x: Input hidden states.

        Returns:
            Hidden states after one Transformer block.
        """
        # Pre-normalization followed by causal self-attention and a
        # residual connection.
        x = (
            x
            + self.attention(
                self.norm1(x)
            )
        )

        # Pre-normalization followed by the SwiGLU feed-forward network
        # and a second residual connection.
        x = (
            x
            + self.ffn(
                self.norm2(x)
            )
        )

        return x


class ModernGPT(nn.Module):
    """Decoder-only Transformer language model used by Small-GPT.

    The model maps token IDs to contextual hidden representations and then
    predicts the next token in the sequence.

    Architecture:

        Token IDs
            ↓
        Token Embedding
            ↓
        Transformer Blocks
            ├── RMSNorm
            ├── GQA + RoPE
            ├── Residual Connection
            ├── RMSNorm
            ├── SwiGLU
            └── Residual Connection
            ↓
        RMSNorm
            ↓
        Language Model Head
            ↓
        Vocabulary Logits

    Args:
        vocab_size: Number of tokens in the tokenizer vocabulary.
        block_size: Maximum supported context length.
        dim: Transformer hidden dimension.
        n_layers: Number of Transformer blocks.
        n_heads: Number of query attention heads.
        n_kv_heads: Number of key/value attention heads.
        ffn_dim: Intermediate dimension of the SwiGLU network.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Maximum sequence length supported by this model instance.
        self.block_size = block_size

        # Convert input token IDs into dense vector representations.
        self.embedding = nn.Embedding(
            vocab_size,
            dim,
        )

        # Stack the requested number of decoder Transformer blocks.
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

        # Final normalization before vocabulary prediction.
        self.norm = RMSNorm(dim)

        # Project hidden states into vocabulary logits.
        self.lm_head = nn.Linear(
            dim,
            vocab_size,
            bias=False,
        )

        # Weight tying shares the token embedding matrix with the language
        # model output projection.
        #
        # This reduces the total number of parameters and is a common
        # technique in language-model architectures.
        self.lm_head.weight = (
            self.embedding.weight
        )

        # Initialize model parameters after constructing the complete
        # architecture.
        self.apply(
            self._init_weights
        )

    def _init_weights(
        self,
        module: nn.Module,
    ) -> None:
        """Initialize learnable weights.

        Linear and embedding weights use a normal distribution centered
        around zero with a standard deviation of 0.02.

        Args:
            module: Submodule being initialized.
        """
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
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the model forward pass.

        Args:
            idx: Input token IDs with shape
                ``[batch_size, sequence_length]``.
            targets: Optional target token IDs used to calculate the
                next-token prediction loss.

        Returns:
            A tuple containing:

                - ``logits``: Vocabulary prediction scores.
                - ``loss``: Cross-entropy loss when ``targets`` are provided,
                  otherwise ``None``.
        """
        B, T = idx.shape

        # Prevent sequences longer than the architecture's supported
        # context length from entering the model.
        if T > self.block_size:
            raise ValueError(
                "Sequence exceeds context length."
            )

        # Convert token IDs into dense hidden representations.
        x = self.embedding(idx)

        # Pass the sequence through every Transformer block.
        # Activation checkpointing recomputes a block during backward instead
        # of keeping all intermediate activations in GPU memory.
        use_ckpt = (
            self.training
            and getattr(self, "gradient_checkpointing", False)
        )

        for block in self.blocks:
            if use_ckpt:
                x = activation_checkpoint(
                    block,
                    x,
                    use_reentrant=False,
                )
            else:
                x = block(x)

        # Normalize the final hidden representation.
        x = self.norm(x)

        # Convert hidden states into logits over the complete vocabulary.
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            # Flatten both logits and target IDs so that cross_entropy can
            # calculate next-token prediction loss across every position.
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
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> torch.Tensor:
        """Generate new tokens autoregressively.

        At every generation step, the model:

            1. Keeps only the most recent context window.
            2. Predicts the next-token probability distribution.
            3. Applies temperature scaling.
            4. Optionally applies top-k filtering.
            5. Samples the next token.
            6. Appends it to the existing sequence.

        Args:
            idx: Initial token IDs with shape
                ``[batch_size, sequence_length]``.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature. Higher values increase
                randomness; lower values make sampling more deterministic.
            top_k: If non-zero, only the K highest-scoring tokens are
                considered during sampling.

        Returns:
            Token IDs containing the original prompt followed by the
            generated tokens.
        """
        # Generation does not require gradients, reducing memory usage and
        # computation overhead during inference.
        self.eval()

        for _ in range(max_new_tokens):
            # Limit the input to the model's supported context window.
            # This allows generation to continue even after the sequence
            # becomes longer than block_size.
            idx_cond = idx[
                :,
                -self.block_size:
            ]

            # Predict vocabulary logits for the current context.
            logits, _ = self(idx_cond)

            # Only the final position is needed because it predicts the
            # next token in an autoregressive language model.
            logits = logits[
                :,
                -1,
                :
            ]

            # Temperature scaling controls the sharpness of the probability
            # distribution. The minimum prevents division by zero.
            logits /= max(
                temperature,
                1e-5,
            )

            if top_k:
                # Restrict sampling to the K highest-probability candidates.
                k = min(
                    top_k,
                    logits.size(-1),
                )

                values, _ = torch.topk(
                    logits,
                    k,
                )

                # The final value in each row is the score threshold for
                # the Kth-highest candidate.
                threshold = values[
                    :,
                    [-1],
                ]

                # Remove all candidates outside the top-k set by assigning
                # them negative infinity. They therefore receive zero
                # probability after softmax.
                logits = torch.where(
                    logits < threshold,
                    torch.full_like(
                        logits,
                        float("-inf"),
                    ),
                    logits,
                )

            # Convert filtered logits into a probability distribution.
            probs = F.softmax(
                logits,
                dim=-1,
            )

            # Sample one token from the resulting probability distribution.
            next_token = torch.multinomial(
                probs,
                1,
            )

            # Append the sampled token to the generated sequence.
            idx = torch.cat(
                [idx, next_token],
                dim=1,
            )

        return idx

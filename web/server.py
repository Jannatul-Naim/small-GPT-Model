from pathlib import Path
import json

import torch

from fastapi import FastAPI
from fastapi.responses import (
    FileResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from model.modern_gpt import ModernGPT
from tokenizer.bpe import BPETokenizer


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = ROOT / "checkpoints" / "gpt.pt"

TOKENIZER = ROOT / "data" / "processed" / "vocab.json"

WEB = ROOT / "web"


# ============================================================
# DEVICE
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("                 MyGPT SERVER")
print("=" * 60)

print(f"Device: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"CUDA: {torch.version.cuda}"
    )

print(f"Checkpoint: {CHECKPOINT}")
print(f"Tokenizer:  {TOKENIZER}")


# ============================================================
# CHECK FILES
# ============================================================

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT}"
    )

if not TOKENIZER.exists():
    raise FileNotFoundError(
        f"Tokenizer not found:\n{TOKENIZER}"
    )

if not (WEB / "index.html").exists():
    raise FileNotFoundError(
        f"index.html not found:\n{WEB / 'index.html'}"
    )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("\nLoading checkpoint...")

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("Loading tokenizer...")

tokenizer = BPETokenizer.load(
    TOKENIZER
)


# ============================================================
# CREATE MODEL
# ============================================================

print("Creating model...")

model = ModernGPT(
    **checkpoint["config"]
).to(device)


# ============================================================
# LOAD MODEL WEIGHTS
# ============================================================

print("Loading model weights...")

model.load_state_dict(
    checkpoint["model_state"]
)


# ============================================================
# EVALUATION MODE
# ============================================================

model.eval()


# ============================================================
# MODEL INFORMATION
# ============================================================

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Parameters: {parameter_count:,}"
)

print(
    f"Block size: {model.block_size}"
)

print("=" * 60)
print("Model loaded successfully.")
print("=" * 60)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="MyGPT",
    version="1.0",
)


# ============================================================
# REQUEST MODEL
# ============================================================

class GenerationRequest(BaseModel):

    prompt: str

    max_new_tokens: int = 200

    temperature: float = 0.8

    top_k: int = 40


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/")
def home():

    return FileResponse(
        WEB / "index.html"
    )


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/static",
    StaticFiles(
        directory=WEB
    ),
    name="static",
)


# ============================================================
# TEXT GENERATION
# ============================================================

@app.post("/stream")
def stream(
    request: GenerationRequest
):

    # --------------------------------------------------------
    # Validate request
    # --------------------------------------------------------

    prompt = request.prompt

    max_new_tokens = max(
        1,
        min(
            request.max_new_tokens,
            2048,
        ),
    )

    temperature = max(
        request.temperature,
        1e-5,
    )

    top_k = max(
        1,
        request.top_k,
    )


    # --------------------------------------------------------
    # Encode prompt
    # --------------------------------------------------------

    initial = tokenizer.encode(
        prompt
    )


    # --------------------------------------------------------
    # Generator
    # --------------------------------------------------------

    def iterator():

        # IMPORTANT:
        #
        # generated must be created INSIDE iterator().
        #
        # Otherwise Python considers generated a local
        # variable because we modify it later in this function.
        #

        generated = torch.tensor(
            [initial],
            dtype=torch.long,
            device=device,
        )


        # ----------------------------------------------------
        # Generate tokens
        # ----------------------------------------------------

        for _ in range(
            max_new_tokens
        ):

            with torch.no_grad():

                # --------------------------------------------
                # Keep only the model context window
                # --------------------------------------------

                context = generated[
                    :,
                    -model.block_size:
                ]


                # --------------------------------------------
                # Forward pass
                # --------------------------------------------

                logits, _ = model(
                    context
                )


                # --------------------------------------------
                # Last token logits
                # --------------------------------------------

                logits = logits[
                    :,
                    -1,
                    :
                ]


                # --------------------------------------------
                # Temperature
                # --------------------------------------------

                logits = (
                    logits / temperature
                )


                # --------------------------------------------
                # Top-K sampling
                # --------------------------------------------

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


                # --------------------------------------------
                # Convert logits to probabilities
                # --------------------------------------------

                probs = torch.softmax(
                    logits,
                    dim=-1,
                )


                # --------------------------------------------
                # Sample next token
                # --------------------------------------------

                next_token = (
                    torch.multinomial(
                        probs,
                        1,
                    )
                )


                # --------------------------------------------
                # Append token
                # --------------------------------------------

                generated = torch.cat(
                    [
                        generated,
                        next_token,
                    ],
                    dim=1,
                )


                # --------------------------------------------
                # Decode token
                # --------------------------------------------

                token = tokenizer.decode(
                    [
                        next_token.item()
                    ]
                )


            # ------------------------------------------------
            # Send token to browser using SSE
            # ------------------------------------------------

            yield (
                "data: "
                + json.dumps(
                    {
                        "token": token
                    }
                )
                + "\n\n"
            )


        # ----------------------------------------------------
        # Generation finished
        # ----------------------------------------------------

        yield (
            "data: [DONE]\n\n"
        )


    # ========================================================
    # RETURN STREAM
    # ========================================================

    return StreamingResponse(
        iterator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# SERVER STARTUP
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "web.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )

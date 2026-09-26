# Ternary Bonsai 2 27B notebook API

Paste the **entire contents of [`colab_kaggle_cell.py`](colab_kaggle_cell.py)** into one Python cell on Google Colab or Kaggle. Select a NVIDIA GPU runtime (T4 or T4×2), enable notebook Internet access, and run. No separate setup cell is required. The cell downloads the official PrismML CUDA runtime and the official `prism-ml/Ternary-Bonsai-2-27B-gguf` PQ2_0 GGUF. It verifies the Hugging Face LFS SHA-256 and GGUF identity before starting inference. It does not include model weights, train or convert models, or fall back to a different model.

The notebook prints an authenticated Cloudflare Quick Tunnel URL, generated key, OpenAI Python client example and curl example **only after** real local inference, streaming, authentication and remote-access checks pass. `GET /v1/models` and `POST /v1/chat/completions` use the PrismML server directly. Optional `BONSAI_API_KEY` supplies your own key (32 characters minimum); otherwise a random key is created. `HF_TOKEN` can be set as a notebook secret/environment variable if Hugging Face requires authentication. Treat notebook outputs as secrets.

## Operational limitations

This is a **session-scoped notebook deployment**, not a persistent production service: disconnecting the notebook terminates the API; the Quick Tunnel URL is ephemeral and Cloudflare Quick Tunnels have availability/rate limits. For unattended production use, deploy the same official server behind a managed HTTPS ingress and a persistent storage volume. Never publish the printed key. The CUDA binary, GPU memory, network, disk space and Hugging Face access must be available; if any are missing the cell fails rather than presenting a simulated API. First launch downloads several gigabytes and may take significant time. Bonsai 2 has no official speculative drafter in the referenced demo; this deployment does not enable speculation or vision projection. Model and runtime have their own upstream license terms; this repository's Apache-2.0 license covers this notebook code only.

## Verification

`python -m py_compile colab_kaggle_cell.py` checks syntax without a GPU. A full integration test requires running the cell in a supported GPU notebook and is not claimed here. For upstream release behavior consult [PrismML Bonsai-demo](https://github.com/PrismML-Eng/Bonsai-demo), [model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf), and [PrismML runtime](https://github.com/PrismML-Eng/llama.cpp).

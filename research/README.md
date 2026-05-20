# Research notes and logs

This directory holds material that reads like a lab notebook rather than evergreen product docs.

| File | Contents |
| --- | --- |
| [`FINDINGS.md`](FINDINGS.md) | Long-form experiment log: metrics, interpretations, remote artifact paths, and dead ends—the full history |

The curated story for collaborators and newcomers lives in the repository root **`README.md`**.

Compressible **attention surgery** code (for use on an existing HF LM) ships in **`../src/attention_compression/`** — see **`qk_surgery.py`**. Install the parent repo with `pip install -e .` to import `attention_compression.qk_surgery` from another codebase.

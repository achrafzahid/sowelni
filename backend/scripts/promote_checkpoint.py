"""
Strip training-only files from a checkpoint → clean inference directory.

Usage:
    python scripts/promote_checkpoint.py \\
        --checkpoint qwen3-asr-darija/final \\
        --output     darija-asr-production
"""
import argparse, shutil
from pathlib import Path

KEEP_PATTERNS = {
    "config.json", "generation_config.json", "model.safetensors",
    "model.safetensors.index.json", "preprocessor_config.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.json", "merges.txt", "chat_template.json",
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--base_model", default="Qwen/Qwen3-ASR-0.6B")
    args = p.parse_args()

    ckpt, out = Path(args.checkpoint), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    for entry in ckpt.iterdir():
        if entry.name in KEEP_PATTERNS or (
            entry.name.startswith("model-") and entry.name.endswith(".safetensors")
        ):
            shutil.copy2(entry, out / entry.name)
            print(f"  ✓ {entry.name}")

    if not (out / "preprocessor_config.json").exists():
        print(f"\n⚠  preprocessor_config.json missing. Run:")
        print(f"   huggingface-cli download {args.base_model} "
              f"preprocessor_config.json --local-dir {out}")

    size = sum(f.stat().st_size for f in out.iterdir() if f.is_file()) / 1e9
    print(f"\nDone → {out}  ({size:.2f} GB)")

if __name__ == "__main__":
    main()

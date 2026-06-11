"""
Measure WER/CER on a validation jsonl.

Usage:
    python scripts/evaluate_wer.py \\
        --model_dir darija-asr-production \\
        --val_file  val_resolved.jsonl \\
        --limit 200
"""
import argparse, json
import librosa, numpy as np, soundfile as sf, torch
from jiwer import cer, wer
from transformers import AutoModel, AutoProcessor

INSTRUCTION = "Transcribe the spoken Moroccan Arabic (Darija) into Arabic script."

def load_audio(path, sr=16000):
    a, r = sf.read(path)
    if a.ndim > 1: a = a.mean(axis=1)
    a = a.astype(np.float32)
    if r != sr: a = librosa.resample(a, orig_sr=r, target_sr=sr)
    return a

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="darija-asr-production")
    p.add_argument("--val_file", required=True)
    p.add_argument("--audio_key", default="audio")
    p.add_argument("--text_key", default="text")
    p.add_argument("--limit", type=int, default=200)
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    proc = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_dir, trust_remote_code=True, torch_dtype=dt, device_map=dev
    ).eval()

    refs, hyps = [], []
    with open(args.val_file) as f:
        for i, line in enumerate(f):
            if i >= args.limit: break
            row = json.loads(line)
            audio = load_audio(row[args.audio_key])
            conv = [{"role":"user","content":[
                {"type":"audio","audio_url":None},
                {"type":"text","text":INSTRUCTION}]}]
            txt = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            inp = proc(text=txt, audios=[audio], sampling_rate=16000,
                       return_tensors="pt", padding=True)
            inp = {k: v.to(dev) for k, v in inp.items() if isinstance(v, torch.Tensor)}
            with torch.inference_mode():
                ids = model.generate(**inp, max_new_tokens=256, do_sample=False)
            plen = inp["input_ids"].shape[1]
            pred = proc.batch_decode(ids[:, plen:], skip_special_tokens=True)[0].strip()
            refs.append(row[args.text_key]); hyps.append(pred)
            if i < 5: print(f"\n[{i}] REF: {row[args.text_key]}\n     HYP: {pred}")

    print(f"\n── {len(refs)} samples ──")
    print(f"  WER: {wer(refs, hyps)*100:.2f}%   CER: {cer(refs, hyps)*100:.2f}%")

if __name__ == "__main__":
    main()

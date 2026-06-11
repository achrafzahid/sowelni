"""
Fine-tune Qwen3-ASR on Moroccan Darija.

Fixes applied vs the original broken run (loss≈8, grad_norm≈1000):
  1. Label masking — only assistant transcription tokens contribute to loss.
  2. max_grad_norm=1.0
  3. Cosine LR schedule + warmup
  4. Real eval during training
  5. bf16 preferred over fp16

Expected jsonl row:
    {"audio": "/abs/path/to/file.wav", "text": "النص بالدارجة"}

Usage:
    python scripts/train.py \\
        --train_file train_resolved.jsonl \\
        --val_file   val_resolved.jsonl \\
        --output_dir qwen3-asr-darija \\
        --num_epochs 3 --batch_size 2 --grad_accum 8
"""
import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import AutoModel, AutoProcessor, Trainer, TrainingArguments

logger = logging.getLogger(__name__)

ASR_INSTRUCTION = "Transcribe the spoken Moroccan Arabic (Darija) into Arabic script."


class DarijaASRDataset(Dataset):
    def __init__(self, jsonl_path, audio_key="audio", text_key="text",
                 target_sr=16000, max_duration_s=30.0):
        self.audio_key = audio_key
        self.text_key = text_key
        self.target_sr = target_sr
        self.max_samples = int(max_duration_s * target_sr)
        self.rows = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if audio_key in row and text_key in row and row[text_key]:
                    self.rows.append(row)
        logger.info("Loaded %d rows from %s", len(self.rows), jsonl_path)

    def __len__(self):
        return len(self.rows)

    def _load_audio(self, path):
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
        if audio.shape[0] > self.max_samples:
            audio = audio[:self.max_samples]
        return audio

    def __getitem__(self, idx):
        row = self.rows[idx]
        return {"audio": self._load_audio(row[self.audio_key]), "text": row[self.text_key]}


@dataclass
class DarijaCollator:
    """
    THE critical fix: labels = -100 for every token that is NOT the
    assistant's transcription.  Without this the model tries to predict
    the prompt tokens (impossible) and loss stays at 6-8.
    """
    processor: Any
    target_sr: int = 16000

    def _prompt_conv(self):
        return [{"role": "user", "content": [
            {"type": "audio", "audio_url": None},
            {"type": "text", "text": ASR_INSTRUCTION},
        ]}]

    def _full_conv(self, transcription):
        return self._prompt_conv() + [
            {"role": "assistant", "content": transcription}
        ]

    def __call__(self, batch):
        audios = [b["audio"] for b in batch]

        prompt_texts = [
            self.processor.apply_chat_template(
                self._prompt_conv(), add_generation_prompt=True, tokenize=False
            ) for _ in batch
        ]
        full_texts = [
            self.processor.apply_chat_template(
                self._full_conv(b["text"]), add_generation_prompt=False, tokenize=False
            ) for b in batch
        ]

        inputs = self.processor(
            text=full_texts, audios=audios, sampling_rate=self.target_sr,
            return_tensors="pt", padding=True,
        )

        prompt_lens = [
            self.processor.tokenizer(p, return_tensors="pt",
                                     add_special_tokens=False)["input_ids"].shape[1]
            for p in prompt_texts
        ]

        labels = inputs["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = -100

        inputs["labels"] = labels
        return dict(inputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--val_file", required=True)
    parser.add_argument("--output_dir", default="qwen3-asr-darija")
    parser.add_argument("--audio_key", default="audio")
    parser.add_argument("--text_key", default="text")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=32)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    model = AutoModel.from_pretrained(
        args.model_id, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )

    if args.use_lora:
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
        if args.lr == 2e-5:
            args.lr = 1e-4

    train_ds = DarijaASRDataset(args.train_file, args.audio_key, args.text_key)
    val_ds = DarijaASRDataset(args.val_file, args.audio_key, args.text_key)
    collator = DarijaCollator(processor=processor)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,              # ← THE critical fix
        bf16=use_bf16, fp16=not use_bf16,
        gradient_checkpointing=True,
        eval_strategy="steps", eval_steps=500,
        save_strategy="steps", save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="steps", logging_steps=20,
        report_to=[],
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    Trainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator,
    ).train()

    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    logger.info("Done. Best model saved to %s", final_dir)


if __name__ == "__main__":
    main()

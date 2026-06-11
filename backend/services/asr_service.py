"""
ASR service wrapping the Qwen3-ASR model via the official ``qwen-asr`` package.

Qwen3-ASR is NOT a standard HuggingFace transformers model — it ships with a
dedicated wrapper (``Qwen3ASRModel``) that handles audio preprocessing,
language detection, and the multi-stage encoder→decoder pipeline internally.

We keep our threading model unchanged: a single-worker ThreadPoolExecutor +
asyncio.Lock so the Socket.io event loop never blocks during inference.
"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import torch
from qwen_asr import Qwen3ASRModel

logger = logging.getLogger(__name__)


class ASRService:
    def __init__(
        self,
        model_dir: str,
        device: Optional[str] = None,
        language: str = "Arabic",
    ):
        self.model_dir = model_dir
        self.device = device or self._pick_device()
        self.dtype = self._pick_dtype()
        self.language = language
        self.model: Optional[Qwen3ASRModel] = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._lock = asyncio.Lock()

    @staticmethod
    def _pick_device() -> str:
        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _pick_dtype(self):
        if "cuda" in self.device:
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32

    def load(self) -> None:
        logger.info(
            "Loading Qwen3-ASR from %s (device=%s, dtype=%s) …",
            self.model_dir, self.device, self.dtype,
        )
        self.model = Qwen3ASRModel.from_pretrained(
            self.model_dir,
            dtype=self.dtype,
            device_map=self.device,
            max_inference_batch_size=4,
            max_new_tokens=256,
        )
        logger.info("ASR model ready (language=%s).", self.language)

    def _transcribe_sync(self, audio: np.ndarray, sampling_rate: int) -> str:
        """Blocking inference path — runs inside the worker thread."""
        results = self.model.transcribe(
            audio=[(audio, sampling_rate)],
            language=[self.language] if self.language else None,
            return_time_stamps=False,
        )
        if not results:
            return ""
        text = getattr(results[0], "text", "") or ""
        return text.strip()

    async def transcribe(
        self, audio: np.ndarray, sampling_rate: int = 16000
    ) -> str:
        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor, self._transcribe_sync, audio, sampling_rate
            )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
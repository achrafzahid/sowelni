"""
Audio processing helpers.

Decodes standalone WebM/Opus blobs into the float32 numpy arrays the ASR
model expects.  Requires ffmpeg on the host (pydub shells out to it).
"""
import io
import logging

import numpy as np
from pydub import AudioSegment

logger = logging.getLogger(__name__)


def webm_to_float32_array(webm_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
    """
    Decode a complete WebM/Opus byte stream into a mono float32 numpy array
    sampled at *target_sr* Hz, normalized to [-1, 1].
    """
    if len(webm_bytes) < 200:  # too small to be valid WebM
        raise ValueError(f"Chunk too small ({len(webm_bytes)} bytes)")
    audio = AudioSegment.from_file(io.BytesIO(webm_bytes), format="webm")
    audio = audio.set_channels(1).set_frame_rate(target_sr).set_sample_width(2)
    samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


def is_mostly_silence(audio: np.ndarray, threshold: float = 0.01) -> bool:
    """RMS-based silence gate."""
    if audio.size == 0:
        return True
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return rms < threshold

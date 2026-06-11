"""
Gemini streaming service — uses the ``google-genai`` SDK.

The streaming iterator is blocking, so we pump tokens into an asyncio.Queue
from a worker thread.  Callers ``async for`` over the result.
"""
import asyncio
import logging
import os
from typing import AsyncGenerator, Optional

from google import genai
from google.genai.types import Content, GenerateContentConfig, Part

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = (
    "You are a warm, helpful conversational assistant for Moroccan Darija speakers.\n\n"
    "The user speaks to you via voice. What you receive is an automatic transcription "
    "of their Darija speech, so it may contain spelling errors, missing diacritics, or words "
    "that look slightly off — interpret charitably and ask for clarification only when truly "
    "necessary.\n\n"
    "Reply in Darija. Use Arabic script unless the user clearly wrote in Latin (Arabizi), "
    "in which case match their style. Keep replies concise, friendly, and conversational — "
    "imagine you are chatting in person, not writing an essay."
)


def _dicts_to_contents(history: list[dict]) -> list[Content]:
    """Convert plain dicts from session state into SDK Content objects."""
    contents = []
    for entry in history:
        role = entry.get("role", "user")
        parts_raw = entry.get("parts", [])
        parts = [Part(text=p) if isinstance(p, str) else p for p in parts_raw]
        contents.append(Content(role=role, parts=parts))
    return contents


class LLMService:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
    ):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name
        self._client: Optional[genai.Client] = None

    def load(self) -> None:
        if not self.api_key:
            logger.warning("GOOGLE_API_KEY not set — LLM disabled.")
            return
        self._client = genai.Client(api_key=self.api_key)
        logger.info("Gemini client ready (model: %s).", self.model_name)

    async def stream(
        self,
        user_text: str,
        history: Optional[list[dict]] = None,
    ) -> AsyncGenerator[str, None]:
        if self._client is None:
            yield "[LLM not configured — set GOOGLE_API_KEY in .env]"
            return

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()

        def producer() -> None:
            try:
                chat = self._client.chats.create(
                    model=self.model_name,
                    config=GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                    ),
                    history=_dicts_to_contents(history) if history else [],
                )
                for chunk in chat.send_message_stream(user_text):
                    text = getattr(chunk, "text", None)
                    if text:
                        asyncio.run_coroutine_threadsafe(queue.put(text), loop)
            except Exception as exc:
                logger.exception("Gemini streaming failed.")
                asyncio.run_coroutine_threadsafe(
                    queue.put(f"\n[Gemini error: {exc}]"), loop
                )
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(SENTINEL), loop)

        loop.run_in_executor(None, producer)

        while True:
            item = await queue.get()
            if item is SENTINEL:
                break
            yield item

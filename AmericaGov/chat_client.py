from __future__ import annotations

import json
import secrets
import sys
from collections.abc import Iterator, Mapping, Sequence
from http.client import HTTPResponse
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_CHAT_URL = "https://america.gov/api/chat"


def _new_id() -> str:
    return secrets.token_urlsafe(12)


class ChatClientError(RuntimeError):
    """Raised when the chat endpoint returns an error or invalid stream data."""


class ChatClient:
    def __init__(
        self,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 60,
    ) -> None:
        self.endpoint = DEFAULT_CHAT_URL
        self.headers = dict(headers or {})
        self.timeout = timeout

    def send(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        chat_id: str | None = None,
        trigger: str = "submit-message",
        message_id: str | None = None,
        language: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Post messages and yield each decoded AI UI stream chunk."""
        payload: dict[str, Any] = {
            "messages": self._prepare_messages(messages),
            "trigger": trigger,
        }
        if chat_id is not None:
            payload["id"] = chat_id
        if message_id is None:
            message_id = next(
                (message.get("id") for message in reversed(messages) if message.get("role") == "user"),
                None,
            )
        if message_id is not None:
            payload["messageId"] = message_id
        if language is not None:
            payload["language"] = language

        request_headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            **self.headers,
        }
        request = Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=request_headers,
            method="POST",
        )

        try:
            response = urlopen(request, timeout=self.timeout)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ChatClientError(f"Chat endpoint returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise ChatClientError(f"Could not reach chat endpoint: {exc.reason}") from exc

        with response:
            yield from self._read_stream(response)

    @staticmethod
    def _prepare_messages(messages: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        prepared = []
        for message in messages:
            if message.get("role") != "assistant":
                prepared.append(message)
                continue

            metadata = message.get("metadata")
            parts = message.get("parts", [])
            text_parts = [
                part
                for part in parts
                if isinstance(part, Mapping) and part.get("type") == "text"
            ] if isinstance(parts, Sequence) else []
            last_text_part = text_parts[-1] if text_parts else {}
            provider_metadata = last_text_part.get("providerMetadata", {})
            america_history = (
                provider_metadata.get("americaHistory", {})
                if isinstance(provider_metadata, Mapping)
                else {}
            )
            signature = (
                america_history.get("signature")
                if isinstance(america_history, Mapping)
                else None
            )
            if not isinstance(signature, str) and isinstance(metadata, Mapping):
                signature = metadata.get("interruptedSignature")
            if not isinstance(signature, str) and isinstance(metadata, Mapping):
                history_signature = metadata.get("historySignature")
                if isinstance(history_signature, str):
                    prepared.append(message)
                    continue
            if not isinstance(signature, str) or not signature:
                continue

            normalized: dict[str, Any] = {
                "role": "assistant",
                "parts": [{
                    "type": "text",
                    "text": "\n".join(part.get("text", "") for part in text_parts),
                }],
                "metadata": {"interruptedSignature": signature},
            }
            if "id" in message:
                normalized["id"] = message["id"]
            if isinstance(metadata, Mapping) and metadata.get("evidenceDegraded") is True:
                normalized["metadata"]["evidenceDegraded"] = True
            prepared.append(normalized)
        return prepared

    @staticmethod
    def _read_stream(response: HTTPResponse) -> Iterator[dict[str, Any]]:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("data:"):
                line = line[5:].lstrip()
            if line == "[DONE]":
                return
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ChatClientError(f"Invalid JSON stream chunk: {line!r}") from exc
            if isinstance(chunk, dict):
                if chunk.get("type") == "reasoning-start":
                    yield {"type": "status", "status": "thinking"}
                    continue
                if chunk.get("type") in {"reasoning-delta", "reasoning-end"}:
                    continue
                if chunk.get("type") == "data-step-checkpoint":
                    data = chunk.get("data")
                    if isinstance(data, dict) and isinstance(data.get("signature"), str):
                        yield {
                            "type": "message-metadata",
                            "messageMetadata": {"historySignature": data["signature"]},
                        }
                    continue
                yield chunk
            else:
                raise ChatClientError("Expected each stream chunk to be a JSON object")


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Chat with the America.gov streaming chat endpoint.")
    parser.add_argument("message", nargs="?", help="Optional one-shot message; omit to start an interactive chat")
    parser.add_argument("--language", default="en", help="Language code (default: en)")
    args = parser.parse_args(argv)

    client = ChatClient()
    messages: list[dict[str, Any]] = []
    chat_id = _new_id()

    def send_prompt(prompt: str) -> None:
        user_message = {
            "id": _new_id(),
            "role": "user",
            "parts": [{"type": "text", "text": prompt}],
        }
        messages.append(user_message)
        response_text: list[str] = []
        response_metadata: dict[str, Any] = {}
        response_provider_metadata: dict[str, Any] = {}
        assistant_id = None
        thinking_status_shown = False
        answer_started = False
        print("Assistant: ", end="", flush=True)
        try:
            for chunk in client.send(messages, chat_id=chat_id, language=args.language):
                if chunk.get("type") == "start" and isinstance(chunk.get("messageId"), str):
                    assistant_id = chunk["messageId"]
                if chunk.get("type") in {"text-start", "text-delta", "text-end"}:
                    provider_metadata = chunk.get("providerMetadata")
                    if isinstance(provider_metadata, dict):
                        response_provider_metadata.update(provider_metadata)
                if chunk.get("type") == "text-delta":
                    delta = chunk.get("delta", "")
                    if isinstance(delta, str):
                        if thinking_status_shown and not answer_started:
                            print("Answer: ", end="", flush=True)
                        answer_started = True
                        print(delta, end="", flush=True)
                        response_text.append(delta)
                elif chunk.get("type") == "status" and chunk.get("status") == "thinking":
                    if not thinking_status_shown:
                        print("Thinking...", flush=True)
                        thinking_status_shown = True
                elif chunk.get("type") in {"message-metadata", "finish"}:
                    metadata = chunk.get("messageMetadata")
                    if isinstance(metadata, dict):
                        response_metadata.update(metadata)
                elif chunk.get("type") == "error":
                    print(chunk.get("errorText", "Stream error"), file=sys.stderr)
        except ChatClientError as exc:
            messages.pop()
            print(f"\nError: {exc}", file=sys.stderr)
            return

        print()
        if response_text:
            text_part = {"type": "text", "text": "".join(response_text), "state": "done"}
            if response_provider_metadata:
                text_part["providerMetadata"] = response_provider_metadata
            assistant_message = {
                "id": assistant_id or _new_id(),
                "role": "assistant",
                "parts": [text_part],
            }
            if isinstance(response_metadata.get("historySignature"), str):
                assistant_message["metadata"] = response_metadata
            messages.append(assistant_message)

    if args.message is not None:
        send_prompt(args.message)
    else:
        print("Chat ready. Enter /reset to clear the conversation or /quit to exit.")
        while True:
            try:
                prompt = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if prompt.lower() in {"/quit", "/exit"}:
                break
            if prompt.lower() == "/reset":
                messages.clear()
                chat_id = _new_id()
                print("Conversation cleared.")
                continue
            if prompt:
                send_prompt(prompt)


if __name__ == "__main__":
    main()

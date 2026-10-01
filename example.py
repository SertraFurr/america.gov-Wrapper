import secrets

from AmericaGov import ChatClient, ChatClientError


def main() -> None:
    client = ChatClient()
    messages = []
    chat_id = secrets.token_urlsafe(12)

    print("Chat ready. Enter /reset to clear the conversation or /quit to exit.")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.lower() in {"/quit", "/exit"}:
            break
        if question.lower() == "/reset":
            messages.clear()
            chat_id = secrets.token_urlsafe(12)
            print("Conversation cleared.")
            continue
        if not question:
            continue

        messages.append({
            "id": secrets.token_urlsafe(12),
            "role": "user",
            "parts": [{"type": "text", "text": question}],
        })
        answer = []
        response_metadata = {}
        response_provider_metadata = {}
        assistant_id = None
        thinking_status_shown = False
        answer_started = False
        print("Assistant: ", end="", flush=True)

        try:
            for chunk in client.send(messages, chat_id=chat_id):
                if chunk.get("type") == "start" and isinstance(chunk.get("messageId"), str):
                    assistant_id = chunk["messageId"]
                if chunk.get("type") in {"text-start", "text-delta", "text-end"}:
                    provider_metadata = chunk.get("providerMetadata")
                    if isinstance(provider_metadata, dict):
                        response_provider_metadata.update(provider_metadata)
                if chunk.get("type") == "text-delta":
                    text = chunk.get("delta", "")
                    if isinstance(text, str):
                        if thinking_status_shown and not answer_started:
                            print("Answer: ", end="", flush=True)
                        answer_started = True
                        print(text, end="", flush=True)
                        answer.append(text)
                elif chunk.get("type") == "status" and chunk.get("status") == "thinking":
                    if not thinking_status_shown:
                        print("Thinking...", flush=True)
                        thinking_status_shown = True
                elif chunk.get("type") in {"message-metadata", "finish"}:
                    metadata = chunk.get("messageMetadata")
                    if isinstance(metadata, dict):
                        response_metadata.update(metadata)
        except ChatClientError as error:
            messages.pop()
            print(f"\nError: {error}")
            continue

        print()
        if answer:
            text_part = {"type": "text", "text": "".join(answer), "state": "done"}
            if response_provider_metadata:
                text_part["providerMetadata"] = response_provider_metadata
            assistant_message = {
                "id": assistant_id or secrets.token_urlsafe(12),
                "role": "assistant",
                "parts": [text_part],
            }
            if isinstance(response_metadata.get("historySignature"), str):
                assistant_message["metadata"] = response_metadata
            messages.append(assistant_message)


if __name__ == "__main__":
    main()

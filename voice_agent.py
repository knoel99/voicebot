"""Transcribe audio turns and keep a short conversation history."""

import os

from mistralai.client import Mistral

MODEL = "voxtral-mini-transcribe-3-0"


def transcribe(client: Mistral, audio: str) -> str:
    if audio.startswith(("http://", "https://")):
        result = client.audio.transcriptions.complete(model=MODEL, file_url=audio)
    else:
        with open(audio, "rb") as handle:
            result = client.audio.transcriptions.complete(
                model=MODEL,
                file={"file_name": os.path.basename(audio), "content": handle},
            )
    return result.text


def main() -> None:
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    history: list[str] = []
    state = "listen"
    print("Paste an audio URL or path. Empty line ends the conversation.")
    while state == "listen":
        audio = input("audio> ").strip()
        if audio == "":
            state = "done"
            continue
        text = transcribe(client, audio)
        history.append(text)
        print(f"turn {len(history)}: {text}")
        if len(history) > 1:
            print(f"previous: {history[-2]}")
    print(f"ended after {len(history)} turn(s)")


if __name__ == "__main__":
    main()

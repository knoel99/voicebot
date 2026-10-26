"""Transcribe one audio URL or local file and print the text."""

import os
import sys

from mistralai.client import Mistral

MODEL = "voxtral-mini-transcribe-3-0"


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python transcribe.py <audio-url-or-path>", file=sys.stderr)
        sys.exit(2)

    audio = sys.argv[1]
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    if audio.startswith(("http://", "https://")):
        result = client.audio.transcriptions.complete(model=MODEL, file_url=audio)
    else:
        with open(audio, "rb") as handle:
            result = client.audio.transcriptions.complete(
                model=MODEL,
                file={"file_name": os.path.basename(audio), "content": handle},
            )
    print(result.text)


if __name__ == "__main__":
    main()

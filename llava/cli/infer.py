import argparse

from termcolor import colored

import llava
from llava import conversation as clib
from llava.media import Image, Video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "-m", type=str, required=True)
    parser.add_argument("--conv-mode", "-c", type=str, default="auto")
    parser.add_argument("--text", type=str)
    parser.add_argument("--media", type=str, nargs="+")
    args = parser.parse_args()

    # Load model
    model = llava.load(args.model_path)

    # Set conversation mode
    clib.default_conversation = clib.conv_templates[args.conv_mode].copy()

    # Prepare multi-modal prompt
    prompt = []
    if args.media is not None:
        for media in args.media or []:
            if any(media.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                media = Image(media)
            elif any(media.endswith(ext) for ext in [".mp4", ".mkv", ".webm"]):
                media = Video(media)
            else:
                raise ValueError(f"Unsupported media type: {media}")
            prompt.append(media)
    if args.text is not None:
        prompt.append(args.text)

    # Generate response
    response = model.generate_content(prompt)
    print(colored(response, "cyan", attrs=["bold"]))


if __name__ == "__main__":
    main()



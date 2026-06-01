import os
import sys
from typing import cast

from dotenv import load_dotenv
from langfuse import get_client
from langfuse.model import ChatMessageWithPlaceholdersDict_Message


def get_prompts():
    prompts = dict()

    for root, dirs, files in os.walk("../prompts"):
        if len(files) == 0:
            continue

        name = root.replace("../prompts/", "")
        parts = []
        for file in files:
            with open(f"{root}/{file}", "r") as f:
                content = f.read()
                index, role = os.path.splitext(file)[0].split("-", 1)
                parts.append({"role": role, "content": content, "index": int(index)})
        parts.sort(key=lambda x: x["index"])
        prompts[name] = parts

    return prompts


def compare_prompts(prompt, parts):
    if len(prompt.prompt) != len(parts):
        return False

    for prompt_part, part in zip(prompt.prompt, parts):
        if (
            prompt_part["role"] != part["role"]
            or prompt_part["content"] != part["content"]
        ):
            return False

    return True


def up():
    langfuse = get_client()
    for name, parts in get_prompts().items():
        same_prompt = True
        try:
            prompt = langfuse.get_prompt(name, type="chat", label="latest")
            same_prompt = compare_prompts(prompt, parts)
        except Exception:
            print(f"NEW: {name}")
            same_prompt = False

        if not same_prompt:
            langfuse.create_prompt(
                name=name,
                type="chat",
                prompt=[
                    {"role": part["role"], "content": part["content"]} for part in parts
                ],
                labels=["production", "latest"],
            )
            print(f"Up: {name}")


def get_parts(name: str):
    # for all .md files in folder ../prompts/{name}, read content and role from filename, order by index
    parts = []
    for root, dirs, files in os.walk(f"../prompts/{name}"):
        for file in files:
            with open(f"{root}/{file}", "r") as f:
                content = f.read()
                index, role = os.path.splitext(file)[0].split("-", 1)
                parts.append({"role": role, "content": content, "index": int(index)})
    parts.sort(key=lambda x: x["index"])
    return parts


def down():
    langfuse = get_client()
    response = langfuse.api.prompts.list(label="latest", limit=100)
    if response.meta.limit < response.meta.total_items:
        print("Error: Limit is less that total Items, please change code")
        return

    for prompt in response.data:
        name = prompt.name

        prompt = langfuse.get_prompt(name, type="chat", label="latest")
        parts = get_parts(name)

        if not compare_prompts(prompt, parts):
            print(f"Down: {name}")
            # delete all files in ../prompts/{name}
            for root, dirs, files in os.walk(f"../prompts/{name}"):
                for file in files:
                    os.remove(f"{root}/{file}")
            # write all prompts to ../prompts/{name} with filename {index}-{role}.md
            for index, part in enumerate(
                cast(list[ChatMessageWithPlaceholdersDict_Message], prompt.prompt)
            ):
                role = part["role"]
                content = part["content"]
                os.makedirs(f"../prompts/{name}", exist_ok=True)
                with open(f"../prompts/{name}/{index}-{role}.md", "w") as f:
                    f.write(content)
                print(f"  {index}-{role}.md")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="Langfuse Prompt Sync",
        description="Sync prompts between local files and Langfuse",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["down", "up"],
        required=True,
        help="Sync direction: 'up' (local → Langfuse) or 'down' (Langfuse → local)",
    )
    args = parser.parse_args()

    # Load env
    load_dotenv()
    # Set up langfuse
    langfuse = get_client()
    if not langfuse.auth_check():
        print("Langfuse Authentication failed. Please check your credentials and host.")
        sys.exit(1)

    if args.type == "down":
        down()
        print("✓ Synced all changes from Langfuse to local")
    elif args.type == "up":
        up()
        print("✓ Synced all changes from local to Langfuse")

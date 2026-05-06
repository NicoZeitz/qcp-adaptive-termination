import os

from modules.initialization import initialize, langfuse


def get_names():
    names = set()

    for root, _dirs, files in os.walk("../prompts"):
        for file in files:
            names.add(root.replace("../prompts/", "") + "/" + file.replace(".md", ""))

    return names


def up():
    for name in get_names():
        with open(f"../prompts/{name}.md", "r") as f:
            prompt_content = f.read()

        same_prompt = True
        try:
            prompt = langfuse.get_prompt(name, label="latest")
            same_prompt = prompt.prompt == prompt_content
        except Exception:
            print(f"NEW: {name}")
            same_prompt = False

        if not same_prompt:
            langfuse.create_prompt(
                name=name,
                prompt=prompt_content,
                type="text",
                labels=["production", "latest"],
            )
            print(f"Up: {name}")


def down():
    response = langfuse.api.prompts.list(label="latest", limit=100)
    if response.meta.limit < response.meta.total_items:
        print("Error: Limit is less that total Items, please change code")
        return

    for prompt in response.data:
        name = prompt.name
        with open(f"../prompts/{name}.md", "w") as f:
            prompt = langfuse.get_prompt(name, label="latest")
            f.write(prompt.prompt)
            print(f"Down: {name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="Langfuse Prompt Sync",
        description="Sync prompts between local files and Langfuse"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["down", "up"],
        required=True,
        help="Sync direction: 'up' (local → Langfuse) or 'down' (Langfuse → local)"
    )
    args = parser.parse_args()

    initialize()
    if args.type == "down":
        down()
        print("✓ Synced all changes from Langfuse to local")
    elif args.type == "up":
        up()
        print("✓ Synced all changes from local to Langfuse")

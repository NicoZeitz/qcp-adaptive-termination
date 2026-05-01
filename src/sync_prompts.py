import os

from modules.initialization import initialize, langfuse

given_names = [
    "Query Coordinator/Aggregate/Instructions",
    "Query Coordinator/Aggregate/Prompt",
    "Query Coordinator/Local Replan/Instructions",
    "Query Coordinator/Local Replan/Prompt",
    "Query Coordinator/Query Plan/Instructions",
    "Query Coordinator/Query Plan/Prompt",
    "Query Coordinator/Query Replan/Instructions",
    "Query Coordinator/Query Replan/Prompt",
    "Query Coordinator/Relevance/Instructions",
    "Query Coordinator/Relevance/Prompt",
    "SQL Data Agent/Code/Instructions",
    "SQL Data Agent/Code/Prompt",
    "SQL Data Agent/Relevance/Instructions",
    "SQL Data Agent/Relevance/Prompt",
    "SQL Data Agent/Schema Description/Instructions",
    "SQL Data Agent/Schema Description/Prompt",
    "Text Data Agent/Code/Instructions",
    "Text Data Agent/Code/Prompt",
    "Text Data Agent/Relevance/Instructions",
    "Text Data Agent/Relevance/Prompt",
    "Text Data Agent/Summary/Instructions",
    "Text Data Agent/Summary/Prompt",
]

# add to names
def get_names():
    names = set(given_names)

    for root, _dirs, files in os.walk("../prompts"):
        for file in files:
            names.add(root.replace("../prompts/", "") + "/" + file.replace(".md", ""))

    return names

def main():
    for name in get_names():
        with open(f"../prompts/{name}.md") as f:
            prompt_content = f.read()

        same_prompt = True
        try:
            prompt = langfuse.get_prompt(name, label="latest")
            same_prompt = prompt.prompt == prompt_content
        except Exception:
            print("  Prompt does not exist")
            same_prompt = False

        if not same_prompt:
            langfuse.create_prompt(
                name=name,
                prompt=prompt_content,
                type="text",
                labels=["production", "latest"]
            )
            print(f"Created prompt: {name}")

if __name__ == "__main__":
    initialize()
    main()
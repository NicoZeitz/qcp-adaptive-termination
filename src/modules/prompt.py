from typing import Any
import chevron
from langfuse import get_client

# TODO: https://langfuse.com/integrations/frameworks/openai-agents#link-langfuse-prompt
def load_prompt(name: str, parameters: dict[str, Any]) -> str:
    print("Getting prompt: " + name)
    prompt = get_client().get_prompt(name)
    return prompt.compile(**parameters)
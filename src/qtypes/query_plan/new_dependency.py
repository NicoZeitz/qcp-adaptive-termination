from dataclasses import dataclass

from pydantic import JsonValue

from utils.preview import CHAR_LIMIT_SPAN, preview_text


@dataclass
class NewDependency:
    new_dependencies: list[str]

    def to_span_data(self) -> JsonValue:
        return {
            "new_dependencies": [
                preview_text(dep, CHAR_LIMIT_SPAN) for dep in self.new_dependencies
            ],
        }

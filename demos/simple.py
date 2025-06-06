from enum import StrEnum

from anthropic import Anthropic
from pydantic import BaseModel

from llmstruct import extract_json_from_text


class Alignment(StrEnum):
    """Represents a character's moral alignment."""

    CHAOTIC_GOOD = "Chaotic Good"
    NEUTRAL_GOOD = "Neutral Good"
    LAWFUL_GOOD = "Lawful Good"
    CHAOTIC_NEUTRAL = "Chaotic Neutral"
    NEUTRAL = "Neutral"
    LAWFUL_NEUTRAL = "Lawful Neutral"
    CHAOTIC_EVIL = "Chaotic Evil"
    NEUTRAL_EVIL = "Neutral Evil"
    LAWFUL_EVIL = "Lawful Evil"


class Character(BaseModel):
    """Represents a character in a fantasy role-playing game."""

    name: str
    origin: str
    interests: list[str]
    alignment: Alignment


def main():
    """A simple demonstration of the library's capabilities."""
    client = Anthropic()

    print("--- Simple Demo ---")
    print("Asking Anthropic for a character profile...")

    message = client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"Please generate a random character profile for a fantasy RPG. The JSON should conform to this pydantic model: {Character.model_json_schema()}",
            }
        ],
    )

    response_text = message.content[0].text
    print(f"Input text from LLM:\n{response_text}")

    result = extract_json_from_text(response_text, Character)

    print(f"\nExtraction status: {result.status.value}")
    if result.parsed_objects:
        print("Parsed objects:")
        for obj in result.parsed_objects:
            print(f"- {obj.model_dump_json(indent=2)}")
    else:
        print("No objects were parsed.")

    print("-" * 20)


if __name__ == "__main__":
    main()

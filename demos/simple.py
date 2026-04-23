from enum import StrEnum
from typing import Annotated

from annotated_types import Len
from anthropic import Anthropic
from pydantic import BaseModel

from llmstruct import ExtractionError, ValidationError, extract_structure_from_text


class Power(StrEnum):
    """Represents a hero's power."""

    FLIGHT = "Flight"
    SUPER_STRENGTH = "Super Strength"
    SUPER_SPEED = "Super Speed"
    INVISIBILITY = "Invisibility"
    TELEPORTATION = "Teleportation"
    TIME_CONTROL = "Time Control"
    TELEKINESIS = "Telekinesis"


class Superhero(BaseModel):
    """A basic superhero."""

    real_name: str
    cover_name: str
    origin: str
    interests: tuple[str, ...]
    powers: Annotated[tuple[Power, ...], Len(min_length=1, max_length=3)]


def main():
    client = Anthropic()

    prompt = f"""
    Generate two random superheroes.
    Explain how they are similar and different, and how they met.
    Then, write the heroes' data as a JSON array,
    where each object conforms to this schema:
    {Superhero.model_json_schema()}
    """

    message = client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    response_text = message.content[0].text

    try:
        parsed_objects = extract_structure_from_text(response_text, Superhero)
        for obj in parsed_objects:
            print(obj.model_dump_json(indent=2))
    except (ExtractionError, ValidationError) as e:
        print(f"An error occurred during extraction or validation: {e}")
        print("Response text:")
        print(response_text)


if __name__ == "__main__":
    main()

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError


class ExtractionStatus(StrEnum):
    """Status of the JSON extraction process."""

    SUCCESS = "success"
    FAILURE = "failure"


M = TypeVar("M", bound=BaseModel)


@dataclass
class Result(Generic[M]):
    """
    Holds the result of the JSON extraction.

    Attributes:
        status: The status of the extraction (SUCCESS or FAILURE).
        parsed_objects: A tuple of parsed Pydantic objects if successful,
                        otherwise an empty tuple.
    """

    status: ExtractionStatus
    parsed_objects: tuple[M, ...] = field(default_factory=tuple)


def _extract_balanced_segment(
    text_segment: str, *, open_char: str, close_char: str
) -> str | None:
    """
    Extracts a balanced segment starting with open_char and ending with the
    corresponding close_char from the beginning of text_segment.
    Assumes text_segment[0] == open_char.
    Handles strings and escaped characters within strings.
    Returns the balanced segment string or None if not found or malformed.
    """
    if not text_segment or text_segment[0] != open_char:
        return None

    balance = 0
    in_string = False
    escape_next_char = False

    for i, char in enumerate(text_segment):
        if escape_next_char:
            escape_next_char = False
            continue  # This character is escaped, don't process it further for special meaning

        if char == "\\":
            escape_next_char = True
            continue

        if char == '"':
            in_string = not in_string  # Toggle in_string state

        if not in_string:  # Only adjust balance if not inside a string
            if char == open_char:
                balance += 1
            elif char == close_char:
                balance -= 1

        if balance == 0:
            # We found the closing bracket for the initial opening one
            # Ensure we are not ending in the middle of a string if something is malformed,
            # though json.loads will primarily handle this.
            # For this function's purpose, balance == 0 is the primary signal.
            return text_segment[: i + 1]

    return None  # Unbalanced


def extract_json_from_text(text: str, model_type: type[M]) -> Result[M]:
    """
    Extracts JSON (object or array of objects) from plain text and parses it
    using the provided Pydantic model.

    It looks for the first occurrence of a valid JSON object or array
    that can be successfully parsed into the given Pydantic model type.

    Args:
        text: The input plain text, potentially containing JSON.
        model_type: The Pydantic model class to parse the JSON into.

    Returns:
        A Result object containing the status and a tuple of parsed Pydantic objects.
    """
    if not text:
        return Result(status=ExtractionStatus.FAILURE)

    for i in range(len(text)):
        char = text[i]
        json_candidate_str: str | None = None

        # Potential start of a JSON object
        if char == "{":
            json_candidate_str = _extract_balanced_segment(
                text[i:], open_char="{", close_char="}"
            )
        # Potential start of a JSON array
        elif char == "[":
            json_candidate_str = _extract_balanced_segment(
                text[i:], open_char="[", close_char="]"
            )

        if json_candidate_str:
            try:
                loaded_json_data = json.loads(json_candidate_str)

                if isinstance(loaded_json_data, dict):
                    # It's a JSON object, try to parse as a single model instance
                    parsed_model_instance = model_type.model_validate(loaded_json_data)
                    return Result(
                        status=ExtractionStatus.SUCCESS,
                        parsed_objects=(parsed_model_instance,),
                    )
                elif isinstance(loaded_json_data, list):
                    # It's a JSON array, try to parse each item as a model instance
                    parsed_objects_list: list[M] = []
                    for item_idx, item in enumerate(loaded_json_data):
                        if not isinstance(item, dict):
                            # If an item in the list is not a dict, it can't be parsed
                            # into a typical Pydantic model instance.
                            # This whole array candidate is considered a mismatch for the model.
                            raise ValidationError.from_exception_data(
                                title=model_type.__name__,
                                line_errors=[
                                    {
                                        "type": "dict_expected",
                                        "loc": (item_idx,),
                                        "input": item,
                                    }
                                ],
                            )
                        parsed_objects_list.append(model_type.model_validate(item))

                    if not parsed_objects_list and loaded_json_data:
                        # If the list was not empty but we didn't parse anything (e.g. list of numbers)
                        # this might indicate it wasn't a list of our model_type.
                        # However, an empty list [] is valid and could result in an empty tuple.
                        # The current logic correctly handles empty list as success with 0 objects.
                        pass

                    return Result(
                        status=ExtractionStatus.SUCCESS,
                        parsed_objects=tuple(parsed_objects_list),
                    )
                else:
                    # Valid JSON, but not a dict or list (e.g., a string, number, boolean "true")
                    # This cannot be parsed into a Pydantic BaseModel instance directly in this context.
                    # We continue searching for a dict or list.
                    pass

            except json.JSONDecodeError:
                # The balanced segment was not valid JSON.
                # Continue searching from the next character (or rather, from i+1 in the outer loop).
                pass
            except ValidationError:
                # The JSON was valid, but did not conform to the Pydantic model.
                # Continue searching.
                pass
            # Adding a broad exception catch here can be useful for debugging,
            # but for production, specific exceptions are better.
            # except Exception as e:
            # print(f"Unexpected error during parsing candidate '{json_candidate_str[:50]}...': {e}")
            # pass

    # If the loop finishes, no suitable JSON was found and parsed.
    return Result(status=ExtractionStatus.FAILURE)

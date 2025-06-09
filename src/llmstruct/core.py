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


def extract_structure_from_text(text: str, model_type: type[M]) -> Result[M]:
    """
    Extracts JSON (object or array of objects) from plain text and parses it
    using the provided Pydantic model.
    """
    if not text:
        return Result(status=ExtractionStatus.FAILURE)

    current_pos = 0
    while current_pos < len(text):
        char = text[current_pos]  # Current character being scanned
        json_candidate_str: str | None = None

        # Determine if the current character could start a JSON object or array
        if char == "{":
            json_candidate_str = _extract_balanced_segment(
                text[current_pos:], open_char="{", close_char="}"
            )
        elif char == "[":
            json_candidate_str = _extract_balanced_segment(
                text[current_pos:], open_char="[", close_char="]"
            )

        if json_candidate_str:
            try:
                # Attempt to parse the candidate string as JSON
                loaded_json_data = json.loads(json_candidate_str)

                # If parsing succeeds, try to validate with Pydantic model
                if isinstance(loaded_json_data, dict):
                    parsed_model_instance = model_type.model_validate(loaded_json_data)
                    return Result(
                        status=ExtractionStatus.SUCCESS,
                        parsed_objects=(parsed_model_instance,),
                    )
                elif isinstance(loaded_json_data, list):
                    parsed_objects_list: list[M] = []
                    for item_idx, item in enumerate(loaded_json_data):
                        if not isinstance(item, dict):
                            # This item cannot be a typical Pydantic model instance.
                            # The whole list candidate is considered a mismatch.
                            raise ValidationError.from_exception_data(
                                title=model_type.__name__,
                                line_errors=[
                                    {
                                        "type": "dict_type",
                                        "loc": (item_idx,),
                                        "input": item,
                                    }
                                ],
                            )
                        parsed_objects_list.append(model_type.model_validate(item))
                    return Result(
                        status=ExtractionStatus.SUCCESS,
                        parsed_objects=tuple(parsed_objects_list),
                    )
                else:
                    # Valid JSON, but not a dict or list (e.g., "string", number, true, null).
                    # This cannot be parsed into a Pydantic BaseModel instance.
                    # Skip this entire segment and continue searching.
                    current_pos += len(json_candidate_str)
                    continue

            except ValidationError:
                # Valid JSON structure (json.loads succeeded), but Pydantic validation failed.
                # Skip this entire identified JSON segment and continue search from after it.
                current_pos += len(json_candidate_str)
                continue
            except json.JSONDecodeError:
                # _extract_balanced_segment found something like { ... } or [ ... ],
                # but it wasn't syntactically valid JSON (e.g., "{ [ ... ] }").
                # The initial '{' or '[' at current_pos was misleading for a complete JSON structure.
                # Advance by only 1 to see if the *next* character starts a new, valid segment.
                current_pos += 1
                continue
        else:
            # No balanced segment starting with '{' or '[' was found by _extract_balanced_segment
            # (e.g., it returned None because of imbalance before end of text),
            # or current character is not '{' or '['.
            # Advance by 1 to check the next character.
            current_pos += 1
            # No explicit continue needed here as it's the end of the while loop body
            # and current_pos will be re-checked.

    return Result(status=ExtractionStatus.FAILURE)

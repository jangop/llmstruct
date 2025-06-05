import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Generic, TypeVar

from pydantic import BaseModel, RootModel, ValidationError


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


if __name__ == "__main__":

    class User(BaseModel):
        id: int
        name: str
        is_active: bool = True

    class Product(BaseModel):
        product_id: str
        price: float
        tags: list[str] | None = None

    # Test cases
    text_samples = [
        ("No JSON here, just plain text.", User, ExtractionStatus.FAILURE, 0),
        (
            'Bla bla { "id": 1, "name": "Alice" } some more text.',
            User,
            ExtractionStatus.SUCCESS,
            1,
        ),
        (
            'Text with an array: [{"id": 10, "name": "Bob"}, {"id": 20, "name": "Charlie", "is_active": false}] end.',
            User,
            ExtractionStatus.SUCCESS,
            2,
        ),
        (
            'Malformed JSON { "id": 1, "name": "Alice", ',
            User,
            ExtractionStatus.FAILURE,
            0,
        ),
        (
            'JSON that doesn\'t match model: { "user_id": 1, "username": "Alice" }',
            User,
            ExtractionStatus.FAILURE,
            0,
        ),
        (
            'Some { { { { nested { "id": 3, "name": "NestedValid" } } } } } text',
            User,
            ExtractionStatus.SUCCESS,
            1,
        ),
        (
            'Outer { "key": "value" } then inner valid user: { "id": 4, "name": "InnerUser" }',
            User,
            ExtractionStatus.SUCCESS,
            1,
        ),  # Should pick the first valid User object
        (
            'Product: { "product_id": "P123", "price": 99.99, "tags": ["electronics", "gadget"] }.',
            Product,
            ExtractionStatus.SUCCESS,
            1,
        ),
        (
            'Array of products: [{"product_id":"P001","price":10.0}, {"product_id":"P002","price":20.5, "tags":["new"]}]',
            Product,
            ExtractionStatus.SUCCESS,
            2,
        ),
        (
            'This is tricky: { "not_a_user": true } and then this: { "id": 5, "name": "ValidUserLater" }.',
            User,
            ExtractionStatus.SUCCESS,
            1,
        ),
        (
            'Mismatched array type: [{"id": 6, "name":"OK"}, {"product_id":"P003","price":5.0}]',
            User,
            ExtractionStatus.FAILURE,
            0,
        ),  # Second item is not a User
        (
            'Bla bla [ { "id": 7, "name": "Eve" } ] end.',
            User,
            ExtractionStatus.SUCCESS,
            1,
        ),
        (
            'Leading garbage { [ { "id": 8, "name": "Frank" } ] } trailing.',
            User,
            ExtractionStatus.SUCCESS,
            1,
        ),  # Note: this will find the inner [ ... ] as the first valid JSON array
        (
            'Text with { "id": 100, "name": "Test Name", "is_active": true } and some other stuff.',
            User,
            ExtractionStatus.SUCCESS,
            1,
        ),
        (
            'Text containing a JSON string, not an object: "This is a json string, not an object" and then { "id": 9, "name": "RealObject" }',
            User,
            ExtractionStatus.SUCCESS,
            1,
        ),
        (
            'Bla bla [ { "id": 1, "name": "Valid"}, { "id": "invalid_id_type", "name": "Problem" } ] more bla.',
            User,
            ExtractionStatus.FAILURE,
            0,
        ),
        # New tests for escaped characters and brackets in strings
        (
            'Escaped quotes: { "id": 11, "name": "User with \\"quotes\\" in name" }.',
            User,
            ExtractionStatus.SUCCESS,
            1,
        ),
        (
            'Brackets in string: { "id": 12, "name": "User with {curly} and [square] brackets in name" }.',
            User,
            ExtractionStatus.SUCCESS,
            1,
        ),
        (
            'Array with tricky strings: [ { "id": 13, "name": "String with [bracket] and \\"quote\\" and {brace}" } ]',
            User,
            ExtractionStatus.SUCCESS,
            1,
        ),
        (
            'Malformed JSON due to unclosed string: { "id": 14, "name": "Unclosed string example... }',
            User,
            ExtractionStatus.FAILURE,
            0,
        ),
        (
            "An empty array [] is valid for User model (results in 0 users).",
            User,
            ExtractionStatus.SUCCESS,
            0,
        ),
        (
            "An empty object {} is not a valid User (fails Pydantic validation).",
            User,
            ExtractionStatus.FAILURE,
            0,
        ),
        (
            'Text with only a valid JSON string: "this is just a string"',
            User,
            ExtractionStatus.FAILURE,
            0,
        ),  # Correctly skipped as not dict/list
        (
            "Text with only a valid JSON number: 12345",
            User,
            ExtractionStatus.FAILURE,
            0,
        ),  # Correctly skipped
    ]

    for i, (text, model, expected_status, expected_count) in enumerate(text_samples):
        print(
            f"\n--- Test Case {i + 1}: {text[:50]}{'...' if len(text) > 50 else ''} ---"
        )
        # print(f"Input Text: \"{text[:70]}{'...' if len(text)>70 else ''}\"")
        # print(f"Model: {model.__name__}")

        result = extract_json_from_text(text, model)

        print(f"  Status: {result.status.value} (Expected: {expected_status.value})")
        # assert result.status == expected_status, f"Test {i+1} Status Mismatch: Expected {expected_status}, Got {result.status}"

        if result.status == ExtractionStatus.SUCCESS:
            print(
                f"  Parsed Objects ({len(result.parsed_objects)}): (Expected count: {expected_count})"
            )
            for obj_idx, obj in enumerate(result.parsed_objects):
                print(f"    - Obj {obj_idx}: {obj.model_dump_json(indent=2)}")
            # assert len(result.parsed_objects) == expected_count, \
            #     f"Test {i+1} Count Mismatch: Expected {expected_count}, Got {len(result.parsed_objects)}"
        else:
            # assert expected_count == 0, f"Test {i+1} Expected 0 objects on failure, got {expected_count}"
            print(f"  No objects parsed. (Expected count: {expected_count})")

        if (
            result.status != expected_status
            or (
                result.status == ExtractionStatus.SUCCESS
                and len(result.parsed_objects) != expected_count
            )
            or (result.status == ExtractionStatus.FAILURE and expected_count != 0)
        ):
            print(f"  !!!!!!!! TEST {i + 1} FAILED !!!!!!!!")
            print(f'  Input Text: "{text}"')
            print(f"  Model: {model.__name__}")
            if result.status == ExtractionStatus.SUCCESS:
                print(f"  Actual count: {len(result.parsed_objects)}")
            raise AssertionError(f"Test {i + 1} failed.")

    print("\n--- All tests passed if no assertion errors! ---")

    class UserList(RootModel[list[User]]):
        pass

    try:
        valid_list_data = [
            {"id": 1, "name": "RootUser1"},
            {"id": 2, "name": "RootUser2"},
        ]
        user_list_model = UserList.model_validate(valid_list_data)
        print(f"\nPydantic RootModel[list[User]] parsed: {user_list_model.root}")

        invalid_list_data = [
            {"id": 1, "name": "RootUser1"},
            {"id": "bad", "name": "RootUser2"},
        ]
        UserList.model_validate(invalid_list_data)  # This will raise ValidationError
    except ValidationError as e:
        print(
            "\nPydantic RootModel[list[User]] validation error (expected for invalid_list_data):"
        )
        print(f"  {e.errors(include_input=False)}")

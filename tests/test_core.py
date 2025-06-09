import pytest
from pydantic import BaseModel, RootModel, ValidationError

from llmstruct import ExtractionStatus, extract_structure_from_text


class User(BaseModel):
    id: int
    name: str
    is_active: bool = True


class Product(BaseModel):
    product_id: str
    price: float
    tags: list[str] | None = None


class UserList(RootModel[list[User]]):
    pass


# Test cases data
TEST_CASES = [
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


@pytest.mark.parametrize("text,model,expected_status,expected_count", TEST_CASES)
def test_extract_json_from_text(text, model, expected_status, expected_count):
    """Test JSON extraction from various text inputs."""
    result = extract_structure_from_text(text, model)

    # Check status
    assert result.status == expected_status, (
        f"Status mismatch for text '{text[:50]}...': "
        f"Expected {expected_status}, Got {result.status}"
    )

    # Check count of parsed objects
    if result.status == ExtractionStatus.SUCCESS:
        assert len(result.parsed_objects) == expected_count, (
            f"Count mismatch for text '{text[:50]}...': "
            f"Expected {expected_count}, Got {len(result.parsed_objects)}"
        )
    else:
        assert expected_count == 0, (
            f"Expected 0 objects on failure for text '{text[:50]}...', "
            f"but expected_count was {expected_count}"
        )


def test_user_list_root_model_valid():
    """Test that UserList RootModel works with valid data."""
    valid_list_data = [
        {"id": 1, "name": "RootUser1"},
        {"id": 2, "name": "RootUser2"},
    ]
    user_list_model = UserList.model_validate(valid_list_data)
    assert len(user_list_model.root) == 2
    assert user_list_model.root[0].id == 1
    assert user_list_model.root[0].name == "RootUser1"
    assert user_list_model.root[1].id == 2
    assert user_list_model.root[1].name == "RootUser2"


def test_user_list_root_model_invalid():
    """Test that UserList RootModel raises ValidationError with invalid data."""
    invalid_list_data = [
        {"id": 1, "name": "RootUser1"},
        {"id": "bad", "name": "RootUser2"},  # Invalid id type
    ]

    with pytest.raises(ValidationError) as exc_info:
        UserList.model_validate(invalid_list_data)

    # Check that the error is about the invalid id type
    errors = exc_info.value.errors()
    assert len(errors) > 0
    assert any("type" in str(error) for error in errors)


def test_extract_json_user_object():
    """Test extracting a single User object from text."""
    text = 'Here is a user: { "id": 42, "name": "TestUser", "is_active": false }'
    result = extract_structure_from_text(text, User)

    assert result.status == ExtractionStatus.SUCCESS
    assert len(result.parsed_objects) == 1

    user = result.parsed_objects[0]
    assert user.id == 42
    assert user.name == "TestUser"
    assert user.is_active is False


def test_extract_json_product_array():
    """Test extracting multiple Product objects from an array."""
    text = """Products: [
        {"product_id": "A001", "price": 19.99, "tags": ["electronics"]},
        {"product_id": "B002", "price": 29.99}
    ]"""
    result = extract_structure_from_text(text, Product)

    assert result.status == ExtractionStatus.SUCCESS
    assert len(result.parsed_objects) == 2

    product1 = result.parsed_objects[0]
    assert product1.product_id == "A001"
    assert product1.price == 19.99
    assert product1.tags == ["electronics"]

    product2 = result.parsed_objects[1]
    assert product2.product_id == "B002"
    assert product2.price == 29.99
    assert product2.tags is None


def test_extract_json_no_valid_json():
    """Test that extraction fails gracefully when no valid JSON is found."""
    text = "This is just plain text with no JSON whatsoever."
    result = extract_structure_from_text(text, User)

    assert result.status == ExtractionStatus.FAILURE
    assert len(result.parsed_objects) == 0


def test_extract_json_malformed_json():
    """Test that extraction fails gracefully with malformed JSON."""
    text = 'Malformed: { "id": 1, "name": "Incomplete"'
    result = extract_structure_from_text(text, User)

    assert result.status == ExtractionStatus.FAILURE
    assert len(result.parsed_objects) == 0


def test_extract_json_wrong_schema():
    """Test that extraction fails when JSON doesn't match the model schema."""
    text = 'Wrong schema: { "user_id": 1, "username": "Alice" }'
    result = extract_structure_from_text(text, User)

    assert result.status == ExtractionStatus.FAILURE
    assert len(result.parsed_objects) == 0

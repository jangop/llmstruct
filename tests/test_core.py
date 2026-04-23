import pytest
from pydantic import BaseModel, RootModel
from pydantic import ValidationError as PydanticValidationError

from llmstruct import ExtractionError, ValidationError, extract_structure_from_text


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
    ("No JSON here, just plain text.", User, ExtractionError, 0),
    (
        'Bla bla { "id": 1, "name": "Alice" } some more text.',
        User,
        None,
        1,
    ),
    (
        'Text with an array: [{"id": 10, "name": "Bob"}, {"id": 20, "name": "Charlie", "is_active": false}] end.',
        User,
        None,
        2,
    ),
    (
        'Malformed JSON { "id": 1, "name": "Alice", ',
        User,
        ExtractionError,
        0,
    ),
    (
        'JSON that does not match model: { "user_id": 1, "username": "Alice" }',
        User,
        ValidationError,
        0,
    ),
    (
        'Some { { { { nested { "id": 3, "name": "NestedValid" } } } } } text',
        User,
        None,
        1,
    ),
    (
        'Outer { "key": "value" } then inner valid user: { "id": 4, "name": "InnerUser" }',
        User,
        ValidationError,
        0,
    ),  # First object fails validation, second should be found
    (
        'Product: { "product_id": "P123", "price": 99.99, "tags": ["electronics", "gadget"] }.',
        Product,
        None,
        1,
    ),
    (
        'Array of products: [{"product_id":"P001","price":10.0}, {"product_id":"P002","price":20.5, "tags":["new"]}]',
        Product,
        None,
        2,
    ),
    (
        'This is tricky: { "not_a_user": true } and then this: { "id": 5, "name": "ValidUserLater" }.',
        User,
        ValidationError,
        0,
    ),  # First object fails validation, second should be found
    (
        'Mismatched array type: [{"id": 6, "name":"OK"}, {"product_id":"P003","price":5.0}]',
        User,
        ValidationError,
        0,
    ),  # Second item is not a User
    (
        'Bla bla [ { "id": 7, "name": "Eve" } ] end.',
        User,
        None,
        1,
    ),
    (
        'Leading garbage { [ { "id": 8, "name": "Frank" } ] } trailing.',
        User,
        None,
        1,
    ),  # Note: this will find the inner [ ... ] as the first valid JSON array
    (
        'Text with { "id": 100, "name": "Test Name", "is_active": true } and some other stuff.',
        User,
        None,
        1,
    ),
    (
        'Text containing a JSON string, not an object: "This is a json string, not an object" and then { "id": 9, "name": "RealObject" }',
        User,
        None,
        1,
    ),
    (
        'Bla bla [ { "id": 1, "name": "Valid"}, { "id": "invalid_id_type", "name": "Problem" } ] more bla.',
        User,
        ValidationError,
        0,
    ),
    # New tests for escaped characters and brackets in strings
    (
        'Escaped quotes: { "id": 11, "name": "User with \\"quotes\\" in name" }.',
        User,
        None,
        1,
    ),
    (
        'Brackets in string: { "id": 12, "name": "User with {curly} and [square] brackets in name" }.',
        User,
        None,
        1,
    ),
    (
        'Array with tricky strings: [ { "id": 13, "name": "String with [bracket] and \\"quote\\" and {brace}" } ]',
        User,
        None,
        1,
    ),
    (
        'Malformed JSON due to unclosed string: { "id": 14, "name": "Unclosed string example... }',
        User,
        ExtractionError,
        0,
    ),
    (
        "An empty array [] is valid for User model (results in 0 users).",
        User,
        None,
        0,
    ),
    (
        "An empty object {} is not a valid User (fails Pydantic validation).",
        User,
        ValidationError,
        0,
    ),
    (
        'Text with only a valid JSON string: "this is just a string"',
        User,
        ExtractionError,
        0,
    ),  # Correctly skipped as not dict/list
    (
        "Text with only a valid JSON number: 12345",
        User,
        ExtractionError,
        0,
    ),  # Correctly skipped
]


@pytest.mark.parametrize("text,model,expected_exception,expected_count", TEST_CASES)
def test_extract_structure_from_text(text, model, expected_exception, expected_count):
    """Test JSON extraction from various text inputs."""
    if expected_exception:
        with pytest.raises(expected_exception):
            extract_structure_from_text(text, model)
    else:
        parsed_objects = extract_structure_from_text(text, model)
        assert len(parsed_objects) == expected_count, (
            f"Count mismatch for text '{text[:50]}...': "
            f"Expected {expected_count}, Got {len(parsed_objects)}"
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

    with pytest.raises(PydanticValidationError) as exc_info:
        UserList.model_validate(invalid_list_data)

    # Check that the error is about the invalid id type
    errors = exc_info.value.errors()
    assert len(errors) > 0
    assert any("type" in str(error) for error in errors)


def test_extract_structure_user_object():
    """Test extracting a single User object from text."""
    text = 'Here is a user: { "id": 42, "name": "TestUser", "is_active": false }'
    user = extract_structure_from_text(text, User)[0]

    assert user.id == 42
    assert user.name == "TestUser"
    assert user.is_active is False


def test_extract_structure_product_array():
    """Test extracting multiple Product objects from an array."""
    text = """Products: [
        {"product_id": "A001", "price": 19.99, "tags": ["electronics"]},
        {"product_id": "B002", "price": 29.99}
    ]"""
    products = extract_structure_from_text(text, Product)

    assert len(products) == 2

    product1 = products[0]
    assert product1.product_id == "A001"
    assert product1.price == 19.99
    assert product1.tags == ["electronics"]

    product2 = products[1]
    assert product2.product_id == "B002"
    assert product2.price == 29.99
    assert product2.tags is None


def test_extract_structure_no_valid_json():
    """Test that extraction fails gracefully when no valid JSON is found."""
    text = "This is just plain text with no JSON whatsoever."
    with pytest.raises(ExtractionError):
        extract_structure_from_text(text, User)


def test_extract_structure_malformed_json():
    """Test that extraction fails gracefully with malformed JSON."""
    text = 'Malformed: { "id": 1, "name": "Incomplete"'
    with pytest.raises(ExtractionError):
        extract_structure_from_text(text, User)


def test_extract_structure_wrong_schema():
    """Test that extraction fails when JSON doesn't match the model schema."""
    text = 'Wrong schema: { "user_id": 1, "username": "Alice" }'
    with pytest.raises(ValidationError):
        extract_structure_from_text(text, User)


def test_extract_structure_empty_input():
    """Test that extraction fails with empty input."""
    with pytest.raises(ExtractionError):
        extract_structure_from_text("", User)


def test_extract_structure_json_string():
    """Test that JSON strings (not objects) are skipped."""
    text = 'Text with JSON string: "just a string" and then valid object: { "id": 1, "name": "Valid" }'
    users = extract_structure_from_text(text, User)
    assert len(users) == 1
    assert users[0].id == 1
    assert users[0].name == "Valid"


def test_extract_structure_json_number():
    """Test that JSON numbers are skipped."""
    text = (
        'Text with JSON number: 123 and then valid object: { "id": 2, "name": "Valid" }'
    )
    users = extract_structure_from_text(text, User)
    assert len(users) == 1
    assert users[0].id == 2
    assert users[0].name == "Valid"


def test_extract_structure_json_boolean():
    """Test that JSON booleans are skipped."""
    text = 'Text with JSON boolean: true and then valid object: { "id": 3, "name": "Valid" }'
    users = extract_structure_from_text(text, User)
    assert len(users) == 1
    assert users[0].id == 3
    assert users[0].name == "Valid"


def test_extract_structure_json_null():
    """Test that JSON null is skipped."""
    text = (
        'Text with JSON null: null and then valid object: { "id": 4, "name": "Valid" }'
    )
    users = extract_structure_from_text(text, User)
    assert len(users) == 1
    assert users[0].id == 4
    assert users[0].name == "Valid"


def test_extract_structure_nested_objects():
    """Test extraction from nested object structures."""
    text = 'Outer: { "outer_key": { "id": 5, "name": "NestedUser" } }'
    with pytest.raises(ValidationError):
        extract_structure_from_text(text, User)
    # The function finds the outer object first, which doesn't match User schema


def test_extract_structure_multiple_valid_objects():
    """Test extraction of multiple valid objects in sequence."""
    text = 'First: { "id": 6, "name": "First" } Second: { "id": 7, "name": "Second" }'
    users = extract_structure_from_text(text, User)
    assert len(users) == 1  # Should return the first valid object
    assert users[0].id == 6
    assert users[0].name == "First"


def test_extract_structure_list_with_non_dict_items():
    """Test that lists with non-dict items raise ValidationError."""
    text = '[{"id": 8, "name": "Valid"}, "not a dict", {"id": 9, "name": "AlsoValid"}]'
    with pytest.raises(ValidationError):
        extract_structure_from_text(text, User)


def test_extract_structure_complex_nesting():
    """Test extraction from complex nested structures."""
    text = 'Complex: { "level1": { "level2": [ { "id": 10, "name": "DeepUser" } ] } }'
    with pytest.raises(ValidationError):
        extract_structure_from_text(text, User)
    # The function finds the outer object first, which doesn't match User schema

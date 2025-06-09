# LLMStruct

`llmstruct` is a Python library for reliably extracting structured JSON from text and validating it with Pydantic models.

## Motivation

Many developers prefer to interact directly with an LLM provider's Python module (e.g., `anthropic`, `openai`) to maintain flexibility and avoid heavy abstractions.
While this approach is powerful, it introduces the recurring challenge of parsing structured data, like JSON, from the model's often unstructured text responses.

LLM outputs can be noisy—containing anything from conversational filler to malformed JSON.
Building a resilient extraction and validation pipeline for every project is tedious.

`llmstruct` addresses this by offering a lightweight, focused solution.
It allows you to keep your direct API access while providing a reliable mechanism to extract structured data from text and validate it against Pydantic models.
This handles the messy parts of data extraction,
letting you focus on your core application logic.

## Features

- Extracts JSON objects or arrays of JSON objects from messy text.
- Validates extracted JSON against Pydantic models.
- Resilient to surrounding text and other noise.

## Installation

Simply install the library from pipy using `pip`:

```sh
pip install llmstruct
```

Or add it to your project using `uv`:

```sh
uv add llmstruct
```

## Usage

Here is a quick overview of how to use `llmstruct`:

```python
from pydantic import BaseModel
from anthropic import Anthropic
from llmstruct import extract_structure_from_text

# Define your data structure using Pydantic
class Superhero(BaseModel):
    real_name: str
    cover_name: str
    origin: str
    interests: tuple[str, ...]
    powers: tuple[str, ...]

# Create a client for your LLM provider
# Note: this requires the ANTHROPIC_API_KEY environment variable to be set
client = Anthropic()

# Create a prompt that asks for JSON data
prompt = f"""
Generate two random superheroes.
Explain how they are similar and different, and how they met.
Then, write the heroes' data as a JSON array,
where each object conforms to this schema:
{Superhero.model_json_schema()}
"""

# Get the response from the LLM
message = client.messages.create(
    model="claude-3-5-sonnet-latest",
    max_tokens=1024,
    messages=[{"role": "user", "content": prompt}],
).content[0].text


# Extract the JSON from the response text
result = extract_structure_from_text(message, Superhero)

# Now you can use the parsed and validated objects
for hero in result.parsed_objects:
    print(hero.model_dump_json(indent=2))
```

This will output:

```json
{
  "real_name": "Elena Rodriguez",
  "cover_name": "Quantum Shift",
  "origin": "Brilliant physicist exposed to experimental quantum energy",
  "interests": [
    "Theoretical physics",
    "Urban community development",
    "Martial arts"
  ],
  "powers": [
    "Teleportation",
    "Time Control",
    "Telekinesis"
  ]
}
{
  "real_name": "Jack Harper",
  "cover_name": "Stormweaver",
  "origin": "Electrical engineer transformed by freak lightning strike",
  "interests": [
    "Weather science",
    "Rock climbing",
    "Emergency response"
  ],
  "powers": [
    "Flight",
    "Super Speed",
    "Telekinesis"
  ]
}
```

Note that we did not have to request *only* JSON from the LLM.
In fact, we were able to request the generation of priming context.

## Demos

The `demos` directory contains more examples of how to use this library.
Note that these demos use Anthropic's API, so they need an API key for the client to be instantiated.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

import json

import httpx
from anthropic import Anthropic
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from llmstruct import ExtractionStatus, extract_json_from_text


class ProductSummary(BaseModel):
    """A summary of a product from Open Food Facts."""

    product_name: str = Field(..., description="The name of the product.")
    brand: str = Field(..., description="The brand of the product.")
    ingredients: list[str] = Field(
        ..., description="A list of the product's main ingredients."
    )
    is_vegetarian: bool = Field(
        ..., description="A boolean indicating if the product is vegetarian."
    )
    is_vegan: bool = Field(
        ..., description="A boolean indicating if the product is vegan."
    )


def fetch_product_data(barcode: str) -> dict:
    """Fetches product data from the Open Food Facts API."""
    url = f"https://world.openfoodfacts.org/api/v2/product/{barcode}.json"
    with httpx.Client() as client:
        response = client.get(url)
        response.raise_for_status()
        data = response.json()
        if not data.get("product"):
            raise ValueError(f"Product not found for barcode: {barcode}")
        return data


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    retry=retry_if_exception_type(ValueError),
)
def get_product_summary(client: Anthropic, product_data: dict) -> ProductSummary:
    """
    Asks an LLM to summarize product data and extracts a structured object.
    Uses tenacity to retry on failure.
    """

    prompt = f"""
        Please analyze the following product data from the Open Food Facts API
        and extract a concise summary that conforms to the following JSON schema:
        ```json
        {json.dumps(ProductSummary.model_json_schema(), indent=2)}
        ```

        Here is the product data:
        <product_data>
        {product_data}
        </product_data>
        """

    message = (
        client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        .content[0]
        .text
    )

    result = extract_json_from_text(message, ProductSummary)

    if result.status != ExtractionStatus.SUCCESS:
        print(f"Extraction failed. Status: {result.status.value}")
        print(f"LLM Response:\n---\n{message}\n---")
        raise ValueError("Extraction failed, retrying...")

    return result.parsed_objects[0]


def main():
    """
    An advanced demo that fetches real-world data, uses an LLM to process it,
    and retries with tenacity if the extraction fails.
    """
    client = Anthropic()
    product_barcode = "3017620422003"  # Nutella

    try:
        # Fetch data from an external API
        product_data = fetch_product_data(product_barcode)

        # Use an LLM and llmstruct to get a structured summary, with retries
        summary = get_product_summary(client, product_data)

        print(summary.model_dump_json(indent=2))

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            print(f"\nProcess failed due to API rate limiting: {e}")
        else:
            print(f"\nProcess failed due to an HTTP error: {e}")
    except ValueError as e:
        print(f"\nProcess failed: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    main()

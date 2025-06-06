import httpx
from anthropic import Anthropic
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed

from llmstruct import ExtractionStatus, Result, extract_json_from_text


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


def get_product_from_api(barcode: str) -> dict:
    """Fetches product data from the Open Food Facts API."""
    print(f"Fetching product data for barcode: {barcode}")
    url = f"https://world.openfoodfacts.org/api/v2/product/{barcode}.json"
    with httpx.Client() as client:
        response = client.get(url)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == 0:
            raise ValueError(f"Product not found for barcode: {barcode}")
        return data


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_structured_summary_from_llm(
    client: Anthropic, raw_product_data: dict
) -> Result[ProductSummary]:
    """
    Asks the LLM to summarize product data and extracts a structured object.
    Retries if the extraction fails.
    """
    print("\n--- Attempting to get and extract summary from LLM ---")
    # Step 1: Use an LLM to extract and structure the data
    print("Asking LLM to summarize product data...")
    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""
                Here is a large JSON object with product data from the Open Food Facts API.
                Please analyze it and extract a concise summary that conforms to this Pydantic model:
                {ProductSummary.model_json_schema()}

                Here is the product data:
                {raw_product_data}
                """,
            }
        ],
    )
    llm_response_text = message.content[0].text

    # Step 2: Use llmstruct to parse the LLM's output
    print("Attempting to extract JSON from LLM response:")
    print(f"'{llm_response_text}'")
    result = extract_json_from_text(llm_response_text, ProductSummary)

    if result.status == ExtractionStatus.SUCCESS:
        print("Successfully extracted object from LLM response.")
        return result

    print("Failed to extract valid object from LLM response.")
    raise ValueError("Extraction failed, will retry...")


def main():
    """
    An advanced demo that fetches real-world data, uses an LLM to process it,
    and retries with tenacity if the extraction fails.
    """
    print("--- Advanced Demo with Retries ---")
    client = Anthropic()
    product_barcode = "3017620422003"  # Nutella

    try:
        raw_product_data = get_product_from_api(product_barcode)
    except httpx.HTTPStatusError as e:
        print(f"HTTP error fetching product data: {e}")
        return
    except (ValueError, KeyError) as e:
        print(f"Error processing product data: {e}")
        return

    try:
        final_result = get_structured_summary_from_llm(client, raw_product_data)
        print(f"\nFinal extraction status: {final_result.status.value}")
        if not final_result.parsed_objects:
            print("No objects were parsed after all attempts.")
            return

        print("Parsed objects:")
        for obj in final_result.parsed_objects:
            print(f"- {obj.model_dump_json(indent=2)}")

    except Exception as e:
        print(f"\nExtraction process failed after multiple retries: {e}")

    print("-" * 20)


if __name__ == "__main__":
    main()

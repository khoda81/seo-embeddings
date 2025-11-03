import json
import logging

from openai import OpenAI

logger = logging.getLogger(__name__)


def generate_similar_terms(term: str, model: str, openai_client: OpenAI) -> list[str]:
    prompt = f"""
    You are an SEO keyword assistant specialized in Persian search queries.
    Given the query "{term}", generate 5 to 10 semantically related search keywords
    that focus on the same *topic* ({term.split()[-1]}), not on other products.
    Include a mix of informational and commercial intents, but all must clearly
    belong to the same topical field.
    Return the result as a compact JSON array of strings, with no explanations.
    """

    completion = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        # temperature=0.3,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "topical_keywords_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "topical_terms": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["topical_terms"],
                    "additionalProperties": False,
                },
            },
        },
    )

    # Parse and print
    resp = completion.choices[0].message.content.strip()

    try:
        keywords = json.loads(resp)["topical_terms"]

    except json.JSONDecodeError:
        # fallback if model adds text
        keywords = [k.strip() for k in resp.split("\n") if k.strip()]
        logger.warning(
            "Failed to parse JSON response, falling back to line split: %s", resp
        )

    return keywords

import asyncio
import json
import os

import aiohttp
import dotenv
import tqdm.asyncio
from label_studio_sdk import LabelStudio
from label_studio_sdk.label_interface.objects import PredictionValue

# -----------------------------
# ENVIRONMENT SETUP
# -----------------------------
dotenv.load_dotenv()

LABEL_STUDIO_URL = "http://185.8.172.121:8080/"
LABEL_STUDIO_API_KEY = os.environ["LABELSTUDIO_TOKEN"]
OPENROUTER_KEY = os.environ["OPENROUTER_KEY_ANNOTATION"]
MODEL = "openai/gpt-4o"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
PROJECT_ID = 1

# -----------------------------
# CONNECT TO LABEL STUDIO
# -----------------------------
client = LabelStudio(
    base_url=LABEL_STUDIO_URL,
    api_key=LABEL_STUDIO_API_KEY,
)
project = client.projects.get(id=PROJECT_ID)
li = project.get_label_interface()

print("Connected to Label Studio project:", project.title)


# -----------------------------
# PROMPT BUILDER
# -----------------------------
def build_prompt(query, options):
    opts = "\n".join(f"- {o['value']}" for o in options)
    return f"""You are a Persian SEO expert.
Given the query: «{query}»
and the following list of candidate keywords:

{opts}

Select only the items that are topically or semantically relevant
to the same intent as the query.
Return them exactly as an array of strings in JSON under the key 'relevant'.
Do not include unrelated, misspelled, or foreign words."""


# -----------------------------
# OPENROUTER REQUEST FUNCTION
# -----------------------------
async def get_prediction(session, task, sem: asyncio.Semaphore):
    """Send a prompt to OpenRouter and return relevant terms"""
    query = task.data.get("query")
    options = task.data.get("options")

    if not query or not options:
        print("Skipping invalid task:", task.id)
        return

    prompt = build_prompt(query, options)

    async with sem:
        for _ in range(3):  # retry
            try:
                async with session.post(
                    API_URL,
                    headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
                    json={
                        "model": MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "topical_keywords_response",
                                "strict": True,
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "relevant": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        }
                                    },
                                    "required": ["relevant"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "temperature": 0.2,
                    },
                    timeout=aiohttp.ClientTimeout(total=90),
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"HTTP {resp.status}: {text}")

                    completion = await resp.json()
                    msg = completion["choices"][0]["message"]["content"]
                    relevant = json.loads(msg)["relevant"]

                    return {"task": task, "prediction": relevant}

            except Exception as e:
                print(f"Retrying task {task.id}: {e}")
                await asyncio.sleep(3)

        print(f"Failed after retries: {task.id}")


# -----------------------------
# MAIN PIPELINE
# -----------------------------
async def main():
    # filter unannotated tasks
    task_filter = {
        "filters": {
            "conjunction": "and",
            "items": [
                {
                    "filter": "filter:tasks:total_annotations",
                    "operator": "equal",
                    "value": 0,
                    "type": "Number",
                },
                {
                    "filter": "filter:tasks:cancelled_annotations",
                    "operator": "equal",
                    "value": 0,
                    "type": "Number",
                },
            ],
        }
    }

    print("Fetching all tasks")
    tasks = list(client.tasks.list(project=PROJECT_ID, query=json.dumps(task_filter)))
    print(f"Found {len(tasks)} unannotated tasks")

    sem = asyncio.Semaphore(16)  # concurrency control
    async with aiohttp.ClientSession() as session:
        # Iterate asynchronously
        for coro in tqdm.asyncio.tqdm.as_completed(
            [get_prediction(session, t, sem) for t in tasks],
            total=len(tasks),
            unit="tasks",
        ):
            try:
                result = await coro

                task = result["task"]
                new_predictions = result["prediction"]

                score = None

                if task.predictions:
                    prev_predictions = set(
                        task.predictions[-1].result[0]["value"]["choices"]
                    )

                    common = prev_predictions.intersection(new_predictions)
                    union = prev_predictions.union(new_predictions)

                    score = len(common) / len(union)

                # Build Label Studio prediction
                predicted_label = li.get_control("rel").label(new_predictions)
                prediction = PredictionValue(
                    model_version=MODEL,
                    score=score,
                    result=[predicted_label],
                )

                # Attach prediction to the corresponding task
                client.predictions.create(task=task.id, **prediction.model_dump())

            except Exception as e:
                print("❌ Error:", e)


if __name__ == "__main__":
    asyncio.run(main())

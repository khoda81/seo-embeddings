import asyncio
import os

import aiohttp
import dotenv
import tqdm.asyncio
from googletrans import Translator
from label_studio_sdk import AsyncLabelStudio, LabelStudio, RoleBasedTask

# import tqdm.auto as tqdm

dotenv.load_dotenv()


# === CONFIG ===
LABEL_STUDIO_URL = "http://185.8.172.121:8080/"
# API key is available at the Account & Settings page in Label Studio UI
LABEL_STUDIO_API_KEY = os.environ["LABELSTUDIO_TOKEN"]

MAX_WORKERS = 4  # you can safely increase this to 64 or even 128 if API is fast


# Connect to the Label Studio API
als_client = AsyncLabelStudio(
    base_url=LABEL_STUDIO_URL,
    api_key=LABEL_STUDIO_API_KEY,
)

# Connect to the Label Studio API
ls_client = LabelStudio(
    base_url=LABEL_STUDIO_URL,
    api_key=LABEL_STUDIO_API_KEY,
)


project = ls_client.projects.get(id=1)
print(project)
tasks = ls_client.tasks.list(project=project.id)
annotations = []

# A basic request to verify connection is working
me = ls_client.users.whoami()

print("username:", me.username)
print("email:", me.email)

translator = Translator()


async def enrich_translations(task: RoleBasedTask):
    translations = await translator.translate(
        [entry["value"] for entry in task.data["options"]],
        src="auto",
        dest="fa",
    )

    for entry, translation in zip(task.data["options"], translations):
        entry["hint"] = translation.text

    return task


async def translate_task(
    task: RoleBasedTask,
    sem: asyncio.Semaphore,
):
    annotations.extend(task.annotations)
    if task.annotations or all("hint" in cand for cand in task.data["options"]):
        return

    while True:
        async with sem:
            try:
                task = await enrich_translations(task)
                break

            except Exception as e:
                print(f"Translating {task.id}: {e}")
                await asyncio.sleep(3)

    while True:
        try:
            return await als_client.tasks.update(
                task.id, project=project.id, data=task.data
            )

        except Exception as e:
            print(f"Retrying task update {task.id}: {e}")
            await asyncio.sleep(3)


tasks = list(ls_client.tasks.list(project=project.id))


sem = asyncio.Semaphore(MAX_WORKERS)  # concurrency control
async with aiohttp.ClientSession() as session:
    # Iterate asynchronously
    for coro in tqdm.asyncio.tqdm.as_completed(
        [translate_task(t, sem) for t in tasks], unit="tasks"
    ):
        try:
            result = await coro

        except Exception as e:
            print("❌ Error:", e)

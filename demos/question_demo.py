"""The same dashboard, interrupted by questions.

Watch the transition: the dashboard runs for three seconds, the question takes
over the screen, and the dashboard comes back with the answer in its footer. A
second question follows, so the path through the queue is exercised twice.

    uv run python demos/question_demo.py

Answer with a number, or type anything else to answer in your own words.
"""

import asyncio

from dashboard_demo import Dashboard

from agl.core.terminal import Answer, Option, Question
from agl.core.terminal.impl.rich_terminal import RichTerminal

STORAGE_QUESTION = Question(
    header="T-04  login screen",
    title="Which storage layer should the token cache use?",
    options=(
        Option("DataStore", "Survives process death, async API"),
        Option("In-memory only", "Simplest, lost on restart"),
        Option("EncryptedSharedPreferences", "Synchronous, encrypted at rest"),
    ),
)

RETRY_QUESTION = Question(
    header="T-05  auth interceptor",
    title="What should the interceptor do when a refresh fails?",
    options=(
        Option("Fail the request", "Surface the error to the caller"),
        Option("Queue and retry once", "Hold requests until the refresh settles"),
    ),
)


def describe(answer: Answer) -> str:
    return f"{answer.text}{' (your own words)' if answer.was_free_text else ''}"


async def main() -> None:
    dashboard = Dashboard()
    async with RichTerminal().live(dashboard.build, fps=8) as session:
        await asyncio.sleep(3)

        storage = await session.ask(STORAGE_QUESTION)
        dashboard.note = f"token cache → {describe(storage)}"
        await asyncio.sleep(4)

        retry = await session.ask(RETRY_QUESTION)
        dashboard.note = f"refresh failure → {describe(retry)}"
        await asyncio.sleep(4)

    print(f"storage: {storage!r}")
    print(f"retry:   {retry!r}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("stopped")

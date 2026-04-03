"""CrewAI + LLMock: test multi-agent resilience without tokens.

Demonstrates how to run CrewAI agents against LLMock, letting you
validate agent orchestration, retry behavior, and task delegation
without spending real API budget.

Usage:
    # Terminal 1: start LLMock with some chaos
    llmock serve --error-rate 429=0.2 --response-style varied

    # Terminal 2: run this script
    python examples/crewai_resilient_agents.py

Requirements:
    pip install crewai crewai-tools
"""

import os

from crewai import Agent, Task, Crew
from crewai import LLM

LLMOCK_URL = os.getenv("LLMOCK_BASE_URL", "http://127.0.0.1:8000/v1")


def create_llm() -> LLM:
    """Create a CrewAI-compatible LLM pointed at LLMock."""
    return LLM(
        model="openai/gpt-4o",
        base_url=LLMOCK_URL,
        api_key="mock-key",
    )


# --- Example: Research & Writing Crew ---

def run_research_crew() -> str:
    """Run a simple two-agent crew against LLMock."""
    llm = create_llm()

    researcher = Agent(
        role="Senior Researcher",
        goal="Find key facts about the given topic",
        backstory="You are an experienced researcher who finds concise, relevant facts.",
        llm=llm,
        verbose=True,
    )

    writer = Agent(
        role="Technical Writer",
        goal="Write a clear, concise summary from research findings",
        backstory="You are a skilled writer who turns complex research into readable summaries.",
        llm=llm,
        verbose=True,
    )

    research_task = Task(
        description="Research the topic: 'Why chaos engineering matters for LLM applications'",
        expected_output="A list of 3-5 key facts about chaos engineering for LLM apps.",
        agent=researcher,
    )

    writing_task = Task(
        description="Write a 2-paragraph summary based on the research findings.",
        expected_output="A clear, concise 2-paragraph summary.",
        agent=writer,
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        verbose=True,
    )

    result = crew.kickoff()
    return str(result)


def main() -> None:
    print("=" * 60)
    print("CrewAI + LLMock Integration Demo")
    print("=" * 60)
    print()
    print("Running a 2-agent crew (Researcher + Writer) against LLMock...")
    print("All LLM calls go to the local mock server — zero tokens spent.")
    print()

    result = run_research_crew()

    print()
    print("=" * 60)
    print("Final Output:")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()

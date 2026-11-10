# Wedding planner

A LangChain agent that plans a wedding by calling three subagents:

- Travel, via the Kiwi MCP server
- Venues, via the Wedstay MCP server
- Music, via YouTube Music MCP and Tavily web search

## Setup

Requires [uv](https://docs.astral.sh/uv/), Python 3.12 or 3.13, and Node.js (`npx` launches the venue and music servers).

```bash
uv sync
cp .env.example .env
```

Put your OpenAI, Tavily, and YouTube API keys in `.env`.

## Run

```bash
uv run python wedding_planner.py
uv run python wedding_planner.py "Plan a wedding in Paris on 12 June 2027. The couple likes jazz."
```

With no arguments, the script plans a rock wedding in New York on 5 July 2027.

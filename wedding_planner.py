import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from langchain.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage
from tavily import TavilyClient
from typing import Dict, Any
from pprint import pprint

tavily_client = TavilyClient()

travel_client = MultiServerMCPClient(
    {"travel_server": {"transport": "streamable_http", "url": "https://mcp.kiwi.com"}}
)

travel_agent = create_agent(
    "gpt-5-nano",
    tools=asyncio.run(travel_client.get_tools()),
    checkpointer=InMemorySaver(),
    system_prompt="You are a travel agent. No follow up questions.",
)

venue_client = MultiServerMCPClient(
    {
        "wedstay": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "mcp-remote", "https://mcp.wedstay.com/api/ai/mcp"],
        }
    }
)

venue_agent = create_agent(
    "gpt-5-nano",
    tools=asyncio.run(venue_client.get_tools()),
    checkpointer=InMemorySaver(),
    system_prompt="You are AI assistants direct access to wedding venue search, real pricing, vendor discovery, and booking.",
)

youtube_music_client = MultiServerMCPClient(
    {
        "youtube-music-mcp": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@instructa/mcp-youtube-music"],
            "env": {"YOUTUBE_API_KEY": os.environ["YOUTUBE_API_KEY"]},
        }
    }
)

youtube_tools = asyncio.run(youtube_music_client.get_tools())


@tool
def search_web(query: str) -> Dict[str, Any]:
    """Search the web for information"""

    results = tavily_client.search(query)

    return results


music_agent = create_agent(
    "gpt-5-nano",
    tools=[search_web, *youtube_tools],
    system_prompt="You are music planner for the wedding. You can use youtube music resources to create the playlist and use web search to find contacts of the bands and availability.",
)


@tool
async def call_travel_agent(destination: str) -> str:
    """Call travel agent to plan the wedding trip"""
    response = await travel_agent.ainvoke(
        {"messages": [HumanMessage(content=f"Plan the wedding trip to {destination}")]}
    )
    return response["messages"][-1].content


@tool
async def call_venue_agent(city: str, date: str) -> str:
    """Create list of avaliable vendors available on the wedding day"""
    response = await venue_agent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=f"Find available venues for the wedding in {city} on {date}"
                )
            ]
        }
    )
    return response["messages"][-1].content


@tool
async def call_music_agent(genre: str) -> str:
    """Make the playlist for the wedding with contacts of the musicians"""
    response = await music_agent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=f"Generate the playlist in {genre} genre for the wedding and find contacts of the musicians (original or cover-makers)."
                )
            ]
        }
    )
    return response["messages"][-1].content


wedding_planner = create_agent(
    model="gpt-5-nano",
    tools=[call_travel_agent, call_venue_agent, call_music_agent],
    system_prompt="You are a wedding planner who can call subagents as your assistans to plan the wedding trip, fnd the venue or plan the music. No follow up question - plan the event",
)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = "Plan the wedding on the 5th of July 2027 in New York. Both bride and groom love rock music."

    response = asyncio.run(
        wedding_planner.ainvoke({"messages": [HumanMessage(content=question)]})
    )

    pprint(response)
    pprint(response["messages"][-1].content)

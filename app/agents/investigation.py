"""
Investigation phase — assembles the five specialist agents into a ParallelAgent
so they run concurrently and write their outputs to session state independently.
"""

from google.adk.agents import ParallelAgent

from app.agents.financial import create_financial_agent
from app.agents.legal import create_legal_agent
from app.agents.market import create_market_agent
from app.agents.news_sentiment import create_news_sentiment_agent
from app.agents.people_culture import create_people_culture_agent

investigation_phase = ParallelAgent(
    name="investigation_phase",
    sub_agents=[
        create_financial_agent(),
        create_legal_agent(),
        create_market_agent(),
        create_news_sentiment_agent(),
        create_people_culture_agent(),
    ],
)

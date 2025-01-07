from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai

from dotenv import load_dotenv
load_dotenv()


## Web Search Agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include the sources"],
    show_tool_calls=True,
    markdown=True
)


## Financial Agent
financial_agent = Agent(
    name="Financial Agent",
    role="Get financial data",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True
)


multi_ai_agent = Agent(
    team=[web_search_agent, financial_agent],
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions=["Always include the sources", "Use tables to display the data"],
    show_tool_calls=True,
    markdown=True
)

multi_ai_agent.print_response("Summarize analyst recommendation and share the latest news for NVDA", stream=True)

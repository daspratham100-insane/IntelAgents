from langchain.tools import tool
from tavily import TavilyClient
from bs4 import BeautifulSoup
import requests
import os
from dotenv import load_dotenv

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def web_search(query: str) -> str:
    """
    Search the web and return the top URLs with titles and snippets.
    """

    try:
        results = tavily.search(
            query=query,
            max_results=5,
            search_depth="advanced"
        )

        if not results.get("results"):
            return "No search results found."

        output = []

        for i, r in enumerate(results["results"], start=1):
            title = r.get("title", "No Title")
            url = r.get("url", "")
            snippet = r.get("content", "")

            output.append(
                f"""
Result {i}
Title : {title}
URL   : {url}
Snippet :
{snippet}

{'-'*80}
"""
            )

        return "\n".join(output)

    except Exception as e:
        return f"Search Error: {str(e)}"


@tool
def scrape_url(url: str) -> str:
    """
    Scrape readable text from a webpage.
    """

    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=15
        )

        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "noscript"
        ]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

        return text[:6000]

    except Exception as e:
        return f"Scraping Error: {str(e)}"






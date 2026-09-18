import re

from agents import writer_chain, critic_chain
from tools import web_search, scrape_url


def extract_urls(search_results: str):
    """
    Extract all URLs from Tavily search results.
    """
    urls = re.findall(r"URL\s*:\s*(https?://[^\s]+)", search_results)

    # Remove duplicates while preserving order
    unique_urls = list(dict.fromkeys(urls))

    return unique_urls


def run_research_pipeline(topic: str):

    state = {}

    # --------------------------------------------------
    # STEP 1 : SEARCH
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 1 : Searching the Web...")
    print("=" * 70)

    search_results = web_search.invoke(topic)

    state["search_results"] = search_results

    print(search_results)

    # --------------------------------------------------
    # STEP 2 : EXTRACT URLS
    # --------------------------------------------------

    urls = extract_urls(search_results)

    state["urls"] = urls

    if not urls:
        print("\nNo URLs found.")
        return

    print("\nURLs Found:\n")

    for i, url in enumerate(urls, 1):
        print(f"{i}. {url}")

    # --------------------------------------------------
    # STEP 3 : SCRAPE ALL URLS
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 2 : Scraping Websites...")
    print("=" * 70)

    scraped_text = ""

    for i, url in enumerate(urls, start=1):

        print(f"\nScraping ({i}/{len(urls)})")
        print(url)

        page = scrape_url.invoke(url)

        scraped_text += f"""

=========================================================
SOURCE URL:
{url}

CONTENT:
{page}

"""

    state["scraped_content"] = scraped_text

    # --------------------------------------------------
    # STEP 4 : WRITER
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 3 : Writing Research Report...")
    print("=" * 70)

    research = f"""

SEARCH RESULTS

{search_results}


SCRAPED CONTENT

{scraped_text}

"""

    report = writer_chain.invoke({

        "topic": topic,

        "research": research

    })

    state["report"] = report

    print(report)

    # --------------------------------------------------
    # STEP 5 : CRITIC
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 4 : Reviewing Report...")
    print("=" * 70)

    feedback = critic_chain.invoke({

        "report": report

    })

    state["feedback"] = feedback

    print(feedback)

    # --------------------------------------------------
    # SOURCES
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("SOURCES USED")
    print("=" * 70)

    for url in urls:
        print(url)

    return state


if __name__ == "__main__":

    topic = input("\nEnter a research topic: ")

    run_research_pipeline(topic)
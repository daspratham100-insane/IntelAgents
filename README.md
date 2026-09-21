# IntelAgents
# Multi-Agent AI Research System

An autonomous multi-agent research assistant built with LangChain, OpenAI, Tavily, BeautifulSoup, and Streamlit.

The system searches the web, reads relevant sources, generates a research report, and evaluates the report using a critic chain.

---

## Features

- Real-time web search using Tavily
- URL content extraction using BeautifulSoup
- Specialized Search Agent
- Specialized Reader Agent
- Automated research report generation
- Report evaluation using a Critic Chain
- Shared state for passing information between components
- Streamlit-based user interface
- Environment-variable-based API key management
- Modular and extensible project architecture
- Support for changing the underlying language model

---

## System Architecture

The application follows a modular multi-agent architecture. Each component has a specific responsibility in the research workflow.

```text
                         ┌──────────────────────┐
                         │      User Topic      │
                         └──────────┬───────────┘
                                    │
                                    v
                         ┌──────────────────────┐
                         │   Streamlit UI       │
                         │       app.py         │
                         └──────────┬───────────┘
                                    │
                                    v
                         ┌──────────────────────┐
                         │   Research Pipeline  │
                         │    pipeline.py       │
                         └──────────┬───────────┘
                                    │
                   ┌────────────────┴────────────────┐
                   │                                 │
                   v                                 v
        ┌──────────────────────┐          ┌──────────────────────┐
        │    Search Agent      │          │    Reader Agent      │
        │  Finds relevant URLs │          │ Extracts page text  │
        └──────────┬───────────┘          └──────────┬───────────┘
                   │                                 │
                   v                                 v
        ┌──────────────────────┐          ┌──────────────────────┐
        │   Tavily Web Search  │          │ BeautifulSoup Parser │
        └──────────┬───────────┘          └──────────┬───────────┘
                   │                                 │
                   └────────────────┬────────────────┘
                                    v
                         ┌──────────────────────┐
                         │   Shared Research   │
                         │        State        │
                         └──────────┬───────────┘
                                    │
                                    v
                         ┌──────────────────────┐
                         │    Writer Chain      │
                         │  Creates the report  │
                         └──────────┬───────────┘
                                    │
                                    v
                         ┌──────────────────────┐
                         │    Critic Chain      │
                         │ Reviews the report  │
                         └──────────┬───────────┘
                                    │
                                    v
                         ┌──────────────────────┐
                         │   Final Research    │
                         │       Report        │
                         └──────────────────────┘

## Project Structure 
multi-agent-system/
│
├── .env
├── .gitignore
├── agents.py
├── app.py
├── pipeline.py
├── requirements.txt
├── tools.py
└── README.md
## File Responsibilities
app.py ---	Streamlit user interface
pipeline.py--	Main orchestration and research workflow
agents.py--	Search Agent, Reader Agent, Writer Chain, and Critic Chain
tools.py--	Tavily search and URL-scraping tools
requirements.txt--	Python dependencies
.env--	API keys and environment variables
.gitignore--	Files excluded from Git

## Technology Stack

Python — Core programming language
LangChain — Agent and language-model orchestration
OpenAI — Large language model provider
Tavily — Real-time web search
BeautifulSoup — HTML parsing and text extraction
Requests — HTTP requests
Streamlit — User interface
python-dotenv — Environment-variable management


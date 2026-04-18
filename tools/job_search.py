import sys
import os
import json
import re
from tavily import TavilyClient
from typing import Annotated
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Optional
from openai import AzureOpenAI

# Load variables from .env into the environment
load_dotenv()
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Initialize Tavily and Azure Open API clients
tavily_client= TavilyClient(api_key=tavily_api_key)

@dataclass
class JobListing:
    title: str
    company: str
    location: str
    url: str
    description_snippet: str
    skills_mentioned: list[str] 


def search_jobs(role: str, location: str, max_results: int = 5, job_boards: Optional[list[str]] = None) -> list[JobListing]:
    
    if job_boards:
        # If caller specified boards, include them as hints 
        boards_hint = " OR ".join(f"site:{b}" for b in job_boards)
        query = f'"{role}" job {location} ({boards_hint})'
    else:
        # Generic query that works globally - "job listing" signals to Tavily
        # to prioritise job board pages over blog posts or salary articles
        query = f'"{role}" job listing {location}'
    
    response = tavily_client.search(
        query=query,
        search_depth="advanced", 
        max_results=max_results, 
        include_answer=False, 
        include_raw_content=False)
    
    listings = []
    for result in response.get("results", []):
        snippet = result.get("content", "")
        title = result.get("title", "")

        listing = JobListing(
            title=_clean_job_title(title, role),
            company=_extract_company(result.get("title", "")),
            location=location,
            url=result.get("url", ""),
            description_snippet=snippet[:800],   # cap at 800 chars to save tokens
            skills_mentioned=_extract_skills_with_llm(snippet, role),
        )
        listings.append(listing)
 
    return listings

def _clean_job_title(page_title: str, searched_role: str) -> str:
   # Strip the job board name from the end
    for suffix in [" | LinkedIn", " | Indeed", " | Glassdoor", " | SEEK",
                   " - LinkedIn", " - Indeed", " - Glassdoor", " - SEEK",
                   " | Jobs", " | Careers"]:
        if page_title.endswith(suffix):
            page_title = page_title[:-len(suffix)]
 
    # Split on common separators — the role is usually the first part
    for sep in [" - ", " | ", " at ", " @ ", " — "]:
        if sep in page_title:
            return page_title.split(sep)[0].strip()
 
    return page_title.strip() or searched_role

 
def _extract_company(title: str) -> str:
    """
    Attempts to extract the company name from the page title.
    Common patterns: "Role - Company | Board" or "Role at Company"
    """
    # Remove trailing job board names first
    for suffix in [" | LinkedIn", " | Indeed", " | Glassdoor", " | SEEK",
                   " - LinkedIn", " - Indeed", " - Glassdoor", " - SEEK"]:
        if title.endswith(suffix):
            title = title[:-len(suffix)]
 
    for separator in [" - ", " | ", " at ", " @ ", " — "]:
        if separator in title:
            parts = title.split(separator)
            if len(parts) >= 2:
                return parts[1].strip()
 
    return "Unknown Company"

#Use LLM GPT-4o-mini to identify the skills for the job title given for gap analyser
def _extract_skills_with_llm(description: str, role: str) -> list[str]:
    if not description.strip():
        return []
 
    try:
        azure_client = AzureOpenAI(
            azure_endpoint=os.getenv["AZURE_OPENAI_ENDPOINT"],
            api_key=os.getenv["AZURE_OPENAI_API_KEY"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )
 
        response = azure_client.chat.completions.create(
            model=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract required skills, technologies, tools, and qualifications "
                        "from this job description. Return ONLY a JSON array of short skill "
                        "strings. No explanations. Example: "
                        '[\"Python\", \"SQL\", \"5 years experience\", \"AWS\"]'
                    ),
                },
                {
                    "role": "user",
                    "content": f"Job role: {role}\n\nDescription:\n{description[:600]}",
                },
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=200,
        )
 
        raw = response.choices[0].message.content
        parsed = json.loads(raw)
 
        # The model might return {"skills": [...]} or just [...] wrapped in an object
        if isinstance(parsed, dict):
            skills = parsed.get("skills", parsed.get("required_skills", list(parsed.values())[0] if parsed else []))
        else:
            skills = parsed
 
        # Sanitise: keep only strings, strip whitespace, lowercase for consistency
        return sorted({str(s).strip() for s in skills if isinstance(s, str) and s.strip()})
 
    except Exception:
        # Fallback: simple word-boundary matching for common patterns
        # This intentionally catches all exceptions — skill extraction is non-critical
        return _fallback_skill_extraction(description)

#Simple fallback when LLM is unavailable.
def _fallback_skill_extraction(text: str) -> list[str]:

    # Match things in brackets, after "experience with", "proficiency in", etc.
    patterns = [
        r'\b[A-Z][a-zA-Z]+(?:\.[a-zA-Z]+)+\b',   # e.g. "Node.js", "ASP.NET"
        r'\b[A-Z]{2,}\b',                           # e.g. "SQL", "AWS", "API"
    ]
    found = set()
    for pattern in patterns:
        found.update(re.findall(pattern, text))
 
    # Remove common non-skill uppercase words
    noise = {"THE", "AND", "FOR", "WITH", "YOU", "OUR", "ARE", "THIS",
             "WILL", "HAVE", "FROM", "THAT", "THEY", "BEEN", "YOUR"}
    return sorted(found - noise)[:15]   # cap at 15 to avoid noise


if __name__ == "__main__":
    #Example: # python job_search.py "Data Scientist" "London"
    test_role = sys.argv[1] if len(sys.argv) > 1 else "AI Engineer"
    test_location = sys.argv[2] if len(sys.argv) > 2 else "Melbourne"
 
    print(f"\nSearching for '{test_role}' jobs in '{test_location}'...")
    results = search_jobs(test_role, test_location, max_results=3)
 
    for r in results:
        print(f"\n{'='*60}")
        print(f"Title    : {r.title}")
        print(f"Company  : {r.company}")
        print(f"URL      : {r.url}")
        print(f"Skills   : {r.skills_mentioned}")
        print(f"Snippet  : {r.description_snippet[:200]}...")



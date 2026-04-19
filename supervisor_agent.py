import os
import asyncio
from dataclasses import dataclass
from dotenv import load_dotenv

from agent_framework import Agent
from agent_framework_openai import OpenAIChatClient
from mcp.client.stdio import StdioServerParameters

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent))
from tools.job_search import search_jobs, JobListing
from tools.parse_resume import parse_resume, ParsedResume
from tools.gap_analyser import analyse_multiple_jobs, GapAnalysis
from tools.cover_letter_writter import generate_cover_letter, CoverLetterResult

load_dotenv()

@dataclass
class JobResearchResult:
    candidate_name: str
    role_searched: str
    location: str
    jobs_found: int
    analyses: list[dict]            # gap analysis per job, serialisable
    top_match: dict | None          # best job analysis
    cover_letter: str | None        # for the top match
    cover_letter_safety_passed: bool
    summary: str                    # human-readable summary from the LLM


# Creates the Azure OpenAI client for the supervisor agent's reasoning.
def build_chat_client() -> AzureOpenAIChatClient:

    return AzureOpenAIChatClient(
        model_id=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini"),
        endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    )


# Main orchestration function
async def run_job_research_pipeline(
    resume_pdf_path: str,
    role: str,
    location: str,
    max_jobs: int = 5,
    min_match_score: int = 40,
    generate_letter: bool = True,
    tone: str = "professional",
) -> JobResearchResult:

    # 1. Parse the resume
    print(f"\n[Supervisor] Step 1: Parsing resume from {resume_pdf_path}")
    resume = parse_resume(resume_pdf_path)
    print(f"[Supervisor] Resume parsed: {resume.name}, {len(resume.skills)} skills")

    # 2. Search for jobs
    print(f"\n[Supervisor] Step 2: Searching for '{role}' jobs in {location}")
    jobs = search_jobs(role=role, location=location, max_results=max_jobs)
    print(f"[Supervisor] Found {len(jobs)} listings")

    if not jobs:
        return JobResearchResult(
            candidate_name=resume.name, role_searched=role, location=location,
            jobs_found=0, analyses=[], top_match=None, cover_letter=None,
            cover_letter_safety_passed=False,
            summary=f"No jobs found for '{role}' in {location}. Try a broader search."
        )

    # 3. Gap analysis for all jobs
    print(f"\n[Supervisor] Step 3: Running gap analysis on {len(jobs)} jobs...")
    analyses = analyse_multiple_jobs(
        resume=resume,
        jobs=jobs,
        min_score=min_match_score,
    )
    print(f"[Supervisor] {len(analyses)} jobs passed minimum score threshold ({min_match_score}+)")

    if not analyses:
        return JobResearchResult(
            candidate_name=resume.name, role_searched=role, location=location,
            jobs_found=len(jobs), analyses=[], top_match=None, cover_letter=None,
            cover_letter_safety_passed=False,
            summary=f"Found {len(jobs)} jobs but none scored above {min_match_score}. Consider broadening your search or upskilling."
        )

    # 4. Generate cover letter for top match 
    top_analysis = analyses[0]              # already sorted best-first
    # Find the matching JobListing for the top analysis
    top_job = next(
        (j for j in jobs if j.company == top_analysis.company),
        jobs[0]                             # fallback to first job if not found
    )

    cover_letter_text = None
    safety_passed = False

    if generate_letter and top_analysis.match_score >= 50:
        print(f"\n[Supervisor] Step 4: Generating cover letter for {top_analysis.job_title} at {top_analysis.company} (score: {top_analysis.match_score})")
        letter_result = generate_cover_letter(
            resume=resume,
            job=top_job,
            gap=top_analysis,
            tone=tone,
        )
        cover_letter_text = letter_result
        #safety_passed = letter_result.safety_passed
    else:
        print(f"\n[Supervisor] Step 4: Skipping cover letter (score {top_analysis.match_score} < 50 or disabled)")

    # 5. Generate a human-readable summary via the agent
    print(f"\n[Supervisor] Step 5: Generating summary...")
    summary = generate_summary(resume, analyses, top_analysis)

    analyses_dicts = []
    for a in analyses:
        analyses_dicts.append({
            "job_title": a.job_title,
            "company": a.company,
            "match_score": a.match_score,
            "apply_recommendation": a.apply_recommendation,
            "matched_skills": [s.skill for s in a.matched_skills],
            "missing_skills": [s.skill for s in a.missing_skills],
            "strengths": a.strengths,
            "quick_wins": a.quick_wins,
            "resume_tweaks": a.recommended_resume_tweaks,
        })

    return JobResearchResult(
        candidate_name=resume.name,
        role_searched=role,
        location=location,
        jobs_found=len(jobs),
        analyses=analyses_dicts,
        top_match=analyses_dicts[0] if analyses_dicts else None,
        cover_letter=cover_letter_text,
        cover_letter_safety_passed=safety_passed,
        summary=summary,
    )


# Helper: generate a readable summary
def generate_summary(
    resume: ParsedResume,
    analyses: list[GapAnalysis],
    top: GapAnalysis,
) -> str:

    from openai import AzureOpenAI
    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    )

    scores = [f"{a.job_title} at {a.company}: {a.match_score}/100" for a in analyses[:3]]
    prompt = f"""Summarise these job search results in 3-4 sentences for {resume.name}:
        Top matches: {', '.join(scores)}
        Best match: {top.job_title} at {top.company} ({top.match_score}/100)
        Recommendation: {top.apply_recommendation}
        Key missing skills: {[s.skill for s in top.missing_skills[:3]]}
        Be specific, encouraging, and actionable."""

    response = client.chat.completions.create(
        model=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


# python supervisor_agent.py resume.pdf "Data Scientist" "London"
if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else "resume.pdf"
    role = sys.argv[2] if len(sys.argv) > 2 else "Software Engineer"
    location = sys.argv[3] if len(sys.argv) > 3 else "Remote"
    max_jobs = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    min_score = int(sys.argv[5]) if len(sys.argv) > 5 else 30

    result = asyncio.run(run_job_research_pipeline(
        resume_pdf_path=pdf,
        role=role,
        location=location,
        max_jobs=max_jobs,
        min_match_score=min_score,
    ))

    print(f"\n{'='*60}")
    print(f"Candidate   : {result.candidate_name}")
    print(f"Searching   : {role} in {location}")
    print(f"Jobs found  : {result.jobs_found}")
    print(f"Top match   : {result.top_match['job_title'] if result.top_match else 'None'}")
    print(f"Match score : {result.top_match['match_score'] if result.top_match else 'N/A'}")
    print(f"\nSummary:\n{result.summary}")
    if result.cover_letter:
        print(f"\nCover Letter:\n{result.cover_letter[:300]}...")
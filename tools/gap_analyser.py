# Compares your parsed resume against one or more job listings and produces
# a structured gap analysis: what you have, what's missing, what partially 
# matches, and a match score.

import os
import json
from typing import Optional
from pydantic import BaseModel, Field    # for structured output validation
from openai import AzureOpenAI
from dotenv import load_dotenv
 
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent))
from job_search import JobListing
 
import sys
from parse_resume import parse_resume, ParsedResume
from job_search import JobListing

load_dotenv()

class SkillMatch(BaseModel):
    skill: str
    in_resume: bool
    resume_context: Optional[str] = Field(
        None, description="Where in resume this skill appears, e.g. 'MLflow in MLOps project'"
    )
 
class GapAnalysis(BaseModel):
    job_title: str
    company: str
    match_score: int = Field(
        ..., ge=0, le=100,
        description="0-100 overall match. 70+ = strong, 40-69 = partial, <40 = weak"
    )
    matched_skills: list[SkillMatch] = Field(
        description="Skills the job wants that are in the resume"
    )
    missing_skills: list[SkillMatch] = Field(
        description="Skills the job wants that are NOT in the resume"
    )
    partial_matches: list[SkillMatch] = Field(
        description="Skills where resume shows related but not exact experience"
    )
    strengths: list[str] = Field(
        description="2-3 specific strengths the candidate has for this role"
    )
    quick_wins: list[str] = Field(
        description="Things candidate could add/do in 1-2 weeks to improve match"
    )
    recommended_resume_tweaks: list[str] = Field(
        description="Specific bullet point changes to tailor resume for this role"
    )
    apply_recommendation: str = Field(
        description="One of: 'Strong match — apply now', 'Good match — tailor first', 'Weak match — skip or upskill'"
    )

# Main function the supervisor calls - Compares resume against a single job listing.
def analyse_gap(resume: ParsedResume, job: JobListing) -> GapAnalysis:
    """
    Flow:
    1. Format resume skills + experience as a compact string
    2. Format job description + required skills
    3. Ask GPT-4o-mini to compare them, enforcing the GapAnalysis schema
    4. Validate output with Pydantic (raises ValidationError if schema breaks)
    5. Return the validated GapAnalysis object
    """
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    )
 
    # Build a compact resume summary
    resume_summary = f"""
        CANDIDATE: {resume.name}
        SKILLS: {', '.join(resume.skills)}
        EXPERIENCE:
        {_format_experience(resume.experience)}
        CERTIFICATIONS: {', '.join(resume.certifications)}
        """
 
    job_summary = f"""
        ROLE: {job.title} at {job.company}
        LOCATION: {job.location}
        REQUIRED SKILLS MENTIONED: {', '.join(job.skills_mentioned)}
        DESCRIPTION:
        {job.description_snippet}
        """
 
    # The schema is embedded in the prompt so the model knows exactly what to produce.
    # We also pass it as response_format JSON schema for strict enforcement.
    system_prompt = f"""You are a senior technical recruiter and AI specialist.
        Compare the candidate's resume against the job listing and return a gap analysis.
        Be specific - reference actual items from the resume, not generic advice.
        Return ONLY valid JSON matching this exact schema:
        {GapAnalysis.model_json_schema()}"""
        
    response = client.chat.completions.create(
        model=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"RESUME:\n{resume_summary}\n\nJOB:\n{job_summary}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,    # slight creativity for recommendations, but mostly deterministic
        max_tokens=1500,
    )
 
    raw_json = json.loads(response.choices[0].message.content)
 
    # Pydantic validates the structure, if the LLM dropped a required field,
    # this raises a clear ValidationError instead of silently returning bad data
    return GapAnalysis(**raw_json)
 

def analyse_multiple_jobs(
    resume: ParsedResume,
    jobs: list[JobListing],
    min_score: int = 0
) -> list[GapAnalysis]:

    results = []
    for job in jobs:
        print(f"[GapAnalyser] Analysing: {job.title} at {job.company}...")
        try:
            analysis = analyse_gap(resume, job)
            if analysis.match_score >= min_score:
                results.append(analysis)
        except Exception as e:
            print(f"[GapAnalyser] Error for {job.title}: {e}")
            continue
 
    # Sort best matches first - supervisor uses top result for cover letter
    return sorted(results, key=lambda x: x.match_score, reverse=True)
 
 
def _format_experience(experience: list[dict]) -> str:
    lines = []
    for exp in experience[:4]:          # cap at 4 roles to stay within token budget
        lines.append(f"- {exp.get('role')} at {exp.get('company')} ({exp.get('duration')})")
        for bullet in exp.get("bullets", [])[:3]:   # 3 bullets per role
            lines.append(f"    • {bullet}")
    return "\n".join(lines)


    # Test the gap analyser in isolation using a real resume PDF + a fake job.
    # Example - python gap_analyser.py resume.pdf "Data Scientist" "London"
if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "resume.pdf"
    test_role = sys.argv[2] if len(sys.argv) > 2 else "Software Engineer"
    test_location = sys.argv[3] if len(sys.argv) > 3 else "Remote"
 
    print(f"Parsing resume from {pdf_path}...")
    try:
        resume = parse_resume(pdf_path)
    except FileNotFoundError:
        # If no real PDF available, create a minimal fake resume for testing
        print(f"[Warning] {pdf_path} not found — using placeholder resume data")
        resume = ParsedResume(
            name="Test Candidate",
            email="test@example.com",
            location=test_location,
            summary=f"Experienced professional applying for {test_role} roles.",
            skills=["Communication", "Problem Solving", "Python", "SQL"],
            experience=[{
                "role": "Senior Analyst",
                "company": "Example Corp",
                "duration": "2021-Present",
                "bullets": ["Led cross-functional projects", "Improved team efficiency by 30%"]
            }],
            education=[{"degree": "Bachelor of Science", "institution": "University", "year": "2019"}],
            certifications=[],
            raw_text=""
        )
 
    # Fake job listing - replace skills_mentioned with whatever is relevant for
    # the role you're testing. The LLM will do the actual comparison.
    fake_job = JobListing(
        title=test_role,
        company="Example Company",
        location=test_location,
        url="https://example.com/jobs/1",
        description_snippet=f"We are looking for a {test_role} to join our team in {test_location}. The ideal candidate will have strong analytical skills and experience with relevant tools for this domain.",
        skills_mentioned=["Python", "SQL", "Communication", "Project Management", "Data Analysis"]
    )
 
    result = analyse_gap(resume, fake_job)
    print(f"\nRole tested : {test_role} in {test_location}")
    print(f"Match Score : {result.match_score}/100")
    print(f"Verdict     : {result.apply_recommendation}")
    print(f"Strengths   : {result.strengths}")
    print(f"Missing     : {[s.skill for s in result.missing_skills]}")
    print(f"Quick wins  : {result.quick_wins}")
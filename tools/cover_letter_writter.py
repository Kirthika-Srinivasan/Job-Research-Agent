import os
import json
from dataclasses import dataclass
from openai import AzureOpenAI
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions, TextCategory
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from dotenv import load_dotenv
 
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent))
from job_search import JobListing
from gap_analyser import GapAnalysis
 
from parse_resume import parse_resume, ParsedResume
from job_search import JobListing, search_jobs
from gap_analyser import GapAnalysis, SkillMatch, analyse_gap

load_dotenv()

@dataclass
class CoverLetterResult:
    cover_letter: str               # the final letter text
    word_count: int
    safety_passed: bool             # passed Content Safety check
    safety_flags: list[str]         
    generation_notes: str           

# Generates a tailored cover letter using resume + job + gap analysis.
def generate_cover_letter(
    resume: ParsedResume,
    job: JobListing,
    gap: GapAnalysis,
    tone: str = "professional",
) -> str:
 
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
    )
 
    matched = [s.skill for s in gap.matched_skills[:8]]
    missing = [s.skill for s in gap.missing_skills[:3]]
    strengths_str = "\n".join(f"- {s}" for s in gap.strengths)
    quick_wins_str = "\n".join(f"- {q}" for q in gap.quick_wins)
 
    prompt = f"""Write a {tone} cover letter for {resume.name} applying to 
        {job.title} at {job.company} in {job.location}.
        
        CANDIDATE STRENGTHS FOR THIS ROLE:
        {strengths_str}
        
        MATCHED SKILLS (lead with these):
        {', '.join(matched)}
        
        GAPS TO ACKNOWLEDGE (briefly, positively):
        {', '.join(missing) if missing else 'None significant'}
        
        TAILORING NOTES:
        {quick_wins_str}
        
        RESUME SUMMARY:
        {resume.summary}
        
        INSTRUCTIONS:
        - 3 paragraphs, 250-300 words total
        - Opening: hook with the most relevant matched strength, mention company name
        - Middle: 2-3 specific examples from experience, use numbers where possible
        - Closing: address any gaps as "actively developing" skills, express genuine interest
        - DO NOT invent facts not present in the resume
        - DO NOT include placeholder text like [Your Name] — use the actual name
        - Tone: {tone}, confident but not arrogant
        - End with: "Regards, {resume.name}"
        
        Return ONLY the cover letter text. No meta-commentary."""
 
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": "You are an expert career coach who writes compelling, specific cover letters."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=600,
    )
 
    return response.choices[0].message.content.strip()
 
# Checking and making sure no Hate, SelfHarm, Sexual, Violence related content is being output
def check_content_safety(text: str) -> tuple[bool, list[str]]:
    endpoint = os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT")
    key = os.getenv("AZURE_CONTENT_SAFETY_KEY")
 
    # If Content Safety is not configured, log a warning and pass through
    # This lets the agent work without it during development
    if not endpoint or not key:
        print("[ContentSafety] WARNING: Not configured — skipping safety check.")
        print("[ContentSafety] Set AZURE_CONTENT_SAFETY_ENDPOINT and AZURE_CONTENT_SAFETY_KEY in .env")
        return True, []
 
    try:
        client = ContentSafetyClient(endpoint, AzureKeyCredential(key))
 
        request = AnalyzeTextOptions(
            text=text,
            categories=[
                TextCategory.HATE,
                TextCategory.SELF_HARM,
                TextCategory.SEXUAL,
                TextCategory.VIOLENCE,
            ],
            # output_type="FourSeverityLevels" gives scores 0,2,4,6
            # Threshold: reject if severity >= 2 (anything above "safe")
        )
 
        response = client.analyze_text(request)
 
        flagged = []
        for result in response.categories_analysis:
            if result.severity >= 2:        # 0=safe, 2=low, 4=medium, 6=high
                flagged.append(f"{result.category}(severity={result.severity})")
 
        return len(flagged) == 0, flagged
 
    except HttpResponseError as e:
        print(f"[ContentSafety] API error: {e}")
        # On API error, we pass through rather than blocking - fail open, log it
        return True, []
    
# python cover_letter_writter.py resume.pdf "Data Scientist" "Stripe" "London"
if __name__ == "__main__":
    
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "resume.pdf"
    test_role = sys.argv[2] if len(sys.argv) > 2 else "Software Engineer"
    test_company = sys.argv[3] if len(sys.argv) > 3 else "Example Company"
    test_location = sys.argv[4] if len(sys.argv) > 4 else "Remote"
 
    # Parse the real resume if available, otherwise use a placeholder
    try:
        resume = parse_resume(pdf_path)
        print(f"Parsed resume: {resume.name}")
    except FileNotFoundError:
        print(f"[Warning] {pdf_path} not found — using placeholder")
        resume = ParsedResume(
            name="Test Candidate", email="test@example.com",
            location=test_location,
            summary=f"Experienced professional seeking {test_role} roles.",
            skills=["Python", "SQL", "Communication", "Project Management"],
            experience=[{
                "role": "Senior Analyst", "company": "Previous Corp",
                "duration": "2020-Present",
                "bullets": ["Improved processes by 25%", "Led team of 5"]
            }],
            education=[], certifications=[], raw_text=""
        )
 
    # Build a minimal fake job and gap for the test
    fake_job = JobListing(
        title=test_role, company=test_company, location=test_location,
        url=f"https://example.com/jobs/{test_role.lower().replace(' ', '-')}",
        description_snippet=f"We are hiring a {test_role} at {test_company} in {test_location}.",
        skills_mentioned=resume.skills[:5]   # use candidate's own skills as mock requirements
    )
 
    fake_gap = GapAnalysis(
        job_title=test_role, company=test_company, match_score=72,
        matched_skills=[SkillMatch(skill=s, in_resume=True) for s in resume.skills[:3]],
        missing_skills=[],
        partial_matches=[],
        strengths=[f"Relevant experience from previous roles", f"Matches {test_role} requirements"],
        quick_wins=["Tailor resume summary to mention the company name"],
        recommended_resume_tweaks=["Add specific metrics to experience bullets"],
        apply_recommendation="Good match — tailor first"
    )
 
    result = generate_cover_letter(resume, fake_job, fake_gap)
    safety_result = check_content_safety(result)
    print(f"\n{result}")
    print(f"\n{safety_result}")
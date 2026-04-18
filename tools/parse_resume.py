import os
import json
import pdfplumber                         # clean PDF text extraction
from dataclasses import dataclass, field
from openai import AzureOpenAI           
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ParsedResume:
    name: str
    email: str
    location: str
    summary: str
    skills: list[str]                    # e.g. ["Python", "LangChain"]
    experience: list[dict]               # [{role, company, duration, bullets:[]}]
    education: list[dict]                # [{degree, institution, year}]
    certifications: list[str]
    raw_text: str                        # kept for fallback / debug


# Extract raw text from the PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # extract_text() preserves layout better than raw .objects access
            text = page.extract_text(x_tolerance=3, y_tolerance=3)
            if text:
                pages_text.append(text)
 
    return "\n\n--- PAGE BREAK ---\n\n".join(pages_text)

# Ask the LLM to structure the raw text
def structure_resume_with_llm(raw_text: str) -> dict:
     
     client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )
 
     system_prompt = """You are a resume parser. Extract structured information from 
the resume text and return ONLY valid JSON matching this exact schema:
{
  "name": "string",
  "email": "string",
  "location": "string",
  "summary": "string (2-3 sentences max)",
  "skills": ["list", "of", "skill", "strings"],
  "experience": [
    {
      "role": "string",
      "company": "string", 
      "duration": "string e.g. 2022-Present",
      "bullets": ["achievement 1", "achievement 2"]
    }
  ],
  "education": [
    {
      "degree": "string",
      "institution": "string",
      "year": "string"
    }
  ],
  "certifications": ["list", "of", "cert", "strings"]
}
Return ONLY the JSON object. No markdown, no explanation."""
 
     response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Parse this resume:\n\n{raw_text}"},
        ],
        response_format={"type": "json_object"},   # forces valid JSON output
        temperature=0,                             # 0 = deterministic, no hallucination
        max_tokens=2000,
    )
 
     return json.loads(response.choices[0].message.content)

#  Entry point called by the supervisor agent
# Extract raw text from PDF, Structure it with LLM, Return a ParsedResume dataclass
def parse_resume(pdf_path: str) -> ParsedResume:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Resume PDF not found at: {pdf_path}")
 
    print(f"[ResumeParser] Extracting text from {pdf_path}...")
    raw_text = extract_text_from_pdf(pdf_path)
 
    print(f"[ResumeParser] Structuring with LLM ({len(raw_text)} chars)...")
    structured = structure_resume_with_llm(raw_text)
 
    return ParsedResume(
        name=structured.get("name", ""),
        email=structured.get("email", ""),
        location=structured.get("location", ""),
        summary=structured.get("summary", ""),
        skills=structured.get("skills", []),
        experience=structured.get("experience", []),
        education=structured.get("education", []),
        certifications=structured.get("certifications", []),
        raw_text=raw_text,
    )

# Test
if __name__ == "__main__":
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else "resume.pdf"
    result = parse_resume(pdf)
    print(f"\nName     : {result.name}")
    print(f"Email    : {result.email}")
    print(f"Skills   : {result.skills[:10]}...")
    print(f"Roles    : {[e['role'] for e in result.experience]}")
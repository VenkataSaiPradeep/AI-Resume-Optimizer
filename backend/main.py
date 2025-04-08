import io
import re
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import PyPDF2
from docx import Document
import os
import httpx
from typing import List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Set up Groq API key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Please set the GROQ_API_KEY environment variable")

app = FastAPI()

# Allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Batch processing configuration
BATCH_SIZE = 1
MAX_WORKERS = 2

# Groq API configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama3-70b-8192"  # Use the specific Llama model you want to use on Groq

# API client
http_client = httpx.AsyncClient(timeout=60.0)


async def generate_text(prompt: str) -> str:
    """Generate text using the Groq API"""
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 300
        }

        response = await http_client.post(GROQ_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return "No response generated"

    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return f"Error generating text: {str(e)}"


# Parsing Functions
def parse_pdf(file: bytes) -> str:
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file))
    text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
    return text


def parse_docx(file: bytes) -> str:
    doc = Document(io.BytesIO(file))
    return "\n".join(para.text for para in doc.paragraphs)


# Regex patterns for extracting details
email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
phone_pattern = r"\(?\+?[0-9]*\)?[-.\s]?[0-9]+[-.\s]?[0-9]+[-.\s]?[0-9]+"
skills_keywords = ["python", "java", "javascript", "c++", "react", "node.js", "sql", "html", "css", "typescript",
                   "ruby", "swift", "c", "c#", "springboot", "docker", "jenkins", "kubernetes", "aws", "mysql","postgresql", "git", "github",
                   "perl", "sqlserver", "oracle", "gcp", "google cloud", "azure"]


def extract_experience(text: str) -> str:
    experience_keywords = ["experience", "years", "worked", "role", "position", "job", "experience in"]
    experience_pattern = r"(\d{1,2}\s?(years?|months?)\s?(of\s?)?[\w\s]+)"

    for line in text.split("\n"):
        if any(keyword in line.lower() for keyword in experience_keywords):
            match = re.search(experience_pattern, line, re.IGNORECASE)
            if match:
                return match.group(0).strip()
    return "Not found"


def extract_resume_data(text: str) -> dict:
    email = re.search(email_pattern, text)
    email = email.group() if email else "Not found"
    phone = re.search(phone_pattern, text)
    phone = phone.group() if phone else "Not found"
    name = next((line.strip() for line in text.split("\n") if line.strip()), "Not found")
    experience = extract_experience(text)
    skills = [skill for skill in skills_keywords if skill.lower() in text.lower()]
    return {"name": name, "email": email, "phone": phone, "experience": experience, "skills": skills}


# AI-Powered Recommendations
async def generate_career_recommendations(skills: list) -> str:
    try:
        prompt = (
            f"You are a career advisor. Given the following technical skills: {', '.join(skills)}, "
            "suggest three highly relevant career paths with job responsibilities and demand with each heading. Each point should be a clear, concise bullet point (starting with '-')"
        )
        return await generate_text(prompt)
    except Exception as e:
        print(f"Error in career recommendations: {e}")
        return "Unable to generate recommendations"


async def generate_resume_optimizations(resume_text: str) -> str:
    try:
        prompt = (
            "You are an expert resume reviewer. Here is a resume:\n\n"
            f"{resume_text}\n\n"
            "Identify exactly 3 specific, actionable improvements that will make this resume more appealing to employers. "
            "Each point should be a clear, concise bullet point (starting with '-'), focusing on clarity, quantifiable achievements, "
            "and industry standards. Do not provide generic advice."
        )
        return await generate_text(prompt)
    except Exception as e:
        print(f"Error processing resume: {e}")
        return "Unable to generate optimizations."


# Resume Parsing Endpoint
@app.post("/parse-resume")
async def parse_resume(file: UploadFile = File(...)):
    try:
        file_content = await file.read()

        if file.filename.endswith('.pdf'):
            text = parse_pdf(file_content)
        elif file.filename.endswith('.docx'):
            text = parse_docx(file_content)
        else:
            return JSONResponse(status_code=400, content={"message": "Unsupported file format"})

        parsed_data = extract_resume_data(text)

        # Run AI tasks concurrently
        career_recommendations, resume_optimizations = await asyncio.gather(
            generate_career_recommendations(parsed_data["skills"]),
            generate_resume_optimizations(text)
        )

        return {
            "parsed_data": parsed_data,
            "career_recommendations": career_recommendations,
            "resume_optimizations": resume_optimizations
        }
    except Exception as e:
        print(f"Error processing resume: {e}")
        return JSONResponse(status_code=500, content={"message": f"Error processing resume: {str(e)}"})


# Bulk Resume Parsing Endpoint
@app.post("/parse-resumes")
async def parse_multiple_resumes(files: List[UploadFile] = File(...)):
    results = []

    async def process_file(file):
        try:
            return await parse_resume(file)
        except Exception as e:
            return {"error": str(e), "filename": file.filename}

    # Process files in batches
    for i in range(0, len(files), BATCH_SIZE):
        batch = files[i:i + BATCH_SIZE]
        batch_tasks = [process_file(file) for file in batch]
        batch_results = await asyncio.gather(*batch_tasks)
        results.extend(batch_results)

    return results


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    # Initialize the HTTP client
    global http_client
    http_client = httpx.AsyncClient(timeout=60.0)


@app.on_event("shutdown")
async def shutdown_event():
    # Close the HTTP client
    await http_client.aclose()
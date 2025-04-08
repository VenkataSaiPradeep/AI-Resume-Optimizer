import spacy
import PyPDF2
import io
import re
from typing import Dict, List, Optional
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ResumeParser:
    def __init__(self):
        # Load spaCy NLP model
        self.nlp = spacy.load("en_core_web_sm")

    def extract_text_from_pdf(self, pdf_file) -> str:
        """
        Extract text from PDF file

        Args:
            pdf_file: File-like object of the PDF

        Returns:
            Extracted text from the PDF
        """
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return self.clean_text(text)
        except Exception as e:
            print(f"Error extracting PDF text: {e}")
            logger.error(f"Error extracting PDF text: {e}")
            return ""

    def clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing unnecessary characters

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        # Removing unwanted characters like extra spaces or line breaks
        return re.sub(r'\s+', ' ', text).strip()

    def parse_resume(self, pdf_file) -> Dict[str, any]:
        """
        Parse resume and extract key information

        Args:
            pdf_file: PDF file to parse

        Returns:
            Dictionary with parsed resume information
        """
        # Extract text from PDF
        resume_text = self.extract_text_from_pdf(pdf_file)

        if not resume_text:
            return {"error": "Failed to extract text from resume"}

        # Process text with spaCy
        doc = self.nlp(resume_text)

        # Extract key information
        parsed_resume = {
            "name": self._extract_name(doc),
            "email": self._extract_email(resume_text),
            "phone": self._extract_phone(resume_text),
            "skills": self._extract_skills(doc),
            "education": self._extract_education(resume_text),
            "work_experience": self._extract_work_experience(resume_text)
        }

        return parsed_resume

    def _extract_name(self, doc) -> Optional[str]:
        """
        Extract name from document using named entity recognition

        Args:
            doc: spaCy processed document

        Returns:
            Extracted name or None
        """
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text
        return None

    def _extract_email(self, text: str) -> Optional[str]:
        """
        Extract email using regex pattern

        Args:
            text: Resume text

        Returns:
            Extracted email or None
        """
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        return emails[0] if emails else None

    def _extract_phone(self, text: str) -> Optional[str]:
        """
        Extract phone number using regex pattern

        Args:
            text: Resume text

        Returns:
            Extracted phone number or None
        """
        phone_patterns = [
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # (123) 456-7890, 123-456-7890
            r'\+\d{1,2}\s?\(\d{3}\)\s?\d{3}[-.\s]?\d{4}'  # +1 (123) 456-7890
        ]

        for pattern in phone_patterns:
            phones = re.findall(pattern, text)
            if phones:
                return phones[0]

        return None

    def _extract_skills(self, doc) -> List[str]:
        """
        Extract potential skills from document

        Args:
            doc: spaCy processed document

        Returns:
            List of extracted skills
        """
        # Predefined skills list, can be expanded based on use case
        predefined_skills = [
            'python', 'java', 'javascript', 'react', 'machine learning',
            'data analysis', 'communication', 'leadership', 'problem solving',
            'tensorflow', 'git', 'sql', 'nodejs', 'agile', 'project management'
        ]

        skills = []
        for skill in predefined_skills:
            if skill.lower() in doc.text.lower():
                skills.append(skill)

        return skills

    def _extract_education(self, text: str) -> List[Dict[str, str]]:
        """
        Extract education information

        Args:
            text: Resume text

        Returns:
            List of education entries
        """
        education_patterns = [
            r'(B\.?S\.?|B\.?A\.?|M\.?S\.?|M\.?A\.?|Ph\.?D\.?) in (\w+(?:\s+\w+)*)',
            r'Degree in (\w+(?:\s+\w+)*)'
        ]

        education_entries = []
        for pattern in education_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                education_entries.append({
                    "degree": match[0],
                    "major": match[1] if len(match) > 1 else "Unknown"
                })

        return education_entries

    def _extract_work_experience(self, text: str) -> List[Dict[str, str]]:
        """
        Extract work experience information

        Args:
            text: Resume text

        Returns:
            List of work experience entries
        """
        work_pattern = r'(\w+(?:\s+\w+)*)\s*(?:at|@)\s*(\w+(?:\s+\w+)*)\s*(?:from|-)?\s*(\d{4}(?:-\d{4})?)'

        work_entries = []
        matches = re.findall(work_pattern, text, re.IGNORECASE)

        for match in matches:
            work_entries.append({
                "position": match[0],
                "company": match[1],
                "duration": match[2]
            })

        return work_entries

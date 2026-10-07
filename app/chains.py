import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

# Load environment variables from root .env file
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(base_dir, ".env"))


class Chain:
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment or .env file.")

        self.model_name = model_name
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=api_key,
            model_name=model_name
        )

    def extract_jobs(self, cleaned_text: str) -> list:
        """
        Extracts structured job postings (role, experience, skills, description) from raw webpage text.
        """
        prompt_extract = PromptTemplate.from_template(
            """
            ### INPUT JOB TEXT:
            {page_data}
            
            ### INSTRUCTION:
            The input text is a job posting, job description, or role title.
            Your task is to extract the job details and return them in a JSON list of objects, each containing:
            - `role` (string)
            - `experience` (string)
            - `skills` (list of strings)
            - `description` (string)
            
            Even if the input text is short or concise (e.g., "software engineering"), extract or infer reasonable key skills and description for the specified role.
            Only return valid JSON (no markdown text, no preamble).
            ### JSON OUTPUT:
            """
        )

        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})

        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too large or failed to parse job specifications.")

        return res if isinstance(res, list) else [res]

    def write_mail(self, job: dict, links: list, company_name: str = "AtliQ Technologies") -> str:
        """
        Generates a personalized cold email targeting the client based on job details and portfolio links.
        """
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB REQUIREMENTS:
            {job_description}

            ### INSTRUCTION:
            You are Mohan, Business Development Lead at {company_name}.
            {company_name} is a software engineering and product development firm. We help growing companies build custom software, web apps, and modern infrastructure.

            Write a short, direct, human cold outreach email to the client regarding the requirements listed above.
            Include relevant portfolio links from this list to showcase our relevant work: {link_list}

            HUMAN WRITING STYLE RULES (STRICT):
            - Sound 100% like a real human engineer/founder writing a quick, high-impact note.
            - DO NOT use generic AI openings like "I hope this email finds you well", "I am writing to express my interest", or "In today's fast-paced digital landscape".
            - DO NOT use AI buzzwords like "seamlessly", "game-changer", "testament", "cutting-edge", "delighted to reach out", or "transformative".
            - Keep it short (3-4 concise paragraphs max).
            - Use clear, straightforward language. Get straight to the value proposition.
            - Include ONLY the provided links naturally in text or bullet points.
            - Format with a Subject Line first.
            
            ### COLD EMAIL (NO PREAMBLE OR AI INTROS):
            """
        )

        chain_email = prompt_email | self.llm
        res = chain_email.invoke(input={
            "job_description": str(job),
            "link_list": ", ".join(links) if links else "None provided",
            "company_name": company_name
        })

        return res.content

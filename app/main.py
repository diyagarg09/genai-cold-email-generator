import os
import sys
import requests
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chains import Chain
from portfolio import Portfolio
from utils import clean_text


def create_streamlit_app(portfolio_db):
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    
    st.title("📧 Cold Email Generator")
    st.markdown("Generate personalized B2B client outreach & job application cold emails tailored to relevant portfolio case studies.")

    # Sidebar model selection
    with st.sidebar:
        st.header("⚙️ Configuration")
        selected_model = st.selectbox(
            "Select Inference Engine Model:",
            ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"],
            index=0
        )

    # Initialize Chain with selected model
    try:
        llm_chain = Chain(model_name=selected_model)
    except Exception as e:
        st.error(f"Failed to initialize model '{selected_model}': {e}")
        return

    tab1, tab2 = st.tabs(["🌐 Scraping via URL", "📝 Paste Job Description Direct"])

    # Tab 1: Scraping via URL
    with tab1:
        url_input = st.text_input("Enter Job Posting / Careers URL:", value="https://jobs.nike.com/job/R-33460")
        submit_url = st.button("Generate Cold Email from URL", type="primary")

        if submit_url:
            if not url_input:
                st.error("Please enter a valid URL.")
            else:
                with st.spinner("Scraping webpage & analyzing job postings..."):
                    try:
                        loader = WebBaseLoader([url_input])
                        data = clean_text(loader.load().pop().page_content)
                        portfolio_db.load_portfolio()
                        jobs = llm_chain.extract_jobs(data)

                        if not jobs:
                            st.warning("⚠️ No job specifications could be extracted from the input. Please provide a more detailed job description or URL.")
                        else:
                            for i, job in enumerate(jobs):
                                st.subheader(f"Job #{i+1}: {job.get('role', 'Position')}")
                                st.write(f"**Required Skills:** {', '.join(job.get('skills', [])) if isinstance(job.get('skills'), list) else job.get('skills')}")
                                
                                skills = job.get("skills", [])
                                links = portfolio_db.query_links(skills)
                                email = llm_chain.write_mail(job, links)

                                st.markdown("### Generated Cold Email:")
                                st.code(email, language="markdown")
                    except Exception as e:
                        st.error(f"An error occurred while fetching or processing URL: {e}")

    # Tab 2: Manual Text Input
    with tab2:
        text_input = st.text_area("Paste Raw Job Description Text:", height=200, placeholder="Paste the job requirements, responsibilities, and required tech stack here...")
        submit_text = st.button("Generate Cold Email from Text")

        if submit_text:
            if not text_input.strip():
                st.error("Please paste job description text.")
            else:
                with st.spinner("Processing job text & querying ChromaDB portfolio..."):
                    try:
                        data = clean_text(text_input)
                        portfolio_db.load_portfolio()
                        jobs = llm_chain.extract_jobs(data)

                        for i, job in enumerate(jobs):
                            st.subheader(f"Job #{i+1}: {job.get('role', 'Position')}")
                            st.write(f"**Required Skills:** {', '.join(job.get('skills', [])) if isinstance(job.get('skills'), list) else job.get('skills')}")
                            
                            skills = job.get("skills", [])
                            links = portfolio_db.query_links(skills)
                            email = llm_chain.write_mail(job, links)

                            st.markdown("### Generated Cold Email:")
                            st.code(email, language="markdown")
                    except Exception as e:
                        st.error(f"An error occurred while processing text: {e}")


if __name__ == "__main__":
    try:
        portfolio = Portfolio()
        create_streamlit_app(portfolio)
    except Exception as err:
        st.error(f"Initialization Error: {err}")

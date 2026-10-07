import re
from bs4 import BeautifulSoup


def clean_text(text: str) -> str:
    """
    Cleans raw HTML text content by stripping HTML tags, removing non-ASCII characters,
    URLs, and excess whitespace.
    """
    # Remove HTML tags using BeautifulSoup
    soup = BeautifulSoup(text, "html.parser")
    cleaned = soup.get_text(separator=" ")

    # Remove URLs
    cleaned = re.sub(r'http\S+', '', cleaned)
    # Remove non-alphanumeric characters (keeping basic punctuation)
    cleaned = re.sub(r'[^a-zA-Z0-9\s.,!?-]', ' ', cleaned)
    # Replace multiple whitespaces/newlines with a single space
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    return cleaned

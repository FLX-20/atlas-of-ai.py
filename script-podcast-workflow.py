import os
import logging
from dotenv import load_dotenv
from newsapi import NewsApiClient
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or "YOUR_FALLBACK_KEY_HERE"
PODCAST_NOTES_PATH = "podcast_notes.txt"
PODCAST_OUTPUT_PATH = "podcast_script.txt"

if not NEWS_API_KEY:
    logger.error("Missing NEWS_API_KEY in environment variables.")
    raise ValueError("Missing NEWS_API_KEY.")
if not GROQ_API_KEY:
    logger.error("Missing GROQ_API_KEY in environment variables.")
    raise ValueError("Missing GROQ_API_KEY.")

news_client = NewsApiClient(api_key=NEWS_API_KEY)
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.2-90b-vision-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

PODCAST_TITLE = "Atlas of AI"
PODCAST_FORMAT = "Conversational and co-hosted"
PODCAST_TONE = "Balanced, curious, intelligent, slightly humorous, and accessible"
PODCAST_STRUCTURE = """
    Intro 
    Discussion of 3–5 major news items in AI
    Light banter to keep the tone engaging
    Outro / Summary
"""

HOSTS = [
    {
        "name": "Alan",
        "background": "AI researcher and professor",
        "tone": "Inquisitive, slightly skeptical, and humorous",
        "style": "Engages with the audience through questions and relatable anecdotes"
    },
    {
        "name": "Arabella",
        "background": "Journalist with a tech & policy beat",
        "tone": "Energetic, optimistic",
        "style": "Good at connecting dots and explaining complex ideas clearly",
        "interests": "AI in education, global AI trends, policy and governance"
    }
]


def generate_prompt_template(hosts) -> ChatPromptTemplate:
    host_descriptions = "\n".join([
        f"## {host['name']}:\n"
        f"    Background: {host.get('background', 'N/A')}\n"
        f"    Tone: {host.get('tone', 'N/A')}\n"
        f"    Style: {host.get('style', 'N/A')}\n"
        f"    Interests: {host.get('interests', 'N/A')}" if 'interests' in host else ""
        for host in hosts
    ])

    host_names = " and ".join([host["name"] for host in hosts])

    system_prompt = f"""
You are an expert podcast scriptwriter. Your task is to generate a natural, 
engaging, and insightful dialog for an episode of the podcast “{PODCAST_TITLE}”, co-hosted by {host_names}.

# Podcast Identity:
Title: {PODCAST_TITLE}
Format: {PODCAST_FORMAT}
Tone: {PODCAST_TONE}
Duration: Approx. 4000–6000 words
Structure: {PODCAST_STRUCTURE}

# Host Personas:
{host_descriptions}

# Input:
You will be given bullet-point AI news items for this week 
(formatted as short nodes like headlines or tweets). 
Your job is to transform them into a flowing, informative, and engaging dialog between the hosts.

# Output Guidelines:
Write as a dialog script, alternating between hosts.
Include light transitions between news items (e.g., “Speaking of…”, “And in related news…”)
Insert small reactions or banter (e.g., “That’s wild”, “Classic move”, “I didn’t see that coming”).
Keep each news item focused but add brief context, opinions, and implications.
Avoid jargon unless explained briefly by one of the hosts.
End with a wrap-up summary and hint at next week’s episode.

# YOU MUST FOLLOW THIS OUTPUT FORMAT:
[HostName]: <Dialog here>
[HostName]: <Dialog here>

# NOTES FOR NEWS ITEMS:
{{sources}}
"""
    return ChatPromptTemplate.from_template(system_prompt)


prompt_template = generate_prompt_template(HOSTS)


def load_notes(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.exception(f"Failed to load notes from {file_path}")
        raise e


def generate_script(notes: str) -> str:
    try:
        formatted_prompt = prompt_template.format(sources=notes)
        response = llm.invoke(formatted_prompt)
        return response.content
    except Exception as e:
        logger.exception("Failed to generate podcast script.")
        raise e


def save_script(script: str, output_path: str) -> None:
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(script)
        logger.info(f"Script saved to {output_path}")
    except Exception as e:
        logger.exception("Failed to save podcast script.")
        raise e


def main():
    logger.info("Starting podcast script generation pipeline.")
    notes = load_notes(PODCAST_NOTES_PATH)
    script = generate_script(notes)
    logger.info("\n\n--- Generated Podcast Script ---\n" + script + "\n")
    save_script(script, PODCAST_OUTPUT_PATH)
    logger.info("Podcast script generation complete.")


if __name__ == "__main__":
    main()

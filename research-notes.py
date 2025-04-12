import os
import re
import logging
import datetime
from typing import Optional, Dict, List
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from newsapi import NewsApiClient
from requests.exceptions import HTTPError


class Config:
    def __init__(self, env_path: Path = Path('.') / '.env'):
        load_dotenv(dotenv_path=env_path)
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")

        if not self.news_api_key or not self.groq_api_key:
            raise EnvironmentError(
                "Missing NEWS_API_KEY or GROQ_API_KEY in .env")

        self.remove_tags = ["script", "style", "header",
                            "footer", "nav", "noscript", "aside"]


class NewsFetcher:
    def __init__(self, config: Config, query: str = "AI"):
        self.client = NewsApiClient(api_key=config.news_api_key)
        self.query = query

    def fetch(self, start_date: str, end_date: str, max_articles: int = 5) -> List[Dict]:
        try:
            response = self.client.get_everything(
                q=self.query,
                language="en",
                sort_by="relevancy",
                page=1,
                from_param=start_date,
                to=end_date,
            )
            return response.get("articles", [])[:max_articles]
        except Exception as e:
            logging.error(f"News fetch failed: {e}")
            return []


class ArticleExtractor:
    def __init__(self, config: Config):
        self.remove_tags = config.remove_tags

    def extract_text(self, url: str) -> Optional[Dict[str, str]]:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
            )
        }

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except HTTPError as http_err:
            logging.error(f"HTTP error at {url}: {http_err}")
            return None
        except Exception as err:
            logging.error(f"Error at {url}: {err}")
            return None

        soup = BeautifulSoup(response.content, "html.parser")
        for tag in soup.select(",".join(self.remove_tags)):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()

        return {"title": title, "text": text}


class Summarizer:
    def __init__(self, config: Config, model_name: str = "llama-3.2-90b-vision-preview"):
        self.llm = ChatGroq(
            api_key=config.groq_api_key,
            model=model_name,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

    def summarize(self, system_prompt: str, sources: str) -> str:
        prompt_template = ChatPromptTemplate.from_template(system_prompt)
        formatted_prompt = prompt_template.format(sources=sources)
        return self.llm.invoke(formatted_prompt).content


class PodcastResearcher:
    def __init__(self, output_file: Path = Path("podcast_notes.txt")):
        self.output_file = output_file
        self.output_file.unlink(missing_ok=True)

    def format_article(self, index: int, article: Dict, extracted: Dict) -> str:
        return f"""## Article {index}
title: {article.get('title', '[No title]')}
publishedAt: {article.get('publishedAt', '[No date]')}
content: {article.get('description', '[No description available]')}
url: {article.get('url', '[No URL]')}
text: {extracted['text']}"""

    def save_summary(self, summary: str):
        with self.output_file.open("a", encoding="utf-8") as f:
            f.write(summary + "\n\n")


class AIResearchGenerator:
    def __init__(self, config: Config, days: int = 7):
        self.config = config
        self.news_fetcher = NewsFetcher(config)
        self.extractor = ArticleExtractor(config)
        self.summarizer = Summarizer(config)
        self.writer = PodcastResearcher()
        self.days = days
        self.system_prompt = """
You are an AI host Abby of a podcast called “This Week in AI”.
Your task is to summarize the provided article in concise notes, focusing on the most important points for your podcast.

# PROVIDED ARTICLE
{sources}
        """.strip()

    def get_date_range(self):
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=self.days)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    def run(self):
        start_date, end_date = self.get_date_range()
        articles = self.news_fetcher.fetch(start_date, end_date)

        if not articles:
            logging.warning("No articles found.")
            return

        for i, article in enumerate(articles, start=1):
            url = article.get("url")
            if not url:
                continue

            extracted = self.extractor.extract_text(url)
            if not extracted:
                logging.warning(
                    f"Skipping article {i} due to failed extraction.")
                continue

            formatted = self.writer.format_article(i, article, extracted)
            try:
                summary = self.summarizer.summarize(
                    self.system_prompt, formatted)
                self.writer.save_summary(summary)
                logging.info(f"Article {i} summarized successfully.")
            except Exception as e:
                logging.error(f"Error summarizing article {i}: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    config = Config()
    generator = AIResearchGenerator(config=config)
    generator.run()

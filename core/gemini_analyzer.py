
import os
from typing import List

from dotenv import load_dotenv
from google import genai
from google.genai import types

from core.prompts import (
    SentimentPrompts,
    CategoryPrompts,
    SummaryPrompts,
    PromptTemplate,
)

load_dotenv()


class GeminiAnalyzer:
    """Analyze customer feedback using Google's Gemini API."""

    def __init__(self, model: str = "gemini-3.8-flash"):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing. "
                "Add it to the .env file in the project root."
            )

        self.model = model
        self.client = genai.Client(api_key=api_key)

    def _generate(
        self,
        prompt_template: PromptTemplate,
        variables: dict,
    ) -> str:
        prompt = prompt_template.template.format(**variables)

        try:
            config = types.GenerateContentConfig(
                temperature=prompt_template.temperature,
                max_output_tokens=prompt_template.max_tokens,
            )

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )

            if not response.text:
                raise RuntimeError("Gemini returned an empty response.")

            return response.text.strip()

        except Exception as exc:
            raise RuntimeError(
                f"Gemini API request failed: {exc}"
            ) from exc

    def analyze_sentiment(self, feedback: str) -> str:
        return self._generate(
            SentimentPrompts.ANALYZE,
            {"feedback": feedback},
        )

    def classify_category(self, feedback: str) -> str:
        return self._generate(
            CategoryPrompts.CLASSIFY,
            {"feedback": feedback},
        )

    def summarize_feedback(self, feedback_list: List[str]) -> str:
        joined_feedback = "\n".join(feedback_list[:20])

        return self._generate(
            SummaryPrompts.SUMMARIZE,
            {"joined_feedback": joined_feedback},
        )

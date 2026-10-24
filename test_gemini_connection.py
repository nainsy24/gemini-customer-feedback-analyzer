
from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise SystemExit(
        "GEMINI_API_KEY is missing from the root .env file."
    )

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=(
        "Classify this customer feedback as Positive, Neutral, "
        "or Negative. Reply with one word only: "
        "'The product is excellent, but delivery was late.'"
    ),
)

print("Gemini response:")
print(response.text)

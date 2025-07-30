# Import necessary libraries
from config import settings
from google import genai

# Create a Gemini client instance
gemini_client = genai.Client(api_key=settings.gemini_api_key)

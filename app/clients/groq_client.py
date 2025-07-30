# Import necessary libraries
from config import settings
from groq import Groq

# Create a Groq client instance
groq_client = Groq(api_key=settings.groq_api_key)

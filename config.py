# Import necessary libraries
from pydantic_settings import BaseSettings, SettingsConfigDict

# Define the Settings class using Pydantic
class Settings(BaseSettings):
    environment: str
    gemini_api_key: str
    groq_api_key: str
    chroma_db_path: str = "./chroma_db"
    data_directory_path: str = "./data"
    model_config = SettingsConfigDict(env_file=".env")

# Create an instance of Settings
settings = Settings()

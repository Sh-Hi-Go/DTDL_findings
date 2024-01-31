import os
from chromadb.config import Settings

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Define the folder for storing database
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS"

PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/VECTOR_STORE"

CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)
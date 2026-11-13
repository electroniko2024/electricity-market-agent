"""LLM package. Importing it loads LLM settings from the project-root ``.env`` file."""

from pathlib import Path

PROJECT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional; plain environment variables still work.
    pass
else:
    # override=False: variables already set in the shell or by Docker win over .env values.
    load_dotenv(PROJECT_ENV_FILE, override=False)

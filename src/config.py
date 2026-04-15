import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Each model is keyed by a short identifier and maps to the OpenRouter model path.
MODELS = {
    "gpt-5.4": "openai/gpt-5.4",
    "gpt-5.4-mini": "openai/gpt-5.4-mini",
    "claude-sonnet-4.6": "anthropic/claude-sonnet-4.6",
    "gemini-3.1-pro": "google/gemini-3.1-pro-preview",
    "llama-4-maverick": "meta-llama/llama-4-maverick",
    "mistral-large": "mistralai/mistral-large-2512",
    "deepseek-v3.2": "deepseek/deepseek-v3.2",
}

# Brands are split into "established" (high existing visibility) and "emerging"
# (lower visibility). This lets us measure whether optimization disproportionately
# benefits lesser-known brands, consistent with Aggarwal et al. (KDD 2024).
CATEGORIES = {
    "crm_software": {
        "display_name": "CRM Software",
        "brands": {
            "established": [
                "Salesforce",
                "HubSpot",
            ],
            "emerging": [
                "Copper",
                "Less Annoying CRM",
                "Freshsales",
            ],
        },
    },
    "project_management": {
        "display_name": "Project Management Tools",
        "brands": {
            "established": [
                "Asana",
                "Jira",
            ],
            "emerging": [
                "ClickUp",
                "Notion",
            ],
        },
    },
}

# Prompts are bucketed by specificity to study how query detail affects
# which brands surface. "vague" prompts tend to favor dominant brands;
# "specific" prompts sometimes surface niche players.
PROMPT_TEMPLATES = {
    "vague": [
        "What's the best {category}?",
        "Recommend a {category} for me.",
        "Which {category} should I use?",
    ],
    "medium": [
        "Best {category} for startups",
        "Best {category} for small businesses",
        "Most affordable {category} for growing companies",
        "Best {category} for remote teams",
    ],
    "specific": [
        "Best {category} for a 50-person B2B SaaS startup under $20K/year that integrates with Slack and Gmail",
        "Which {category} is best for a real estate agency managing 200 leads per month with a budget of $500/month?",
        "Best {category} for a remote marketing team of 15 people who need Kanban boards and time tracking",
    ],
}

# Used for LLM-as-judge sentiment evaluation. A smaller model keeps costs low
# while still producing reliable structured JSON output.
JUDGE_MODEL = "openai/gpt-5.4-mini"

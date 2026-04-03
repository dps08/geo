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
                "Zoho CRM",
                "Microsoft Dynamics 365",
            ],
            "emerging": [
                "Copper",
                "Close",
                "Freshsales",
                "Less Annoying CRM",
                "Pipedrive",
            ],
        },
    },
    "project_management": {
        "display_name": "Project Management Tools",
        "brands": {
            "established": [
                "Asana",
                "Monday.com",
                "Jira",
                "Microsoft Project",
            ],
            "emerging": [
                "Basecamp",
                "Teamwork",
                "ClickUp",
                "Wrike",
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
        "What {category} do you recommend?",
        "Top {category} options?",
    ],
    "medium": [
        "Best {category} for startups",
        "Best {category} for small businesses",
        "Top {category} for teams under 50 people",
        "Most affordable {category} for growing companies",
        "Best {category} for remote teams",
        "Which {category} has the best free plan?",
        "Best {category} for B2B companies",
        "Most user-friendly {category} for beginners",
    ],
    "specific": [
        "Best {category} for a 50-person B2B SaaS startup under $20K/year that integrates with Slack and Gmail",
        "Which {category} is best for a real estate agency managing 200 leads per month with a budget of $500/month?",
        "Best {category} for a remote marketing team of 15 people who need Kanban boards and time tracking",
        "Recommend a {category} for a nonprofit with 30 employees that needs donor tracking and email integration",
        "Best {category} for a freelance consultant who needs mobile access and simple invoicing",
        "Which {category} works best for an e-commerce company with 100 employees managing multiple sales pipelines?",
        "Best {category} for a healthcare startup that needs HIPAA compliance and patient record management",
    ],
}

# Used for LLM-as-judge sentiment evaluation. A smaller model keeps costs low
# while still producing reliable structured JSON output.
JUDGE_MODEL = "openai/gpt-5.4-mini"

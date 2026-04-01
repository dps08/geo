"""Configuration for the GEO platform."""

import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Models to query via OpenRouter (latest as of April 2026)
MODELS = {
    "gpt-5.4": "openai/gpt-5.4",
    "gpt-5.4-mini": "openai/gpt-5.4-mini",
    "claude-sonnet-4.6": "anthropic/claude-sonnet-4.6",
    "gemini-3.1-pro": "google/gemini-3.1-pro-preview",
    "llama-4-maverick": "meta-llama/llama-4-maverick",
    "mistral-large": "mistralai/mistral-large-2512",
    "deepseek-v3.2": "deepseek/deepseek-v3.2",
}

# Product categories and brands to track
CATEGORIES = {
    "crm_software": {
        "display_name": "CRM Software",
        "brands": {
            "large": ["Salesforce", "HubSpot", "Zoho CRM", "Microsoft Dynamics 365"],
            "small": ["Copper", "Close", "Freshsales", "Less Annoying CRM", "Pipedrive"],
        },
    },
    "project_management": {
        "display_name": "Project Management Tools",
        "brands": {
            "large": ["Asana", "Monday.com", "Jira", "Microsoft Project"],
            "small": ["Basecamp", "Teamwork", "ClickUp", "Wrike", "Notion"],
        },
    },
}

# Prompt templates with varying specificity levels
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

# LLM-as-judge model for sentiment analysis
JUDGE_MODEL = "openai/gpt-5.4-mini"

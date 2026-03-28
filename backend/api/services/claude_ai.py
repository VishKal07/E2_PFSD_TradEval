import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CLAUDE_API_KEY")

def ask_claude(question, context=""):

    url = "https://api.anthropic.com/v1/messages"

    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    payload = {
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 200,
        "messages": [
            {"role": "user", "content": question + "\n" + context}
        ]
    }

    response = requests.post(url, headers=headers, json=payload)

    data = response.json()

    return data["content"][0]["text"]
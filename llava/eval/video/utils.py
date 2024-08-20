import json
import os
import time

import openai
import requests
from openai import AzureOpenAI


def get_client():

    if os.getenv("OPENAI_API_KEY"):
        client = openai  # use default openai
    elif os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-01",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
    else:
        raise ValueError("At least provide one format for gpt assisted benchmarking")

    return client

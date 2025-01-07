import os
import re

# OpenAI
import openai

from .utilities import *

openai.api_key = os.getenv("OPENAI_API_KEY")
# print(openai.api_key)

# load demo prompt
from .prompts.ext_ans import demo_prompt


def verify_extraction(extraction):
    extraction = extraction.strip()
    if extraction == "" or extraction == None:
        return False
    return True


def create_test_prompt(demo_prompt, query, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"{query}\n\n{response}"
    full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
    return full_prompt


def extract_answer(response, problem, quick_extract=False):
    question_type = problem["question_type"]
    answer_type = problem["answer_type"]
    choices = problem["choices"]
    query = problem["query"]
    pid = problem["pid"]

    if response == "":
        return ""

    if question_type == "multi_choice" and response in choices:
        return response

    if answer_type == "integer":
        try:
            extraction = int(response)
            return str(extraction)
        except:
            pass

    if answer_type == "float":
        try:
            extraction = str(float(response))
            return extraction
        except:
            pass

    # quick extraction
    if quick_extract:
        print("Quickly extracting answer...")
        # The answer is "text". -> "text"
        try:
            result = re.search(r'The answer is "(.*)"\.', response)
            if result:
                extraction = result.group(1)
                return extraction
        except:
            pass

    # general extraction
    try:
        from openai import OpenAI

        client = OpenAI()
        full_prompt = create_test_prompt(demo_prompt, query, response)
        extraction = get_chat_response(full_prompt, client)
        return extraction
    except Exception as e:
        print(e)
        print(f"Error in extracting answer for {pid}")

    return ""



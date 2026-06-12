from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_retries=3,
)

vision_llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    max_retries=3,
)

guardrail_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_retries=3,
)
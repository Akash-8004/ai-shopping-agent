"""
Response Quality Evaluation (LLM-as-Judge)
==========================================
Uses an LLM judge to score shopping agent responses on:
  1. Relevance  (1-5)
  2. Correctness  (1-5)
  3. Format Compliance  (1-5)

Run:  python evals/response_quality_eval.py
"""

import json
import os
import re
import sys
import time
import uuid

# ── Make project importable from evals/ sub-directory ──
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from llms import llm as judge_llm  # noqa: E402
from shopping_agent import create_shopping_agent  # noqa: E402


# ── Colours for terminal output ──────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def score_color(score: float) -> str:
    if score >= 4:
        return GREEN
    elif score >= 3:
        return YELLOW
    return RED


# ── Product catalog reference (given to the judge) ───────────────────────────

CATALOG_REFERENCE = """\
PRODUCT CATALOG (ground truth):
- Honey: Organic Raw Honey $14.99 (organic, ID:1), Wildflower Honey $12.99 (ID:2), \
Organic Manuka Honey $29.99 (organic, ID:3), Clover Honey $8.99 (ID:4), \
Organic Buckwheat Honey $18.99 (organic, ID:5), Orange Blossom Honey $15.99 (ID:6), \
Organic Acacia Honey $17.99 (organic, ID:7), Creamed Honey $11.99 (ID:8)
- Oil: Organic Extra Virgin Olive Oil $16.99 (organic, ID:9), Coconut Oil $12.49 (ID:10), \
Organic Flaxseed Oil $14.99 (organic, ID:11), Avocado Oil $18.99 (ID:12)
- Nuts: Organic Almonds $11.99 (organic, ID:13), Roasted Cashews $9.99 (ID:14), \
Mixed Nuts $13.99 (ID:16)
- Seeds: Organic Chia Seeds $8.49 (organic, ID:15)
- Grains: Organic Quinoa $10.99 (organic, ID:17), Rolled Oats $5.49 (ID:18), \
Organic Brown Rice $7.99 (organic, ID:19), Steel-Cut Oats $6.99 (ID:20)
- Tea: Organic Green Tea $12.99 (organic, ID:21), Chamomile Tea $8.99 (ID:22)
- Coffee: Organic Ethiopian Coffee $16.99 (organic, ID:23), Dark Roast Espresso Blend $14.49 (ID:24)
- Snacks: Organic Granola $9.99 (organic, ID:25), Rice Cakes $4.49 (ID:26), \
Organic Dried Mango $7.99 (organic, ID:27), Trail Mix $8.49 (ID:28)
- Dairy-alt: Organic Almond Milk $4.99 (organic, ID:29), Oat Milk $4.49 (ID:30), \
Organic Coconut Milk $3.99 (organic, ID:31), Soy Milk $3.49 (ID:32)
"""

EXPECTED_FORMAT = (
    "The required product listing format is:\n"
    "#<number>. <name> (ID:<product_id>) - $<price> - rating <rating> - <organic or non-organic>"
)


# ── Judge prompt ─────────────────────────────────────────────────────────────

def build_judge_prompt(user_query: str, agent_response: str, context: str) -> str:
    return f"""/no_think
You are an expert evaluator judging a shopping assistant's response.

{CATALOG_REFERENCE}

{EXPECTED_FORMAT}

EVALUATION CONTEXT: {context}

USER QUERY:
\"{user_query}\"

AGENT RESPONSE:
\"\"\"{agent_response}\"\"\"

Score the response on three criteria (each 1-5, where 5 is perfect):

1. **Relevance**: Does the response address the user's query? Are the products shown related to what was asked?
2. **Correctness**: Are the products factually correct? Do the prices, organic status, and IDs match the catalog? Are filters (price, organic) applied correctly?
3. **Format Compliance**: Does the response use the required numbered list format: #N. Name (ID:X) - $Price - rating R - organic/non-organic?

Return ONLY a JSON object with this exact structure, no other text:
{{"relevance": <int>, "correctness": <int>, "format_compliance": <int>, "reasoning": "<brief explanation>"}}
"""


def parse_judge_response(text: str) -> dict:
    """Extract JSON from judge LLM response, handling code blocks."""
    # Remove markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    cleaned = cleaned.rstrip("`").strip()

    # Try to find JSON object
    match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return {
                "relevance": int(data.get("relevance", 0)),
                "correctness": int(data.get("correctness", 0)),
                "format_compliance": int(data.get("format_compliance", 0)),
                "reasoning": str(data.get("reasoning", "N/A")),
            }
        except (json.JSONDecodeError, ValueError):
            pass

    return {
        "relevance": 0,
        "correctness": 0,
        "format_compliance": 0,
        "reasoning": f"Failed to parse judge response: {text[:200]}",
    }


# ── Test case definitions ────────────────────────────────────────────────────

TEST_CASES = [
    {
        "name": "Organic honey under $20",
        "query": "Show me organic honey under $20",
        "context": (
            "Should return organic honey products priced under $20. "
            "Expected: Organic Raw Honey $14.99 (ID:1), Organic Buckwheat Honey $18.99 (ID:5), "
            "Organic Acacia Honey $17.99 (ID:7). Should NOT include Organic Manuka Honey $29.99."
        ),
    },
    {
        "name": "Olive oil listing",
        "query": "What olive oils do you have?",
        "context": (
            "Should list olive oil products. The catalog has Organic Extra Virgin Olive Oil "
            "$16.99 (ID:9). May also show other oils if search is broad."
        ),
    },
    {
        "name": "Cheapest grains",
        "query": "I want the cheapest grains",
        "context": (
            "Should show grain products, ideally ordered or highlighting cheapest first. "
            "Grains: Rolled Oats $5.49, Steel-Cut Oats $6.99, Organic Brown Rice $7.99, "
            "Organic Quinoa $10.99."
        ),
    },
    {
        "name": "Tea options",
        "query": "Show me all tea options",
        "context": (
            "Should list all tea products: Organic Green Tea $12.99 (ID:21) and "
            "Chamomile Tea $8.99 (ID:22)."
        ),
    },
    {
        "name": "Organic snacks",
        "query": "Find me organic snacks",
        "context": (
            "Should show only organic snacks: Organic Granola $9.99 (ID:25) and "
            "Organic Dried Mango $7.99 (ID:27). Should NOT include Rice Cakes or Trail Mix."
        ),
    },
    {
        "name": "Coffee products",
        "query": "What coffee do you sell?",
        "context": (
            "Should list coffee products: Organic Ethiopian Coffee $16.99 (ID:23) and "
            "Dark Roast Espresso Blend $14.49 (ID:24)."
        ),
    },
]


# ── Main runner ──────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}{CYAN}{'=' * 65}")
    print("  📊  RESPONSE QUALITY EVALUATION  (LLM-as-Judge)")
    print(f"{'=' * 65}{RESET}\n")

    all_scores = []

    for i, tc in enumerate(TEST_CASES):
        label = f"[{i + 1}/{len(TEST_CASES)}] {tc['name']}"
        print(f"{BOLD}{label}{RESET}")
        print(f"  Query: \"{tc['query']}\"")

        try:
            # ── Step 1: Get agent response ──
            thread_id = str(uuid.uuid4())
            agent = create_shopping_agent(f"eval_{thread_id}", InMemorySaver())
            response = agent.invoke(
                {"messages": [HumanMessage(content=tc["query"])]},
                config={"configurable": {"thread_id": thread_id}},
            )
            reply = response["messages"][-1].content

            # Truncate display
            display_reply = reply[:300] + ("..." if len(reply) > 300 else "")
            print(f"  Agent:  {display_reply}")

            # Rate-limit pause before judge call
            time.sleep(3)

            # ── Step 2: Judge the response ──
            judge_prompt = build_judge_prompt(tc["query"], reply, tc["context"])
            judge_response = judge_llm.invoke(judge_prompt)
            scores = parse_judge_response(judge_response.content)

            # Show scores
            rel = scores["relevance"]
            cor = scores["correctness"]
            fmt = scores["format_compliance"]
            avg = round((rel + cor + fmt) / 3, 1)

            print(
                f"  Scores: "
                f"Relevance={score_color(rel)}{rel}/5{RESET}  "
                f"Correctness={score_color(cor)}{cor}/5{RESET}  "
                f"Format={score_color(fmt)}{fmt}/5{RESET}  "
                f"Avg={score_color(avg)}{avg}{RESET}"
            )
            print(f"  Judge:  {scores['reasoning']}")

            all_scores.append(scores)

        except Exception as e:
            print(f"  {RED}⚠️  ERROR: {type(e).__name__}: {e}{RESET}")
            all_scores.append({
                "relevance": 0,
                "correctness": 0,
                "format_compliance": 0,
                "reasoning": f"Error: {e}",
            })

        print()

        # Rate-limit pause between full test cycles
        if i < len(TEST_CASES) - 1:
            time.sleep(5)

    # ── Summary table ──
    print(f"\n{BOLD}{CYAN}{'=' * 65}")
    print("  SUMMARY")
    print(f"{'=' * 65}{RESET}\n")

    header = f"  {'Test Case':<30} {'Rel':>4} {'Cor':>4} {'Fmt':>4} {'Avg':>5}"
    print(f"{BOLD}{header}{RESET}")
    print(f"  {'─' * 50}")

    sum_rel, sum_cor, sum_fmt = 0, 0, 0
    valid = 0

    for tc, scores in zip(TEST_CASES, all_scores):
        rel = scores["relevance"]
        cor = scores["correctness"]
        fmt = scores["format_compliance"]
        avg = round((rel + cor + fmt) / 3, 1)

        if rel > 0 or cor > 0 or fmt > 0:
            valid += 1
            sum_rel += rel
            sum_cor += cor
            sum_fmt += fmt

        print(
            f"  {tc['name']:<30} "
            f"{score_color(rel)}{rel:>4}{RESET} "
            f"{score_color(cor)}{cor:>4}{RESET} "
            f"{score_color(fmt)}{fmt:>4}{RESET} "
            f"{score_color(avg)}{avg:>5}{RESET}"
        )

    if valid > 0:
        avg_rel = round(sum_rel / valid, 1)
        avg_cor = round(sum_cor / valid, 1)
        avg_fmt = round(sum_fmt / valid, 1)
        overall = round((avg_rel + avg_cor + avg_fmt) / 3, 1)

        print(f"  {'─' * 50}")
        print(
            f"  {BOLD}{'AVERAGE':<30} "
            f"{score_color(avg_rel)}{avg_rel:>4}{RESET} "
            f"{score_color(avg_cor)}{avg_cor:>4}{RESET} "
            f"{score_color(avg_fmt)}{avg_fmt:>4}{RESET} "
            f"{score_color(overall)}{overall:>5}{RESET}"
        )

    print(f"\n  {BOLD}Evaluated: {valid}/{len(TEST_CASES)} test cases{RESET}\n")


if __name__ == "__main__":
    main()

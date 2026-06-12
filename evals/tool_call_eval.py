"""
Tool Call Accuracy Eval 
========================================
Send a user query → check if the agent called the right tool with the right args.

Run:  python evals/tool_call_eval.py
"""

import os, sys, time, uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from shopping_agent import create_shopping_agent


# ── Step 1: Helper to extract tool calls from agent response ─────────────────

def get_tool_calls(query):
    """Send a query to a fresh agent, return list of (tool_name, args) tuples."""
    tid = str(uuid.uuid4())
    agent = create_shopping_agent(f"eval_{tid}", InMemorySaver())
    resp = agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": tid}},
    )
    # Extract tool calls from AIMessage objects in the response
    calls = []
    for msg in resp["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                calls.append((tc["name"], tc["args"]))
    return calls


def called(calls, tool_name):
    """Was this tool called?"""
    return any(name == tool_name for name, _ in calls)


def args_of(calls, tool_name):
    """Get the args dict for the first call to this tool."""
    for name, args in calls:
        if name == tool_name:
            return args
    return {}


# ── Step 2: Define test cases ────────────────────────────────────────────────
#
#  Each test = (description, query, check_function)
#
#  The check_function receives the list of tool calls and returns True/False.

TESTS = [
    (
        "organic honey under $20 → search_products(is_organic=True, max_price≤20)",
        "organic honey under $20",
        lambda c: (
            called(c, "search_products")
            and args_of(c, "search_products").get("is_organic") is True
            and (args_of(c, "search_products").get("max_price") or 999) <= 20
        ),
    ),
    (
        "show me olive oil → search_products(query has 'olive' or 'oil')",
        "show me olive oil",
        lambda c: (
            called(c, "search_products")
            and any(kw in args_of(c, "search_products").get("query", "").lower()
                    for kw in ["olive", "oil"])
        ),
    ),
    (
        "remember I prefer organic → save_preference_for_visitor",
        "remember I prefer organic products",
        lambda c: called(c, "save_preference_for_visitor"),
    ),
    (
        "what have I ordered? → get_order_history_for_visitor",
        "what have I ordered before?",
        lambda c: called(c, "get_order_history_for_visitor"),
    ),
    (
        "show me coffee → search_products(query has 'coffee')",
        "show me coffee",
        lambda c: (
            called(c, "search_products")
            and "coffee" in args_of(c, "search_products").get("query", "").lower()
        ),
    ),
    (
        "what are my preferences? → get_preferences_for_visitor",
        "what are my saved preferences?",
        lambda c: called(c, "get_preferences_for_visitor"),
    ),
]


# ── Step 3: Run tests and report ─────────────────────────────────────────────

def main():
    print("\n🔧 TOOL CALL ACCURACY EVAL\n" + "=" * 45)

    passed, total = 0, len(TESTS)

    for i, (desc, query, check_fn) in enumerate(TESTS):
        print(f"\n[{i+1}/{total}] {desc}")
        print(f"  Query: \"{query}\"")

        try:
            calls = get_tool_calls(query)

            # Show what the agent actually called
            if calls:
                for name, args in calls:
                    print(f"  Agent called: {name}({args})")
            else:
                print("  Agent called: (nothing)")

            # Check if it matches expectations
            if check_fn(calls):
                print("  ✅ PASS")
                passed += 1
            else:
                print("  ❌ FAIL")

        except Exception as e:
            print(f"  ⚠️ ERROR: {e}")

        # Small delay to avoid Groq rate limits
        if i < total - 1:
            time.sleep(3)

    # Summary
    print(f"\n{'=' * 45}")
    print(f"Result: {passed}/{total} passed\n")


if __name__ == "__main__":
    main()

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from llms import llm, guardrail_llm
from tools import make_tools


SYSTEM_PROMPT = (
    "You are a helpful shopping assistant. Follow these rules strictly.\n\n"
    "IMAGE SEARCH - when the user provides an image path:\n"
    "1. Call analyze_product_image with the path to identify the product.\n"
    "2. Use the returned search_query and is_organic to call search_products.\n"
    "3. Continue with the BROWSING flow from step 2 onwards.\n\n"
    "BROWSING - when the user describes what they want to buy:\n"
    "1. Call get_preferences_for_visitor first if the request should be personalized.\n"
    "2. Call search_products to find matching items, applying any price, organic, or saved preference filters that are relevant.\n"
    "3. For each candidate, call get_rating to retrieve its average rating.\n"
    "4. Filter by the user's minimum rating if specified.\n"
    "5. Present qualifying products as a numbered list. For each item use this exact format "
    "   (plain text, no backticks, no code blocks, no bold, no italic):\n\n"
    "   #<number>. <name> (ID:<product_id>) - $<price> - rating <rating> - <organic or non-organic>\n\n"
    "   Add a blank line between each product entry for readability. "
    "   Always include (ID:X) so you can reference it later.\n"
    "6. If only one product qualifies, still show it in the list and ask: "
    "   'Would you like to order it? Just say yes or give me the number.'\n"
    "7. Do NOT call checkout_for_visitor at this stage.\n\n"
    "ORDERING - when the user confirms they want to buy (e.g. 'yes', 'sure', 'go ahead', "
    "'order number 2', 'the first one', 'get me #3'):\n"
    "1. Look at your previous message to find the (ID:X) for the chosen product "
    "   (if only one was listed and the user says 'yes', use that product's ID).\n"
    "2. Call checkout_for_visitor with that product_id.\n"
    "3. Confirm the order to the user in plain text.\n\n"
    "MEMORY & PERSONALIZATION:\n"
    "1. If the user says things like 'remember that', 'I prefer', 'I only buy', "
    "'don't recommend', or 'my budget is', call save_preference_for_visitor.\n"
    "2. If the user asks about previous orders, order history, past purchases, or whether "
    "they ordered something before, call get_order_history_for_visitor.\n"
    "3. Before giving personalized recommendations, call get_preferences_for_visitor.\n"
    "4. Use saved preferences whenever relevant, including budget limits, organic preferences, "
    "favorite categories, brands, or products the user wants to avoid.\n"
    "5. If no preferences are available, continue normally.\n"
    "6. Never invent preferences. Only use preferences returned by get_preferences_for_visitor.\n"
    "7. Never invent order history. Only use information returned by get_order_history_for_visitor.\n\n"
    "GENERAL RULES:\n"
    "1. Never place an order unless the user explicitly confirms.\n"
    "2. Never guess a product_id - always take it from the (ID:X) in your own previous message.\n"
    "3. If a tool can answer the user's question, use the tool instead of making assumptions.\n"
)

# input Guardrails
def is_shopping_request(user_input: str):

    prompt = f"""
    You are a classifier.

    Return only YES or NO.

    YES:
    - product search
    - product recommendation
    - orders
    - purchases
    - shopping advice
    - cart related questions
    - Or Something Shopping releted

    NO:
    - weather
    - coding help
    - poems
    - jokes
    - general knowledge
    - Or something out of scope of shopping

    User message:
    {user_input}
    """
    result = guardrail_llm.invoke(prompt)
    return result.content.strip().upper() == "YES"


def create_shopping_agent(visitor_id: str, memory: InMemorySaver | None = None):
    """Create a shopping agent whose personal tools are scoped to visitor_id."""
    return create_agent(
        model=llm,
        tools=make_tools(visitor_id),
        system_prompt=SYSTEM_PROMPT,
        checkpointer=memory or InMemorySaver(),
    )


if __name__ == "__main__":
    session_id = "demo123"
    agent = create_shopping_agent(session_id)

    while True:
        query = input("You: ")
        if not is_shopping_request(query):
            print(
                "I'm a shopping assistant. I can help with products, "
                "recommendations, purchases, and order history."
            )
            continue
        response = agent.invoke(
            {
                "messages": [
                    HumanMessage(content=query),
                ]
            },
            config={
                "configurable": {
                    "thread_id": session_id,
                }
            },
        )

        print(response["messages"][-1].content)

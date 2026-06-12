import base64
import json
import os

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

from llms import vision_llm
from review_api import get_average_rating


DB_PATH = os.path.join(os.path.dirname(__file__), "store.db")


@tool
def search_products(
    query: str,
    max_price: float | None = None,
    is_organic: bool | None = None,
) -> str:
    """
    Search products by keyword with optional price and organic filters.

    Args:
        query: Product name, category, or description keyword.
        max_price: Maximum product price to include.
        is_organic: Filter by organic status.

    Returns:
        JSON list of matching products.
    """
    import sqlite3

    sql = """
        SELECT
            id,
            name,
            category,
            price,
            description,
            is_organic
        FROM products
        WHERE (
            name LIKE ?
            OR category LIKE ?
            OR description LIKE ?
        )
    """

    params = [
        f"%{query}%",
        f"%{query}%",
        f"%{query}%",
    ]

    if max_price is not None:
        sql += " AND price <= ?"
        params.append(max_price)

    if is_organic is not None:
        sql += " AND is_organic = ?"
        params.append(int(is_organic))

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        rows = cursor.fetchall()

    return json.dumps(
        [
            {
                "id": row[0],
                "name": row[1],
                "category": row[2],
                "price": row[3],
                "description": row[4],
                "is_organic": bool(row[5]),
            }
            for row in rows
        ]
    )


@tool
def get_rating(product_id: int) -> str:
    """
    Get the average rating and total review count for a product.

    Args:
        product_id: Unique identifier of the product.

    Returns:
        JSON containing the product ID, average rating, and review count.
    """
    return json.dumps(get_average_rating(product_id))


@tool
def analyze_product_image(image_path: str) -> dict:
    """
    Analyze a product image and extract product information.

    Args:
        image_path: Path to the product image.

    Returns:
        Dictionary containing product_type, search_query, is_organic, and
        description.
    """
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    response = vision_llm.invoke(
        [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Look at this product image. "
                            "If a product is visible, identify it.\n\n"
                            "Return ONLY a JSON object with these fields:\n"
                            "- product_type: what kind of product it is "
                            "(e.g. honey, olive oil, almonds)\n"
                            "- search_query: a short keyword to search for it "
                            "(e.g. 'honey', 'olive oil')\n"
                            "- is_organic: true if the label says organic, "
                            "false if not, null if unclear\n"
                            "- description: one sentence describing the product\n\n"
                            "If no product is visible, return:\n"
                            '{"product_type": null, "search_query": null, '
                            '"is_organic": null, "description": "No product detected"}'
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                        },
                    },
                ]
            )
        ]
    )

    try:
        return json.loads(
            response.content.replace("```json", "").replace("```", "").strip()
        )
    except Exception:
        return {
            "product_type": None,
            "search_query": None,
            "is_organic": None,
            "description": "Failed to parse model response",
        }


def make_tools(visitor_id: str):
    """
    Build the agent tool list for a single visitor/session.

    The shared product tools are global. Checkout, preferences, and order
    history are closures over visitor_id so browser sessions stay isolated.
    """
    import sqlite3

    @tool
    def checkout_for_visitor(product_id: int) -> str:
        """
        Place an order for the current visitor.

        Args:
            product_id: Product ID selected from the assistant's previous list.

        Returns:
            Confirmation message with the order ID, product name, and price.
        """
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT name, price FROM products WHERE id = ?", (product_id,)
            )
            row = cursor.fetchone()

            if not row:
                return f"Error: product with ID {product_id} not found"

            name, price = row
            cursor.execute(
                """
                INSERT INTO orders
                    (visitor_id, product_id, product_name, price, ordered_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (visitor_id, product_id, name, price),
            )
            order_id = cursor.lastrowid
            conn.commit()

        return f"Order #{order_id} confirmed! '{name}' with price for ${price}"

    @tool
    def save_preference_for_visitor(key: str, value: str) -> str:
        """
        Save or update a preference for the current visitor.

        Use this tool when a user explicitly states a preference that may
        personalize future recommendations, such as favorite brands, preferred
        categories, budget ranges, colors, sizes, or shopping interests.

        Args:
            key: Name of the preference, e.g. "favorite_brand" or "budget".
            value: Value of the preference, e.g. "Nike" or "2000".

        Returns:
            Confirmation message indicating the preference was saved.
        """
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO preferences
                    (visitor_id, preference_key, preference_value)
                VALUES (?, ?, ?)
                ON CONFLICT(visitor_id, preference_key)
                DO UPDATE SET
                    preference_value = excluded.preference_value,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (visitor_id, key, value),
            )
            conn.commit()

        return f"Saved preference: {key}={value}"

    @tool
    def get_preferences_for_visitor() -> str:
        """
        Retrieve all saved preferences for the current visitor.

        Use this tool when recommendations, search results, or product rankings
        should be personalized based on saved preferences.

        Returns:
            Preference keys and their values.
        """
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT preference_key, preference_value
                FROM preferences
                WHERE visitor_id = ?
                ORDER BY updated_at DESC
                """,
                (visitor_id,),
            )
            rows = cur.fetchall()

        if not rows:
            return "No preferences found."

        return "\n".join(f"{key}: {value}" for key, value in rows)

    @tool
    def get_order_history_for_visitor() -> str:
        """
        Retrieve the current visitor's recent purchase history.

        Use this tool when the user asks about previous orders, past purchases,
        buying habits, or when recommendations should use purchase history.

        Returns:
            Up to 20 recent orders including product name, price, and date.
        """
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT product_name, price, ordered_at
                FROM orders
                WHERE visitor_id = ?
                ORDER BY ordered_at DESC
                LIMIT 20
                """,
                (visitor_id,),
            )
            orders = cur.fetchall()

        if not orders:
            return "No previous orders found."

        return "\n".join(
            f"- {name} | ${price} | {ordered_at}"
            for name, price, ordered_at in orders
        )

    return [
        search_products,
        get_rating,
        analyze_product_image,
        checkout_for_visitor,
        save_preference_for_visitor,
        get_preferences_for_visitor,
        get_order_history_for_visitor,
    ]
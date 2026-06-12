import os
import sqlite3


DB_PATH = os.path.join(os.path.dirname(__file__), "store.db")


def get_average_rating(product_id: int) -> dict:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT AVG(rating), COUNT(*)
            FROM reviews
            WHERE product_id = ?
            """,
            (product_id,),
        )
        avg_rating, count = cursor.fetchone()

    avg_rating = round(avg_rating, 2) if avg_rating is not None else 0.0
    return {"product_id": product_id, "average_rating": avg_rating, "review_count": count}


def get_average_rating_list(product_id_list: list[int]) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        placeholder = ",".join(["?"] * len(product_id_list))
        cursor.execute(
            f"""
            SELECT product_id, AVG(rating), COUNT(*)
            FROM reviews
            WHERE product_id IN ({placeholder})
            GROUP BY product_id
            """,
            product_id_list,
        )
        rows = cursor.fetchall()

    result = []
    for product_id, avg_rating, count in rows:
        result.append({
            "product_id": product_id,
            "average_rating": round(avg_rating, 2) if avg_rating is not None else 0.0,
            "review_count": count,
        })

    return result


if __name__ == "__main__":
    # Single Product
    result = get_average_rating(1)
    print(result)

    # Multiple Products
    results = get_average_rating_list([1, 3, 5, 7])
    print(results[0])

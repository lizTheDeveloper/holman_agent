import os
import psycopg2

db_connection = psycopg2.connect(
    "postgresql://localhost:5432/holman_rag"
)
cursor = db_connection.cursor()

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

search_query = "lights"
print(f"Searching for documents related to: {search_query}")

search_embeddings = model.encode(search_query)

cursor.execute(
    "SELECT title, content FROM documents ORDER BY embeddings <=> cast(%s as vector(768)) LIMIT 5",
    (search_embeddings.tolist(),)
)
results = cursor.fetchall()
for title, content in results:
    print(f"Title: {title}\n")
cursor.close()
db_connection.close()
# import ollama
# embeddings = ollama.embeddings(model='nomic-embed-text', prompt='The sky is blue because of rayleigh scattering')

# print(embeddings)

import os
import psycopg2

db_connection = psycopg2.connect(
    "postgresql://localhost:5432/holman_rag"
)
cursor = db_connection.cursor()

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=False)

documents = []

for script in os.listdir('./documents'):
    with open(f'./documents/{script}', 'r') as file:
        content = file.read()
        document = {
            "title": script,
            "content": content,
            "embeddings": model.encode(content)
        }
        documents.append(document)
        cursor.execute(
            "INSERT INTO documents (title, content, embeddings) VALUES (%s, %s, %s)",
            (document["title"], document["content"], document["embeddings"].tolist())
        )

        db_connection.commit()
        print(f"Inserted document: {document['title']}")

cursor.close()
db_connection.close()

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

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

documents = []

for root, dirs, files in os.walk(os.path.expanduser('~/src')):
    for script in files:
        if not script.endswith('.py'):
            continue
        ## if any of these files are in a virtual environment, skip them
        if 'venv' in root or 'env' in root:
            continue
        if script.startswith('.'):
            continue
        if script.startswith('test_'):
            continue
        if script.startswith('setup'):
            continue
        with open(os.path.join(root, script), 'r') as file:
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

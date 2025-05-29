"""
Script to create tables for a graph RAG system for LLM agents.
Tables: documents, entities, relationships, observations.
"""
import psycopg2
import os

DB_URL = os.getenv("DB_URL", "postgresql://localhost:5432/holman_rag")

def create_tables():
    """Create the required tables for the graph RAG system."""
    commands = [
        '''
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            embeddings VECTOR(768)
        );
        ''',
        '''
        CREATE TABLE IF NOT EXISTS entities (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT
        );
        ''',
        '''
        CREATE TABLE IF NOT EXISTS relationships (
            id SERIAL PRIMARY KEY,
            from_entity INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            to_entity INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
            type TEXT NOT NULL
        );
        ''',
        '''
        CREATE TABLE IF NOT EXISTS observations (
            id SERIAL PRIMARY KEY,
            entity_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
            relationship_id INTEGER REFERENCES relationships(id) ON DELETE CASCADE,
            content TEXT NOT NULL
        );
        '''
    ]
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        for command in commands:
            cur.execute(command)
        conn.commit()
        cur.close()
        conn.close()
        print("All tables created successfully.")
    except Exception as e:
        print(f"Error creating tables: {e}")

if __name__ == "__main__":
    create_tables()

"""
FastAPI server to wrap PostgreSQL access for documents, entities, relationships, and observations tables.
"""
import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor

DB_URL = os.getenv("DB_URL", "postgresql://localhost:5432/holman_rag")

app = FastAPI(title="Holman RAG DB API")

def get_db_connection():
    """Create a new database connection."""
    try:
        conn = psycopg2.connect(DB_URL)
        return conn
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB connection error: {e}")

# --- Pydantic Models ---
class Document(BaseModel):
    title: str
    content: str
    embeddings: List[float]

class Entity(BaseModel):
    id: Optional[int] = None
    name: str
    type: Optional[str] = None

class Relationship(BaseModel):
    id: Optional[int] = None
    from_entity: int
    to_entity: int
    type: str

class Observation(BaseModel):
    id: Optional[int] = None
    entity_id: Optional[int] = None
    relationship_id: Optional[int] = None
    content: str

# --- Documents Endpoints ---
@app.post("/documents/", response_model=dict)
def add_document(doc: Document):
    """Add a new document."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO documents (title, content, embeddings)
            VALUES (%s, %s, %s)
            RETURNING id;
            """,
            (doc.title, doc.content, doc.embeddings)
        )
        doc_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return {"id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/", response_model=List[dict])
def list_documents():
    """List all documents (title, content, id)."""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT id, title, content FROM documents LIMIT 100;")
        docs = cur.fetchall()
        cur.close()
        conn.close()
        return docs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Entities Endpoints ---
@app.post("/entities/", response_model=dict)
def add_entity(entity: Entity):
    """Add a new entity."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO entities (name, type)
            VALUES (%s, %s)
            RETURNING id;
            """,
            (entity.name, entity.type)
        )
        entity_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return {"id": entity_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/entities/", response_model=List[dict])
def list_entities():
    """List all entities."""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM entities LIMIT 100;")
        entities = cur.fetchall()
        cur.close()
        conn.close()
        return entities
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Relationships Endpoints ---
@app.post("/relationships/", response_model=dict)
def add_relationship(rel: Relationship):
    """Add a new relationship between entities."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO relationships (from_entity, to_entity, type)
            VALUES (%s, %s, %s)
            RETURNING id;
            """,
            (rel.from_entity, rel.to_entity, rel.type)
        )
        rel_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return {"id": rel_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/relationships/", response_model=List[dict])
def list_relationships():
    """List all relationships."""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM relationships LIMIT 100;")
        rels = cur.fetchall()
        cur.close()
        conn.close()
        return rels
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Observations Endpoints ---
@app.post("/observations/", response_model=dict)
def add_observation(obs: Observation):
    """Add a new observation for an entity or relationship."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO observations (entity_id, relationship_id, content)
            VALUES (%s, %s, %s)
            RETURNING id;
            """,
            (obs.entity_id, obs.relationship_id, obs.content)
        )
        obs_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return {"id": obs_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/observations/", response_model=List[dict])
def list_observations():
    """List all observations."""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM observations LIMIT 100;")
        obs = cur.fetchall()
        cur.close()
        conn.close()
        return obs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Health Check ---
@app.get("/healthz")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

import os
import json
import datetime
# import pandas as pd
import numpy as np
import tiktoken
import psycopg2
from urllib.parse import urlparse
from pgvector.psycopg2 import register_vector
from openai import OpenAI

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = 'text-embedding-3-large'
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'
client = OpenAI(api_key= OPENAI_KEY)

def connect_to_db():
    secret = parse_postgres_connection_string()
    conn = psycopg2.connect(
        host=secret['host'],
        port=secret['port'],
        user=secret['username'],
        password=secret['password'],
        database=secret['dbname']
    )
    return conn

def parse_postgres_connection_string():
    """Parse the PostgreSQL connection string into its individual components."""
    result = urlparse(os.getenv("POSTGRES_URL"))

    username = result.username
    password = result.password
    host = result.hostname
    port = result.port
    database = result.path.lstrip('/')

    return {'host': host, 'port': port, 'username': username, 'password': password, 'dbname': database}


def truncate_text_tokens(text, encoding_name=EMBEDDING_ENCODING, max_tokens=EMBEDDING_CTX_LENGTH):
    """Truncate a string to have `max_tokens` according to the given encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.encode(text)[:max_tokens]

def get_embedding(text_to_embed):
    truncated_text = truncate_text_tokens(text_to_embed)
    
    response = client.embeddings.create(
            input=truncated_text,
            model="text-embedding-3-large", 
            dimensions=256,
            
        )
    return np.array(response.data[0].embedding)

def get_similar_works(conn, query_text, threshold, topK = 3):
    query_embedding = get_embedding(query_text)
    # Register pgvector extension
    register_vector(conn)
    cur = conn.cursor()
    # Get the top K most similar works
    cur.execute("SELECT work_id, (embedding <=> %s) as distance FROM mid.work_vector WHERE (embedding <=> %s) <= %s ORDER BY embedding <=> %s LIMIT %s", 
                            (query_embedding,query_embedding,threshold, query_embedding,topK,))
    top_works = cur.fetchall()
    cur.close()

    return [{'work_id': x[0], 'score': round(1-x[1], 6)} for x in top_works]

def get_similar_authors(conn, query_text, threshold, topK = 3):
    query_embedding = get_embedding(query_text)
    # Register pgvector extension
    register_vector(conn)
    cur = conn.cursor()
    # Get the top K most similar works
    cur.execute("SELECT author_id, (embedding <=> %s) as distance FROM mid.author_vector WHERE (embedding <=> %s) <= %s ORDER BY embedding <=> %s LIMIT %s", 
                            (query_embedding,query_embedding,threshold, query_embedding,topK,))
    top_authors = cur.fetchall()
    cur.close()
    return [{'author_id': x[0], 'score': round(1-x[1], 6)} for x in top_authors]
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

class Question(BaseModel):
    query: str
    k: int = 5

class Similarity(BaseModel):
    text:str
    similarity: float

class Answer(BaseModel):
    response: str
    top_k: list[Similarity]

embedder = SentenceTransformer('BAAI/bge-m3')
vectors = pd.DataFrame()

def read_vectors(filename):
    global vectors
    vectors = pd.read_csv(filename)

def embed(filename, destname='data/knowledge_vectors.csv'):
    global vectors
    knowledge_base = pd.read_csv(filename)
    knowledge_base['embeddings'] = knowledge_base['text'].apply(
        lambda x: embedder.encode(x, convert_to_tensor=True).tolist()
    )
    knowledge_base.to_csv(destname, index=False)
    vectors = knowledge_base

def get_top_k(query_text, k=5):
        query_vector = np.array(embedder.encode(query_text, convert_to_tensor=True).tolist())
        kb_vector = vectors.copy()
        kb_vector['embeddings'] = kb_vector['embeddings'].apply(np.array)
        embeddings = np.stack(kb_vector['embeddings'].values)
        similarities = np.dot(embeddings, query_vector)

        top_k_idx = similarities.argsort()[-k:][::-1]

        results = kb_vector.iloc[top_k_idx][['text']]
        results['similarity'] = similarities[top_k_idx]
        return results

def generate_response(query_text, k=5):
        top_k = get_top_k(query_text, k=k)
        context = "\n".join(top_k['text'].values)

        response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": "You are an assistant."},
                {"role": "user", "content": f"Answer the question in human-like formal language based on the following context: {context}\n\nQuestion: {query_text}"}
            ]
        )

        answer = Answer(
            response=response['message']['content'],
            top_k=[Similarity(text=row['text'], similarity=row['similarity']) for _, row in top_k.iterrows()]
        )

        return answer




app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


embed(filename='data/knowledge_base.csv')

@app.get("/")
def status():
    return {"status": "Running."}

@app.post("/ask")
def ask_question(question: Question) -> Answer:
    if not question.query:
        raise HTTPException(status_code=400, detail="Query text is required.")
    response = generate_response(question.query, k=question.k)
    return response

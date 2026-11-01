"""FastAPI serverless endpoint for Vercel. Index is generated BEFORE deployment."""
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
from core.rag import load_index, search, answer_question

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / 'index'
app = FastAPI(title='IronStore Knowledge API', version='2.0')
app.add_middleware(CORSMiddleware, allow_origins=[], allow_methods=['GET','POST'], allow_headers=['Content-Type'])

class Message(BaseModel):
    role: Literal['user','assistant']
    content: str = Field(min_length=1, max_length=12000)

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[Message] = Field(default_factory=list, max_length=12)
    category: str = Field(default='All', max_length=120)

@lru_cache(maxsize=1)
def index_data():
    return load_index(INDEX)

@app.get('/api/health')
def health():
    return {'status':'ok', 'index_ready':(INDEX/'records.json').exists() and (INDEX/'embeddings.npy').exists(), 'key_configured':bool(os.getenv('OPENAI_API_KEY'))}

@app.get('/api/categories')
def categories():
    try:
        payload, _ = index_data()
        return {'categories': ['All']+sorted({r['category'] for r in payload['records']}), 'chunks':len(payload['records'])}
    except (FileNotFoundError, ValueError):
        return {'categories':['All'], 'chunks':0}

@app.post('/api/chat')
def chat(req:ChatRequest):
    if not os.getenv('OPENAI_API_KEY'):
        raise HTTPException(503,'OPENAI_API_KEY is not configured on the server.')
    try:
        payload,vectors=index_data()
    except (FileNotFoundError,ValueError):
        raise HTTPException(503,'Document index missing. Run python scripts/build_index.py before deployment.')
    if req.category != 'All' and req.category not in {r['category'] for r in payload['records']}:
        raise HTTPException(400,'Unknown document category')
    try:
        client=OpenAI(api_key=os.environ['OPENAI_API_KEY'],timeout=25,max_retries=1)
        matches=search(req.question,payload,vectors,client,k=6,category=req.category)
        response=answer_question(req.question,[m.model_dump() for m in req.history[-6:]],matches,client)
        return {'answer':response,'sources':[{'name':m['source'],'page':m['page'],'score':round(m['score'],3),'legacy':m['legacy']} for m in matches]}
    except Exception as exc:
        # Do not leak SDK errors, prompts, or credentials to browser.
        raise HTTPException(502,'The AI service is temporarily unavailable. Please try again.') from exc

@app.get('/', include_in_schema=False)
def homepage():
    return FileResponse(ROOT / 'public' / 'index.html')

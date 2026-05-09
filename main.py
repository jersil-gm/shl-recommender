
import json
import numpy as np
import faiss
from groq import Groq
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
import gc

app = FastAPI()

print("Loading catalog...")
with open("shl_catalog.json") as f:
    assessments = json.load(f)
    
# Build simple text search index (no embedding model needed!)
# We use keyword matching + Groq for intelligence
print("Building keyword index...")
assessment_texts = []
for a in assessments:
    text = f"{a['name']} {a['description']} {' '.join(a['test_types'])}".lower()
    assessment_texts.append(text)

print("Connecting to Groq...")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)

gc.collect()
print(f"Ready! {len(assessments)} assessments loaded.")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool

SYSTEM_PROMPT = """You are an SHL assessment recommender assistant. You help hiring managers and recruiters find the right SHL assessments for their hiring needs.

STRICT RULES:
1. You ONLY discuss SHL assessments. Refuse any other topic politely.
2. Never recommend if you have NO job role at all
3. Ask ONE clarifying question at a time maximum
4. IMPORTANT: If you know the job role + ANY one detail (seniority OR skills OR context), set should_recommend to TRUE immediately
5. Never make up assessment names or URLs - only use what is provided
6. If asked to compare assessments, use only the descriptions provided
7. If user refines or adds requirements mid-conversation, update recommendations immediately

RESPONSE FORMAT:
You must ALWAYS respond with valid JSON and NOTHING else - no text before or after:
{
  "reply": "your conversational response here",
  "should_recommend": true or false,
  "end_of_conversation": true or false
}

WHEN TO SET should_recommend TRUE:
- You know job role + seniority level
- You know job role + key skills  
- You know job role + any context
- User says "junior/mid/senior [any role]"
- User refines existing recommendations
- User asks to add or remove a test type

WHEN TO SET should_recommend FALSE:
- No job role mentioned at all - ask what role
- Only a job role with zero other details - ask one follow-up

REFUSING OFF-TOPIC:
If asked anything not about SHL assessments, politely decline and ask what role they are hiring for.
"""

def keyword_search(query, top_k=10):
    """Simple keyword search - no embedding model needed"""
    query_words = query.lower().split()
    scores = []
    
    for i, text in enumerate(assessment_texts):
        score = sum(1 for word in query_words if word in text)
        scores.append((score, i))
    
    scores.sort(reverse=True)
    return [assessments[i] for score, i in scores[:top_k] if score > 0] or assessments[:top_k]


def build_smart_query(messages):
    conversation = " ".join([m.content for m in messages])
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """Extract the key hiring requirements from this conversation and write a short search query.
Include: job role, seniority, required skills, and any test types mentioned.
Return ONLY the search query, nothing else. Maximum 20 words."""
            },
            {"role": "user", "content": f"Conversation: {conversation}"}
        ],
        temperature=0,
        max_tokens=50
    )
    return response.choices[0].message.content.strip()


def get_catalog_context(messages, top_k=10):
    query = build_smart_query(messages)
    retrieved = keyword_search(query, top_k)

    context = "RELEVANT ASSESSMENTS FROM SHL CATALOG:\n\n"
    for a in retrieved:
        context += f"Name: {a['name']}\n"
        context += f"URL: {a['url']}\n"
        context += f"Test Types: {a['test_types']}\n"
        context += f"Remote Testing: {a['remote_testing']}\n"
        context += f"Description: {a['description'][:200]}\n"
        context += "-" * 40 + "\n"

    return context, retrieved, query


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    messages = request.messages
    catalog_context, retrieved, smart_query = get_catalog_context(messages)

    llm_messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + catalog_context}]
    for msg in messages:
        llm_messages.append({"role": msg.role, "content": msg.content})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=llm_messages,
        temperature=0.3,
        max_tokens=1000
    )

    raw = response.choices[0].message.content.strip()

    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()

    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end != 0:
        raw = raw[start:end]

    try:
        parsed = json.loads(raw)
        reply = parsed.get("reply", "I am sorry, could you repeat that?")
        should_recommend = parsed.get("should_recommend", False)
        end_of_conversation = parsed.get("end_of_conversation", False)
    except json.JSONDecodeError:
        reply = raw
        should_recommend = False
        end_of_conversation = False

    recommendations = []
    if should_recommend:
        results = keyword_search(smart_query, 10)
        for a in results:
            recommendations.append(Recommendation(
                name=a["name"],
                url=a["url"],
                test_type=",".join(a["test_types"])
            ))

    return ChatResponse(
        reply=reply,
        recommendations=recommendations,
        end_of_conversation=end_of_conversation
    )

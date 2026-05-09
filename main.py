
import json
import numpy as np
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

print("Connecting to Groq...")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)

# Build simple lookup structures
assessment_texts = []
for a in assessments:
    text = f"{a['name']} {a['description'][:300]} {' '.join(a['test_types'])}".lower()
    assessment_texts.append(text)

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
5. Never make up assessment names or URLs - only use what is provided to you in the catalog
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
- You know job role + seniority level → TRUE immediately
- You know job role + key skills → TRUE immediately
- You know job role + any context → TRUE immediately
- User says "junior/mid/senior [any role]" → TRUE immediately, no more questions
- User refines existing recommendations → TRUE immediately
- User asks to add or remove a test type → TRUE immediately

WHEN TO SET should_recommend FALSE:
- No job role mentioned at all - ask what role
- Only a vague statement with no role at all

REFUSING OFF-TOPIC:
If asked anything not about SHL assessments, politely decline.
"""

def keyword_search(query, top_k=15):
    """Keyword search returning top matches"""
    query_words = [w for w in query.lower().split() if len(w) > 2]
    scores = []
    
    for i, text in enumerate(assessment_texts):
        score = sum(2 if word in assessments[i]['name'].lower() else 1 
                   for word in query_words if word in text)
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
                "content": """Extract hiring requirements and write a search query.
Include: job role, seniority, skills, test types needed (personality/ability/knowledge).
Return ONLY the query. Max 20 words."""
            },
            {"role": "user", "content": f"Conversation: {conversation}"}
        ],
        temperature=0,
        max_tokens=50
    )
    return response.choices[0].message.content.strip()


def get_best_recommendations(messages, query, top_k=10):
    """Use LLM to pick best assessments from catalog for this query"""
    
    # Get candidates via keyword search
    candidates = keyword_search(query, top_k=20)
    
    # Format candidates for LLM
    catalog_text = ""
    for i, a in enumerate(candidates):
        catalog_text += f"{i}. {a['name']} | Types: {','.join(a['test_types'])} | {a['description'][:150]}\n"
    
    conversation = " ".join([m.content for m in messages])
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"""You are selecting the best SHL assessments for a hiring need.
Given these candidate assessments (numbered 0-19):
{catalog_text}

Pick the best 5-10 assessments for this hiring need. 
Return ONLY a JSON array of numbers like: [0, 3, 5, 7, 2]
No explanation, just the array."""
            },
            {"role": "user", "content": f"Hiring need: {conversation}"}
        ],
        temperature=0,
        max_tokens=100
    )
    
    raw = response.choices[0].message.content.strip()
    
    try:
        # Extract array from response
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start != -1:
            indices = json.loads(raw[start:end])
            selected = [candidates[i] for i in indices if i < len(candidates)]
            return selected[:10]
    except:
        pass
    
    return candidates[:10]


def get_catalog_context(messages, top_k=10):
    query = build_smart_query(messages)
    retrieved = keyword_search(query, top_k=10)

    context = "RELEVANT ASSESSMENTS FROM SHL CATALOG:\n\n"
    for a in retrieved:
        context += f"Name: {a['name']}\n"
        context += f"URL: {a['url']}\n"
        context += f"Test Types: {a['test_types']}\n"
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

    llm_messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + catalog_context}
    ]
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
        best = get_best_recommendations(messages, smart_query, top_k=10)
        for a in best:
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

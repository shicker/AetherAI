# app/routers/chat.py: API routes for chat functionality

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.models import generate_response
from app.database import log_conversation
from app.main import model, tokenizer  # Imported from main

router = APIRouter()

class ChatRequest(BaseModel):
    user_id: str
    user_input: str
    conversation_history: list = []  # Optional history for context

@router.post("/chat")
def chat(request: ChatRequest):
    try:
        response = generate_response(model, tokenizer, request.user_input, request.conversation_history)
        log_conversation(request.user_id, request.user_input, response)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

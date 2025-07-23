from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model(model_path: str):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    return model, tokenizer

def generate_response(model, tokenizer, user_input, conversation_history):
    prompt = f"Conversation history: {conversation_history}\nUser: {user_input}\nAether AI: As an empathetic AI, respond supportively and diagnose potential depression symptoms (e.g., low mood, hopelessness)."
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if any(word in user_input.lower() for word in ["sad", "hopeless"]):
        response += "\n[Diagnosis Insight: This may indicate symptoms of depression. Please consult a professional.]"
    return response

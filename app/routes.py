from flask import Blueprint, request, jsonify, current_app
from app.models import generate_response
from app.database import log_conversation

main_routes = Blueprint('main_routes', __name__)

# Load model and tokenizer once at startup
import config
from app.models import load_model
model, tokenizer = load_model(config.MODEL_PATH)

from app.database import init_db
init_db(config.MONGODB_URI, config.DB_NAME)

@main_routes.route("/")
def index():
    return jsonify({"message": "Welcome to Aether AI - Flask version!"})

@main_routes.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_id = data.get("user_id")
    user_input = data.get("user_input")
    conversation_history = data.get("conversation_history", [])
    response = generate_response(model, tokenizer, user_input, conversation_history)
    log_conversation(user_id, user_input, response)
    return jsonify({"response": response})

from pymongo import MongoClient
import datetime

client = None
db = None

def init_db(uri: str, db_name: str):
    global client, db
    client = MongoClient(uri)
    db = client[db_name]

def log_conversation(user_id, user_input, ai_response):
    collection = db["conversations"]
    log = {
        "user_id": user_id,
        "timestamp": datetime.datetime.now(),
        "user_input": user_input,
        "ai_response": ai_response
    }
    collection.insert_one(log)

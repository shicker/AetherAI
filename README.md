# Aether AI

## Dev Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Preprocess and train data as outlined above.
3. Run Flask app: `python run.py`
4. For production: `docker build -t aether-ai .` then `docker run -p 8000:8000 aether-ai`

## API Usage
- POST `/api/chat` with JSON: `{ "user_id": "123", "user_input": "..." }`
- Returns: `{ "response": "..." }`

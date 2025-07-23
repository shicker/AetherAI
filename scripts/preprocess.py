# scripts/preprocess.py: Data preprocessing for training

import pandas as pd
import PyPDF2
import json
import config

def extract_cornell_conversations():
    """Extract and filter conversations from Cornell corpus."""
    # Load files (simplified; assumes files are in config.CORNELL_DATA_PATH)
    lines = pd.read_csv(config.CORNELL_DATA_PATH + "movie_lines.txt", sep=" +++\\$+++ ", engine="python", names=["lineID", "charID", "movieID", "charName", "text"])
    convs = pd.read_csv(config.CORNELL_DATA_PATH + "movie_conversations.txt", sep=" +++\\$+++ ", engine="python", names=["char1", "char2", "movieID", "lineIDs"])
    
    # Filter for drama genres (example; expand as needed)
    # ... (Add metadata filtering logic here)
    
    # Create input-response pairs
    pairs = []  # List of [input, response]
    for _, row in convs.iterrows():
        line_ids = eval(row["lineIDs"])
        for i in range(len(line_ids) - 1):
            input_text = lines[lines["lineID"] == line_ids[i]]["text"].values[0]
            response_text = lines[lines["lineID"] == line_ids[i+1]]["text"].values[0]
            pairs.append({"input": input_text, "response": response_text})
    
    return pairs

def extract_pdf_text(pdf_path):
    """Extract text from PDFs and create synthetic Q&A for depression knowledge."""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    
    # Synthetic chunking into Q&A (simplified; use NLP for better pairs)
    chunks = text.split("\n\n")  # Basic paragraph splitting
    pairs = [{"input": "What is depression?", "response": chunks[0]}]  # Example
    # Expand with more logic (e.g., keyword-based Q&A generation)
    
    return pairs

def combine_datasets():
    cornell_pairs = extract_cornell_conversations()
    pdf_pairs = extract_pdf_text(config.PDF_BOOKS_PATH)
    combined = cornell_pairs + pdf_pairs
    with open("data/combined_dataset.json", "w") as f:
        json.dump(combined, f)

if __name__ == "__main__":
    combine_datasets()

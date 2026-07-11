# Local-RAG-Chatbot-Retrieval-Augmented-Generation-
A lightweight, fully local RAG system built from scratch using Python, NumPy, and Ollama. Instead of relying on massive cloud APIs, this system runs entirely on a CPU, proving that you don't need expensive GPUs to solve the LLM "Hallucination" problem.

# How It Works
The system uses a 3-step pipeline:

Ingestion: Reads .txt files from a local folder and converts them into 768-dimensional vector embeddings using nomic-embed-text via Ollama.
Retrieval: When a user asks a question, the system converts the question into an embedding and uses Cosine Similarity (calculated manually via NumPy) to find the most semantically similar document.
Generation: It feeds the retrieved document as strict context to a local LLM (qwen2.5:3b), forcing it to answer based only on the provided text, completely eliminating hallucinations.
# Tech Stack
Math & Logic: numpy (Manual Cosine Similarity calculation)
API Integration: urllib (Direct HTTP calls to Ollama's /api/embeddings)
LLM Generation: ollama (Python bindings for chat generation)
Data Handling: pathlib (Local file ingestion)
# How to Run
Ensure Ollama is running on your machine.
Pull the required models:
ollama pull nomic-embed-textollama pull qwen2.5:3b
Add your .txt documents to the knowledge/ folder.
Run the script:
```bash
python rag_chatbot.py
```
# Key Learnings
Understanding why LLMs hallucinate (lack of grounding context).
How to calculate vector similarity mathematically without high-level libraries.
Building a modular pipeline (Separating Retrieval from Generation).

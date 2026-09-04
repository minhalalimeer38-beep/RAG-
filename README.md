# Secure Simple PDF RAG Assistant

A deployment-ready Retrieval-Augmented Generation application using a PDF as the knowledge source.

## Architecture

```text
PDF
 ↓
PyPDF Extraction
 ↓
Text Chunking
 ↓
HuggingFace Embeddings
 ↓
Chroma Vector Database
 ↓
User Question
 ↓
Similarity Retrieval
 ↓
Relevant PDF Context
 ↓
Gemini 2.5 Flash
 ↓
Grounded Answer
```

## Features

- Upload any text-based PDF
- Extract text page by page
- Split text into overlapping chunks
- Generate semantic embeddings with Hugging Face
- Store embeddings in ChromaDB
- Retrieve the top 4 relevant chunks
- Generate an answer with Gemini
- Display retrieved source pages
- Prevent the app from using outside knowledge when answering
- API key is stored in Streamlit Secrets rather than in source code

## API Key Security

DO NOT put your Gemini API key in `app.py`.

DO NOT commit the API key to GitHub.

For Streamlit Community Cloud:

1. Deploy the repository.
2. Open the deployed app's settings.
3. Open the **Secrets** section.
4. Add:

```toml
GOOGLE_API_KEY = "YOUR_REAL_GEMINI_API_KEY"
```

5. Save the secret.
6. Restart/redeploy the application.

The application reads the secret using:

```python
api_key = st.secrets["GOOGLE_API_KEY"]
```

The key is not shown in the UI and is not included in the GitHub repository.

### Local testing

For local testing, create:

```text
.streamlit/secrets.toml
```

with:

```toml
GOOGLE_API_KEY = "YOUR_REAL_GEMINI_API_KEY"
```

The `.gitignore` file already excludes `.streamlit/secrets.toml`, so it should not be committed to GitHub.

## Important Security Notes

- Treat the API key like a password.
- Never paste it into `app.py`, `README.md`, screenshots, GitHub issues, or public repositories.
- Do not share your API key with users.
- If the key is accidentally exposed, revoke/rotate it immediately from the provider's API-key management page.
- A public app can still consume your API quota because requests are processed using your key. For a public production application, add authentication and/or rate limiting.

## Requirements Covered

1. Load PDF — `pypdf`
2. Extract text — `PdfReader`
3. Split text — `RecursiveCharacterTextSplitter`
4. Generate embeddings — Hugging Face `all-MiniLM-L6-v2`
5. Store embeddings — ChromaDB
6. User query — Streamlit text input
7. Retrieve relevant information — Chroma retriever
8. Generate final answer — Gemini 2.5 Flash
9. Multiple questions — users can ask multiple questions in the interface

## Deployment

### GitHub

Upload these files:

```text
app.py
requirements.txt
README.md
.gitignore
.streamlit/config.toml
```

Do NOT upload:

```text
.streamlit/secrets.toml
```

### Streamlit Community Cloud

Set:

```text
Main file: app.py
```

Then configure the secret:

```toml
GOOGLE_API_KEY = "YOUR_REAL_GEMINI_API_KEY"
```

## Suggested Test Questions

Use questions whose answers are actually present in your selected PDF.

Example:

- What is the main topic of the document?
- What is data science?
- What is data analysis?
- What is machine learning?
- What role does Python play in data science?

Also test an unrelated question to check grounded behavior.

## Project Structure

```text
simple-rag-pdf-secure/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── config.toml
```

import streamlit as st
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 PDF RAG Assistant")
st.caption("Ask questions using information retrieved from your uploaded PDF.")

# API key is read from Streamlit Secrets.
# It is NEVER displayed to the user and is NOT stored in GitHub.
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except Exception:
    st.error(
        "GOOGLE_API_KEY is not configured. Add it to Streamlit Cloud Secrets "
        "before using the application."
    )
    st.stop()


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def extract_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": uploaded_file.name,
                        "page": page_number,
                    },
                )
            )

    return documents, len(reader.pages)


def build_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=load_embeddings(),
        collection_name="pdf_rag_collection",
    )

    return vectorstore, chunks


def answer_question(vectorstore, question):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0,
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join(
        f"[Page {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
        for doc in retrieved_docs
    )

    prompt = ChatPromptTemplate.from_template(
        """You are a PDF question-answering assistant.

Use ONLY the retrieved context from the uploaded PDF to answer the question.
Do not use outside knowledge.
If the answer is not supported by the retrieved context, say:
"I could not find the answer in the provided PDF."

Give a clear and concise answer. Do not invent facts.

Retrieved Context:
{context}

Question:
{question}

Answer:"""
    )

    response = llm.invoke(
        prompt.format_messages(
            context=context,
            question=question,
        )
    )

    return response.content, retrieved_docs


uploaded_file = st.file_uploader(
    "Upload a PDF knowledge source",
    type=["pdf"],
)

if uploaded_file is None:
    st.info("Upload a PDF to start the RAG pipeline.")
    st.markdown(
        """
        **Example questions**
        - What is the main topic of this document?
        - What are the key concepts discussed?
        - Explain one important concept from the PDF.
        """
    )
    st.stop()


file_signature = f"{uploaded_file.name}-{uploaded_file.size}"

if st.session_state.get("file_signature") != file_signature:
    with st.spinner("Extracting text and building the vector database..."):
        try:
            documents, total_pages = extract_pdf(uploaded_file)

            if not documents:
                st.error(
                    "No extractable text was found. This may be a scanned PDF."
                )
                st.stop()

            vectorstore, chunks = build_vectorstore(documents)

            st.session_state.file_signature = file_signature
            st.session_state.vectorstore = vectorstore
            st.session_state.chunks = chunks
            st.session_state.total_pages = total_pages

        except Exception as exc:
            st.error(f"Could not process the PDF: {exc}")
            st.stop()


vectorstore = st.session_state.vectorstore
chunks = st.session_state.chunks

col1, col2, col3 = st.columns(3)

col1.metric("PDF Pages", st.session_state.total_pages)
col2.metric("Text Chunks", len(chunks))
col3.metric("Retrieved Chunks", 4)

st.success("PDF indexed successfully.")

st.divider()

question = st.text_input(
    "Ask a question about the PDF",
    placeholder="e.g. What is data science?",
)

if st.button("Get Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving relevant information and generating answer..."):
            try:
                answer, retrieved_docs = answer_question(
                    vectorstore,
                    question.strip(),
                )

                st.subheader("Answer")
                st.write(answer)

                st.subheader("Retrieved Sources")

                for i, doc in enumerate(retrieved_docs, start=1):
                    page = doc.metadata.get("page", "N/A")

                    with st.expander(f"Source {i} — Page {page}"):
                        st.write(doc.page_content)

            except Exception as exc:
                st.error(f"Could not generate the answer: {exc}")

st.divider()

st.caption(
    "RAG pipeline: PDF → text extraction → chunking → embeddings → "
    "Chroma retrieval → Gemini answer generation."
)

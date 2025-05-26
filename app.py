import os
import streamlit as st
import pickle
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer


# Fixed embeddings class with __call__ method for LangChain compatibility
class SentenceTransformerEmbeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

    def __call__(self, text):
        # Make embeddings object callable (required by LangChain FAISS)
        return self.embed_query(text)


# Streamlit UI setup
st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_st.pkl"

main_placeholder = st.empty()

embeddings = SentenceTransformerEmbeddings()

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=[u for u in urls if u.strip() != ""])
    main_placeholder.text("Data Loading...Started...✅✅✅")
    try:
        data = loader.load()
    except Exception as e:
        st.error(f"Error loading URLs: {e}")
        st.stop()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","], chunk_size=1000
    )
    main_placeholder.text("Text Splitting...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    if docs:
        vectorstore = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding Vector Building...✅✅✅")
        time.sleep(1)

        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)

        st.success("Vectorstore built and saved successfully!")
    else:
        main_placeholder.text("❌ No valid content found at the provided URLs.")
        st.warning(
            "No documents were loaded from the URLs. Please check the links and try again."
        )

query = main_placeholder.text_input("Ask a question about the processed news articles:")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )
        docs = retriever.get_relevant_documents(query)

        st.header("Top Matching Document Chunks:")
        for i, doc in enumerate(docs):
            st.write(f"---\n**Chunk {i+1}:**\n{doc.page_content}\n")

    else:
        st.warning("Please process URLs first to build the knowledge base.")

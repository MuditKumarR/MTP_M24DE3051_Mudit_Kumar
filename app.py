import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --- 1. Setup & Config ---
load_dotenv()
st.set_page_config(page_title="Kisan Sahayak AI", page_icon="🌾", layout="wide")

DB_FAISS_PATH = 'vectorstore/db_faiss'
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

# --- 2. Backend Logic ---
@st.cache_resource
def get_rag_chain():
    # Check for API Key
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("⚠️ Error: GOOGLE_API_KEY not found. Check your .env file.")
        return None

    # Load Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    
    # Load Vector DB
    if not os.path.exists(DB_FAISS_PATH):
        st.error("⚠️ Vector Database not found. Please run 'python ingest_data.py' first.")
        return None
        
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Initialize Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3, # Low temp for factual accuracy
        convert_system_message_to_human=True
    )

    # Prompt Template
    template = """
    You are 'Kisan Sahayak', an expert agricultural advisor for Indian farmers. 
    Use the context below to answer the farmer's question.
    
    Guidelines:
    1. Answer primarily in the language of the question (Hindi/English).
    2. If the answer is not in the context, say "I do not have official information on this."
    3. Cite the source document name where possible.
    
    Context: {context}
    Question: {question}
    
    Answer:
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=['context', 'question']
    )

    # Retrieval Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    
    return qa_chain

# --- 3. Frontend Interface ---
def main():
    st.title("🌾 Kisan Sahayak: Rural Advisory System")
    
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This system uses RAG to fetch accurate information from agricultural handbooks.
        
        **Student:** Mudit Kumar  
        **Roll:** M24DE3051  
        """)
        st.divider()
        st.info("System Status: Ready")

    # Chat Session
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("Ask about crop diseases, schemes, or weather..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            chain = get_rag_chain()
            if chain:
                with st.spinner("Consulting knowledge base..."):
                    res = chain.invoke(prompt)
                    answer = res['result']
                    sources = res['source_documents']
                    
                    st.markdown(answer)
                    
                    # Citations
                    with st.expander("📚 Sources Cited"):
                        for doc in sources:
                            st.caption(f"Source: {doc.metadata.get('source', 'Unknown')} | Page: {doc.metadata.get('page', 'N/A')}")
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()

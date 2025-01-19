import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if api_key:
    genai.api_key = api_key
else:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

# Constants
LOCAL_PDF_FOLDER_PATH = "PDFs"
FAISS_INDEX_PATH = "faiss_index"


# Function to extract text from PDF files
def extract_text_from_pdfs(pdf_files):
    text = ""
    for pdf in pdf_files:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading {pdf}: {e}")
    return text


# Function to split text into manageable chunks
def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


# Function to create and save a FAISS vector store
def create_faiss_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)


# Function to load the conversational QA chain
def get_qa_chain():
    prompt_template = """
    You are a highly knowledgeable and precise medical assistant. Use the provided context from the medical report to answer the question in a clear, concise, and accurate manner. 

- If the information is directly available in the context, provide a detailed answer.
- If the information is partially available, clarify what is missing and answer based on the available details.
- If the answer is not available in the context, respond with:
  "The answer is not available in the context."

Always ensure your response adheres to the following:
1. Use medical terminology appropriately but keep it understandable for the user.
2. Provide additional insights or clarifications based on the data when relevant.
3. Avoid guessing or providing incorrect information under any circumstances.

### Context:
{context}

### Question:
{question}

### Detailed Answer:

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


# Function to handle user input and generate responses
def handle_user_query(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        # Load the FAISS vector store
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

        docs = vector_store.similarity_search(user_question)

        # Generate response using the QA chain
        chain = get_qa_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        st.write("Reply:", response.get("output_text", "No response generated."))
    except Exception as e:
        st.error(f"Error handling the query: {e}")


# Function to load PDFs from a specified folder
def load_pdfs(folder_path):
    pdf_files = []
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".pdf"):
            pdf_files.append(os.path.join(folder_path, file_name))
    return pdf_files


# Main Streamlit App
# Main Streamlit App
def main():
    st.set_page_config(page_title="Medical Chatbot", page_icon="🩺")
    st.header("🩺 Your Medical Assistant Chatbot")

    # Add an interactive and user-friendly sidebar
    with st.sidebar:
        st.title("Menu")
        st.info("Upload your medical records to get detailed insights and answers.")

        uploaded_pdfs = st.file_uploader(
            "Upload Medical Records (PDFs)", type="pdf", accept_multiple_files=True
        )

        if uploaded_pdfs:
            pdf_files = uploaded_pdfs
        else:
            pdf_files = []

        if st.button("Process PDFs"):
            if not pdf_files:
                st.warning("Please upload at least one PDF file to process.")
            else:
                with st.spinner("Processing your medical records..."):
                    raw_text = extract_text_from_pdfs(pdf_files)
                    text_chunks = split_text_into_chunks(raw_text)
                    create_faiss_vector_store(text_chunks)
                    st.success("Your records have been processed. Ask your questions below!")

    # Input field for user query
    user_question = st.text_input("Ask a Question About Your Medical Records", help="e.g., What medicines are prescribed?")

    # Display placeholders for interactivity
    if user_question:
        st.subheader("Your Query:")
        st.write(f"❓ {user_question}")

        with st.spinner("Fetching the best answer..."):
            handle_user_query(user_question)


    # Add some decorative elements for better engagement
    st.markdown(
        """
        ---
        **Tips:**
        - Upload multiple files for better results.
        - Ask detailed questions like:
          - "What is the diagnosis in this document?"
          - "What are the prescribed medications?"
          - "What does a high cholesterol level mean?"
        ---
        """
    )


if __name__ == "__main__":
    main()


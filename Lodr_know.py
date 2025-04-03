# --- Langchain & Google AI Imports ---
# ... (keep existing imports)
from langchain.memory import ConversationBufferMemory # <-- Add Memory
from langchain.chains import ConversationalRetrievalChain # <-- Add Conversational Chain

# --- Core Imports ---
import streamlit as st
import os
import requests
import shutil
import tempfile
import pickle
from io import BytesIO
from PIL import Image

# --- PDF and Text Processing ---
from PyPDF2 import PdfReader
import fitz  # PyMuPDF - Added from local
import re    # Added from local

# --- Langchain & Google AI Imports ---
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter # Keep import
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai # Keep genai import if needed elsewhere

# --- Configuration ---
# Set wide layout MUST be the first Streamlit command
st.set_page_config(layout="wide")

# --- CUSTOMIZATION ---
PRIMARY_COLOR = "#C8102E"   # Red from the logo
SECONDARY_COLOR = "#223A70" # Dark blue from the logo
BACKGROUND_COLOR = "#FFFFFF" # White background
TEXT_COLOR = "#000000" # Black text

# Apply custom theme
st.markdown(
    f"""
    <style>
    body {{
        color: {TEXT_COLOR};
        background-color: {BACKGROUND_COLOR};
    }}
    .stApp {{
        background-color: {BACKGROUND_COLOR};
    }}
    .stButton>button {{
        color: white;
        background-color: {PRIMARY_COLOR};
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
    }}
    .stButton>button:hover {{
        background-color: {SECONDARY_COLOR};
    }}
    .chat-message {{
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }}
    .chat-message.user {{
        background-color: #e6f3ff;
        border-left: 5px solid {PRIMARY_COLOR};
    }}
    .chat-message.assistant {{
        background-color: #f0f2f6;
        border-left: 5px solid {SECONDARY_COLOR};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Session State ---
if 'messages' not in st.session_state:
    st.session_state.messages = []
# Initialize conversation memory
if 'memory' not in st.session_state:
    # return_messages=True ensures the history is stored as ChatMessage objects
    # memory_key='chat_history' links this memory to the prompt variable
    st.session_state.memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True, output_key='answer'
    )

# --- Global Variables (from Secrets) ---
# Moved API key retrieval to main block for clarity
google_api_key = None

# --- Helper Functions ---

def display_logo():
    """Displays the logo and title."""
    logo_url = "https://raw.githubusercontent.com/RahulGandhi128/LODR_knowledge/adfb73b9dc65553d5ffe7827f75d43cd0636ca0c/image001.png"

    col1, col2 = st.columns([1, 4])
    with col1:
        try:
            response = requests.get(logo_url, stream=True)
            response.raise_for_status()
            image = Image.open(response.raw) # Use response.raw to directly read image
            st.image(image, use_container_width=True)
        except requests.exceptions.RequestException as e:
            st.warning(f"Error loading company logo: {e}")
        except Exception as img_e: # Catch potential Pillow errors
             st.warning(f"Error opening logo image: {img_e}")

    # Added Title and Description from local version
    with col2:
       st.title("LODR Regulations Assistant")
       st.write("Ask questions about LODR regulations and procedures.")


def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file (BytesIO or path)."""
    # This function remains largely the same, useful if allowing PDF uploads later
    try:
        if isinstance(pdf_file, BytesIO): # If already a file-like object
            pdfReader = PdfReader(pdf_file)
        else: # If it's a file path (less likely in web context unless saving temp)
            pdfReader = PdfReader(pdf_file)

        all_text = ""
        for page in pdfReader.pages:
            text = page.extract_text()
            if text:
                # Attempt to encode/decode to handle potential odd characters
                try:
                    all_text += text.encode('ascii', 'ignore').decode('ascii') + "\n"
                except Exception: # Fallback if encoding fails
                    all_text += text + "\n" # Add raw text if encoding fails
        return all_text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

# --- New Functions from Local Code (Metadata Extraction) ---

def extract_headings(pdf_path):
    """
    Extracts potential headings from a PDF using PyMuPDF (fitz).
    Requires a file path.
    """
    headings = []
    try:
        doc = fitz.open(pdf_path) # Requires a file path
        for page_num, page in enumerate(doc):
            blocks = page.get_text("blocks")
            for block in blocks:
                # block format: (x0, y0, x1, y1, "text", block_no, block_type)
                if block[6] == 0: # Check if it's a text block
                    text = block[4].strip()
                    # Basic patterns for headings (customize as needed)
                    # Pattern 1: Numbered headings (e.g., 1. Introduction, 2.1. Subsection)
                    if re.match(r"^\d+(\.\d+)*\s+[A-Z][a-zA-Z]+", text):
                        headings.append((text, page_num))
                    # Pattern 2: All caps headings (potential)
                    elif len(text) > 5 and text.isupper() and not text.isdigit(): # Avoid short caps or numbers
                        headings.append((text, page_num))
                    # Add more patterns if needed (e.g., bold text checks - more complex)
        doc.close()
    except Exception as e:
        print(f"Warning: Could not extract headings from {pdf_path}. Error: {e}")
        # Continue without headings if extraction fails
    return headings

def get_text_chunks_with_metadata(pdf_path, chunk_size=2000, chunk_overlap=200):
    """
    Splits PDF text into chunks and adds metadata (source, section, page).
    Requires PyMuPDF (fitz) and a file path.
    """
    all_chunks = []
    try:
        doc = fitz.open(pdf_path)
        headings = extract_headings(pdf_path) # Extract headings first
        current_section = "General" # Default section

        # Configure the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""], # Added more separators
            length_function=len
        )

        full_text_for_splitting = ""
        page_map = [] # To map character index back to page number

        # Concatenate text and build page map
        for page_num, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                start_index = len(full_text_for_splitting)
                full_text_for_splitting += text + "\n\n" # Add separator between pages
                end_index = len(full_text_for_splitting)
                page_map.append({"page": page_num + 1, "start": start_index, "end": end_index})

        doc.close() # Close the document after extracting text

        # Split the entire document text at once
        split_texts = text_splitter.split_text(full_text_for_splitting)

        # Process each chunk to add metadata
        char_offset = 0
        for split_text in split_texts:
            chunk_page = 1 # Default page
            # Determine the page number for the chunk's start
            for pm in page_map:
                if char_offset >= pm["start"] and char_offset < pm["end"]:
                    chunk_page = pm["page"]
                    break

            # Determine the section based on the closest preceding heading
            current_section = "General" # Reset for each chunk initially
            for heading_text, heading_page_num in reversed(headings):
                 # Check if heading page is before or on the current chunk's page
                 # This logic might need refinement depending on heading density
                 if heading_page_num < chunk_page: # Check page number (0-based vs 1-based)
                      current_section = heading_text
                      break # Found the most recent relevant heading

            # Create chunk dictionary
            chunk = {
                "text": split_text,
                "metadata": {
                    "source": os.path.basename(pdf_path), # Use filename as source
                    "section": current_section,
                    "page": chunk_page
                }
            }
            all_chunks.append(chunk)
            char_offset += len(split_text) # Rough estimate for next chunk start

    except Exception as e:
        print(f"Error chunking with metadata for {pdf_path}: {e}")
        # Fallback: return a single chunk if metadata extraction fails
        try:
            doc = fitz.open(pdf_path)
            full_fallback_text = ""
            for page in doc:
                full_fallback_text += page.get_text("text")
            doc.close()
            return [{
                "text": full_fallback_text,
                "metadata": {"source": os.path.basename(pdf_path), "section": "General", "page": 1}
            }]
        except Exception as fallback_e:
             print(f"Fallback text extraction failed: {fallback_e}")
             return [] # Return empty if even fallback fails

    return all_chunks

# --- Updated get_text_chunks Function ---

def get_text_chunks(text, chunking_enabled=True, pdf_path=None):
    """
    Splits text into chunks. If pdf_path is provided and chunking is enabled,
    uses the metadata-aware chunking function. Otherwise, uses basic chunking.
    Returns Langchain Document objects.
    """
    # This function is primarily for offline index creation or potential future
    # file upload features. It's not directly used by check_compliance when
    # loading the index from GitHub.
    if chunking_enabled:
        if pdf_path and os.path.exists(pdf_path):
            print(f"\n--- Using Metadata Chunking for PDF: {pdf_path} ---")
            # Use the new metadata-aware chunking for PDFs
            # This function now returns a list of dictionaries
            chunks_with_metadata = get_text_chunks_with_metadata(pdf_path)

            if not chunks_with_metadata:
                 print("Warning: Metadata chunking returned no chunks.")
                 return []

            # Debug output
            print("\n--- DEBUG: Generated Chunks with Section Metadata ---")
            for i, chunk_dict in enumerate(chunks_with_metadata[:3]): # Print first 3
                print(f"Chunk {i+1}: {chunk_dict.get('text', '')[:150]}...")
                print(f"Metadata: {chunk_dict.get('metadata', {})}\n")
            print(f"Total chunks created: {len(chunks_with_metadata)}")

            # Convert list of dictionaries to Langchain Document objects
            # We need a splitter instance mainly for the create_documents method structure
            # The actual splitting happened in get_text_chunks_with_metadata
            doc_creator_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000, # Parameters here are less critical now
                chunk_overlap=200,
                separators=["\n\n", "\n", " "], # Keep standard separators
                length_function=len
            )

            try:
                documents = doc_creator_splitter.create_documents(
                    texts=[chunk["text"] for chunk in chunks_with_metadata],
                    metadatas=[chunk["metadata"] for chunk in chunks_with_metadata]
                )
                print(f"Successfully created {len(documents)} Langchain document objects")
                return documents
            except Exception as e:
                 print(f"Error creating Langchain documents from metadata chunks: {e}")
                 # Fallback to basic chunking of the extracted text if creation fails
                 text = "\n".join([chunk["text"] for chunk in chunks_with_metadata]) # Recombine text
                 pdf_path = None # Force basic chunking below

        # Fallback to basic chunking if no pdf_path or if metadata chunking failed
        print("\n--- Using Basic Text Chunking ---")
        basic_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""], # Use broad separators
            length_function=len
        )
        # Basic chunking requires only the text string
        documents = basic_splitter.create_documents(
            texts=[text], # Process the single text block
            metadatas=[{"source": "LODR_documentation", "section": "General", "page": 1}] # Basic metadata
        )
        print(f"Basic chunking created {len(documents)} documents.")
        return documents

    else: # Chunking disabled
        print("Chunking disabled. Returning text as a single Document.")
        # Return the text as a single Langchain Document
        from langchain.docstore.document import Document # Local import
        return [Document(page_content=text, metadata={"source": "LODR_documentation", "section": "General", "page": 1})]


# --- Vector Store Loading (from GitHub) ---

def load_vector_store_from_github():
    """Downloads and loads the FAISS vector store from GitHub."""
    global google_api_key # Ensure API key is accessible
    if not google_api_key:
        st.error("Google API Key not configured. Cannot initialize embeddings.")
        return None

    try:
        embeddings = GoogleGenerativeAIEmbeddings(google_api_key=google_api_key, model="models/embedding-001")
    except Exception as e:
        st.error(f"Failed to initialize Google Embeddings: {e}")
        return None

    # Ensure FAISS library is available
    try:
        import faiss
    except ImportError:
        st.error("FAISS library not found. Cannot load index. Please ensure 'faiss-cpu' or 'faiss-gpu' is installed.")
        return None


    # --- IMPORTANT: Update these URLs to your actual GitHub raw file paths ---
    faiss_url = "https://raw.githubusercontent.com/RahulGandhi128/LODR_knowledge/main/faiss_index_LODR/index.faiss"
    pkl_url = "https://raw.githubusercontent.com/RahulGandhi128/LODR_knowledge/main/faiss_index_LODR/index.pkl"
    # --- ---

    vector_store = None
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_faiss_path = os.path.join(tmpdir, "index.faiss")
            local_pkl_path = os.path.join(tmpdir, "index.pkl")

            print(f"Downloading FAISS index from {faiss_url} to {local_faiss_path}")
            with requests.get(faiss_url, stream=True, timeout=60) as response: # Added timeout
                response.raise_for_status()
                with open(local_faiss_path, 'wb') as f:
                    shutil.copyfileobj(response.raw, f)
            print("FAISS index downloaded.")

            print(f"Downloading PKL file from {pkl_url} to {local_pkl_path}")
            with requests.get(pkl_url, stream=True, timeout=60) as response: # Added timeout
                response.raise_for_status()
                with open(local_pkl_path, 'wb') as f:
                    shutil.copyfileobj(response.raw, f)
            print("PKL file downloaded.")

            # Check if files exist after download
            if not os.path.exists(local_faiss_path) or not os.path.exists(local_pkl_path):
                 st.error("Failed to download required index files.")
                 return None

            print(f"Loading FAISS index from temporary directory: {tmpdir}")
            # Load FAISS with safe deserialization
            vector_store = FAISS.load_local(
                tmpdir,
                embeddings,
                index_name="index", # Must match the filenames
                allow_dangerous_deserialization=True # Required for loading pickle files
            )
            print("FAISS index loaded successfully from GitHub.")

    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading FAISS vector store components: {e}")
        print(f"Download error details: {e}")
        return None
    except pickle.UnpicklingError as e:
         st.error(f"Error deserializing the index PKL file. It might be corrupt or incompatible: {e}")
         print(f"Pickle error details: {e}")
         return None
    except Exception as e:
        st.error(f"An unexpected error occurred loading FAISS vector store: {e}")
        import traceback
        print(f"FAISS loading error traceback:\n{traceback.format_exc()}")
        return None

    return vector_store


# --- QA Chain Setup ---

# --- QA Chain Setup (Modified for Conversation) ---

# --- Add PromptTemplate import ---
from langchain.prompts import PromptTemplate

# --- Import default condense prompt for reference if needed ---
# from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT

# --- QA Chain Setup (Corrected & Tuned for Better Retrieval) ---

def get_conversational_compliance_chain(vector_store, memory):
    """Creates a ConversationalRetrievalChain tuned for better context handling."""
    global google_api_key
    if not google_api_key:
        st.error("Google API Key not configured. Cannot initialize QA model.")
        return None
    if vector_store is None:
         st.error("Vector store not available. Cannot create chain.")
         return None
    if memory is None:
         st.error("Memory object not available. Cannot create chain.")
         return None

    # --- Define the CORRECTED Prompt Template for Answer Generation ---
    # (This is the same as the previous fix)
    corrected_prompt_template = """
    You are an expert AI assistant specializing in LODR (Listing Obligations and disclosure requirments) regulations and procedures. Your role is to provide accurate guidance and interpretation of LODR rules and procedures based on the official LODR documentation provided in the context and the ongoing conversation history.

    When analyzing queries, please:
    1. Reference specific LODR articles and sections when applicable with page numbers and paragraphs found in the context documents.
    2. Explain procedures and requirements clearly based on the context and history.
    3. Highlight any relevant deadlines or time limits mentioned.
    4. Provide accurate interpretations of LODR rules and guidelines based *only* on the provided context and chat history.
    5. If information is not covered in the documents or conversation, explicitly state that.
    6. Respond concisely, then ask specific follow-up questions if required to clarify ambiguities realted to details from related sections or cross-references, emphasizing that more context yields a better answer.

    Chat History:
    {chat_history}

    Context (LODR Documentation):
    {context}

    User Question:
    {question}

    LODR Analysis and Response:
    """
    ANSWER_PROMPT = PromptTemplate(
        template=corrected_prompt_template, input_variables=["chat_history", "context", "question"]
    )
    # --- ---

    # --- Define a Custom Prompt for Question Condensing ---
    # This prompt tries to prioritize the follow-up question more explicitly
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
    If the follow up question is a new topic, unrelated to the chat history, the standalone question should be essentially the same as the follow up question.
    If the follow up question references the chat history, combine the relevant context from the history with the follow up question into a clear, self-contained question.

    Chat History:
    {chat_history}

    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
    # --- ---


    # Define the LLM (use the same LLM for both steps for consistency)
    try:
        llm = ChatGoogleGenerativeAI(
            google_api_key=google_api_key,
            model="gemini-2.0-flash-thinking-exp-01-21",
            temperature=0.1, # Keep temperature low for factual tasks
            top_p=0.9,
        )
    except Exception as e:
        st.error(f"Failed to initialize Google Chat Model: {e}")
        return None

    # Define the retriever (Consider tuning search_kwargs)
    retriever = vector_store.as_retriever(
        search_type="similarity", # Or try "mmr" (Max Marginal Relevance) for diversity
        search_kwargs={'k': 10} # Retrieve slightly fewer docs, potentially more focused
    )

    # Create the Conversational Retrieval Chain
    try:
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            condense_question_prompt=CONDENSE_QUESTION_PROMPT, # <-- Use the custom condense prompt
            combine_docs_chain_kwargs={"prompt": ANSWER_PROMPT}, # <-- Use the corrected answer prompt
            return_source_documents=True,
            output_key='answer',
            verbose=False # Set to True temporarily in your local environment to debug steps
        )
        print("ConversationalRetrievalChain created with custom condense_question_prompt.")
        return conversation_chain

    except Exception as e:
         st.error(f"Error creating ConversationalRetrievalChain: {e}")
         import traceback
         print(f"Chain creation error traceback:\n{traceback.format_exc()}") # Print full traceback
         return None


# --- Core Compliance Check ---

# --- Core Compliance Check (Modified for Conversation) ---

def check_compliance_with_history(user_submission):
    """Performs document retrieval and runs the conversational QA chain using history."""
    vector_store = None
    response = {"answer": "Could not process the request. Please check logs."} # Default error, note key is 'answer'
    source_documents = []

    # Retrieve memory from session state
    if 'memory' not in st.session_state:
         st.error("Chat memory not found in session state.")
         return {"answer": "Error: Chat history is missing."}, []
    memory = st.session_state.memory

    try:
        # Load vector store (consider caching)
        with st.spinner("Loading knowledge base..."):
             vector_store = load_vector_store_from_github()

        if vector_store is None:
            st.error("Failed to load the knowledge base (vector store). Cannot proceed.")
            return {"answer": "Error: Knowledge base could not be loaded."}, []

        # Get the Conversational QA chain (pass vector store and memory)
        chain = get_conversational_compliance_chain(vector_store, memory)
        if chain is None:
             st.error("Failed to create the QA processing chain.")
             return {"answer": "Error: Could not initialize the analysis process."}, []

        # Run the Conversational QA chain
        # The chain automatically uses the 'question' key and pulls from memory
        print(f"Running Conversational QA chain with question: '{user_submission}'")
        with st.spinner("Analyzing query with document and history..."):
            # The chain expects a dictionary, typically with 'question'
            response = chain({"question": user_submission})
            # The memory is automatically updated by the chain itself

        print("Conversational QA chain finished.")
        # Extract source documents if returned
        source_documents = response.get('source_documents', [])
        # Debug: Print source document metadata
        # if source_documents:
        #     print(f"Retrieved {len(source_documents)} source documents:")
        #     for i, doc in enumerate(source_documents):
        #          print(f"  Doc {i+1} Metadata: {doc.metadata}")


    except Exception as e:
        st.error(f"An error occurred during the conversational compliance check: {e}")
        import traceback
        print(f"Conversational check error traceback:\n{traceback.format_exc()}")
        response = {"answer": f"An unexpected error occurred. Please try again."} # Ensure key is 'answer'

    # Return the response dictionary and source documents
    # The response dictionary now contains 'chat_history', 'question', 'answer', 'source_documents'
    return response, source_documents # Return full response dict and docs

# --- Main Chatbot Interface ---
# --- Main Chatbot Interface ---

def run_chatbot():
    """Runs the Streamlit chatbot interface with conversation history."""
    display_logo()

    # Display chat history (remains the same)
    if st.session_state.messages:
        for message in st.session_state.messages:
            role_class = "user" if message['role'] == 'user' else 'assistant'
            with st.container():
                st.markdown(f"""
                    <div class="chat-message {role_class}">
                        <b>{message['role'].title()}:</b><br>{message['content']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # User input area
    user_query = st.chat_input("Ask about LODR regulations:")

    if user_query:
        # Add user message to session state chat history (for display)
        st.session_state.messages.append({"role": "user", "content": user_query})
        # Display user message
        with st.chat_message("user"):
             st.markdown(user_query)

        # Get AI response using the conversational chain
        # This function now also updates st.session_state.memory internally via the chain
        ai_response_dict, _ = check_compliance_with_history(user_query)

        # Extract the actual answer text (key is 'answer' now)
        ai_response_text = ai_response_dict.get('answer', "Sorry, I encountered an issue processing your request.")

        # Add assistant response to session state chat history (for display)
        st.session_state.messages.append({"role": "assistant", "content": ai_response_text})
        # Display assistant response
        with st.chat_message("assistant"):
             st.markdown(ai_response_text)

        # No st.rerun() needed with st.chat_input

    # Add a clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = [] # Clear display history
        # Clear the Langchain memory
        if 'memory' in st.session_state:
            st.session_state.memory.clear()
        st.rerun() # Rerun to clear the display and memory state


# --- Application Entry Point ---

if __name__ == "__main__":
    # --- Dependency Checks ---
    try:
        import faiss
        print("FAISS library found.")
    except ImportError:
        st.error("Fatal Error: FAISS library is not installed. Please ensure 'faiss-cpu' or 'faiss-gpu' is in your requirements.")
        st.stop() # Stop execution if FAISS is missing

    try:
        import fitz # PyMuPDF check
        print("PyMuPDF (fitz) library found.")
    except ImportError:
        # This might not be fatal for *running* the chatbot if index exists,
        # but it's needed for the *new* chunking functions.
        st.warning("Warning: PyMuPDF (fitz) library not found. Metadata chunking features will not work if index needs regeneration.")
        # Decide if you want to st.stop() here or allow running without it.
        # For now, let's allow it to run, assuming a pre-built index.

    # --- API Key Configuration ---
    try:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
        if not google_api_key:
             st.error("Fatal Error: GOOGLE_API_KEY not found in Streamlit secrets or is empty.")
             st.stop()
        # Optionally configure the GenAI library globally if needed elsewhere
        # genai.configure(api_key=google_api_key)
        print("Google API Key loaded from secrets.")
    except KeyError:
        st.error("Fatal Error: 'GOOGLE_API_KEY' not found in Streamlit secrets.")
        st.info("Please add your Google API Key to your Streamlit Cloud secrets.")
        st.stop()
    except Exception as e:
         st.error(f"Fatal Error: An unexpected issue occurred while accessing secrets: {e}")
         st.stop()

    # --- Run the App ---
    print("Starting Streamlit chatbot application...")
    run_chatbot()

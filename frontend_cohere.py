import streamlit as st
import cohere
import hnswlib
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from tqdm import tqdm
from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API Key (consider using Streamlit secrets in production)
COHERE_API_KEY = "6Z82p3qm3xg21rREfIjjJIsgBGv2q7nLoyvKaZOp"
co = cohere.Client(COHERE_API_KEY)

# Configuration for prompt template
PROMPT_TEMPLATE = """
You are an expert AI assistant helping users navigate and understand documents. 
Your goal is to provide clear, concise, and helpful responses based on the available context.

Context Guidelines:
- Use only the provided document context to answer questions
- If the exact answer is not in the documents, clearly state that
- Provide precise and informative answers
- Break down complex information into easily understandable points
- If multiple documents are relevant, synthesize information from them

Specific Instructions:
1. Always start by briefly acknowledging the source of your information
2. Present information in a structured, easy-to-read format
3. Use bullet points or numbered lists when appropriate
4. If the query requires interpretation beyond the literal text, explain your reasoning

User Query: {query}

Available Context:
{context}

Response:
"""

class Ingestion:
    def __init__(self, pdf_dir):
        self.docs = []
        self.docs_embs = []
        self.retrieve_top_k = 10
        self.rerank_top_k = 3
        self.pdf_dir = pdf_dir
        self.load_and_chunk()
        self.embed()
        self.index()

    def load_and_chunk(self) -> None:
        """
        Loads the text from the sources and chunks the PDF documents.
        """
        try:
            pdf_dir = Path(self.pdf_dir)
            logger.info(f"Processing documents from {pdf_dir}")
            
            if not pdf_dir.exists():
                raise FileNotFoundError(f"Directory not found: {pdf_dir}")
            
            # Initialize text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2500,
                chunk_overlap=500,
                length_function=len,
                is_separator_regex=False
            )
            
            pdf_files = list(pdf_dir.glob("*.pdf"))

            for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
                try:
                    loader = UnstructuredPDFLoader(str(pdf_path), mode="elements")
                    document = loader.load()
                except Exception as e:
                    logger.warning(f"Falling back to PyPDFLoader for {pdf_path}")
                    loader = PyPDFLoader(str(pdf_path))
                    document = loader.load()
                
                document_title = pdf_path.stem
                chunks = text_splitter.split_documents(document)
                
                for chunk in chunks:
                    self.docs.append({
                        "title": document_title, 
                        "text": chunk.page_content, 
                        "url": str(pdf_path)  
                    })
                
                logger.debug(f"Processed {pdf_path.name}: {len(chunks)} chunks")
            
            logger.info(f"Total chunks created: {len(self.docs)}")
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise

    def embed(self) -> None:
        """
        Embeds the document chunks using the Cohere API.
        """
        batch_size = 50
        self.docs_len = len(self.docs)
        for i in range(0, self.docs_len, batch_size):
            batch = self.docs[i : min(i + batch_size, self.docs_len)]
            texts = [item["text"] for item in batch]
            docs_embs_batch = co.embed(
                texts=texts, model="embed-english-v3.0", input_type="search_document"
            ).embeddings
            self.docs_embs.extend(docs_embs_batch)
            time.sleep(1)

    def index(self) -> None:
        """
        Indexes the documents for efficient retrieval.
        """
        self.idx = hnswlib.Index(space="ip", dim=1024)
        self.idx.init_index(max_elements=self.docs_len, ef_construction=512, M=64)
        self.idx.add_items(self.docs_embs, list(range(len(self.docs_embs))))

    def retrieve(self, query: str):
        """
        Retrieves document chunks based on the given query.
        """
        query_emb = co.embed(
            texts=[query], model="embed-english-v3.0", input_type="search_query"
        ).embeddings

        doc_ids = self.idx.knn_query(query_emb, k=self.retrieve_top_k)[0][0]

        rank_fields = ["title", "text"]
        docs_to_rerank = [self.docs[doc_id] for doc_id in doc_ids]

        rerank_results = co.rerank(
            query=query,
            documents=docs_to_rerank,
            top_n=self.rerank_top_k,
            model="rerank-english-v3.0",
            rank_fields=rank_fields
        )
        
        doc_ids_reranked = [doc_ids[result.index] for result in rerank_results.results]

        docs_retrieved = []
        for doc_id in doc_ids_reranked:
            docs_retrieved.append(
                {
                    "title": self.docs[doc_id]["title"],
                    "text": self.docs[doc_id]["text"],
                    "url": self.docs[doc_id]["url"],
                }
            )
        
        return docs_retrieved

def run_chatbot(ingestion, message, chat_history=[]):
    """
    Run the chatbot with retrieval-augmented generation and enhanced prompting.
    """
    # Generate search queries
    response = co.chat(
        message=message,
        model="command-r-plus",
        search_queries_only=True,
        chat_history=chat_history
    )
    
    search_queries = [query.text for query in response.search_queries]

    # If there are search queries, retrieve the documents
    documents = []
    if search_queries:
        for query in search_queries:
            documents.extend(ingestion.retrieve(query))

        # Prepare context from retrieved documents
        context = "\n\n".join([
            f"Document Title: {doc['title']}\nContent: {doc['text']}\nSource: {doc['url']}" 
            for doc in documents
        ])

        # Construct the enhanced prompt
        enhanced_prompt = PROMPT_TEMPLATE.format(
            query=message,
            context=context
        )

        # Use document chunks to respond
        response = co.chat_stream(
            message=enhanced_prompt,
            model="command-r-plus",
            documents=documents,
            chat_history=chat_history,
        )
    else:
        # Fallback to standard chat if no documents retrieved
        response = co.chat_stream(
            message=message,
            model="command-r-plus",
            chat_history=chat_history,
        )
    
    # Collect the full response
    full_response = ""
    citations = []
    cited_documents = []

    for event in response:
        if event.event_type == "text-generation":
            full_response += event.text
        if event.event_type == "stream-end":
            citations = event.response.citations
            cited_documents = event.response.documents
            chat_history = event.response.chat_history

    return full_response, citations, cited_documents, chat_history

def main():
    # Set page configuration
    st.set_page_config(
    page_title="OIC Documentation Assistant using Cohere",
    page_icon="🤖",
    layout="wide"
   )

    # Title
    st.title("🤖 OIC Documentation Assistant using Cohere")
    st.write("Ask me anything regarding OIC2.")

    # Sidebar for PDF upload and configuration

    with st.sidebar:
        st.header("💡 Assistant Features")
        st.markdown("""
        This assistant helps you find information about Oracle Integration Cloud 2 (OIC2).
        It uses:
        - Llama3.1 70B model
        - FAISS vector store
        - BGE embeddings
        """)
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    pdf_directory = "C:/Users/snehsrin/Desktop/All/Self/OIC/oic/oic/spiders/test/"

    # Initialize ingestion when directory is specified
 
    with st.spinner("Initializing document index..."):
        try:
            ingestion = Ingestion(pdf_directory)
            st.sidebar.success("Documents indexed successfully!")
        except Exception as e:
            st.sidebar.error(f"Error indexing documents: {e}")
            return

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Chat input
    user_input = st.chat_input("Ask a question about your documents")

    # Chat interface
    if user_input:
        # Display user message
        st.chat_message("user").write(user_input)

        # Process the user's message
        with st.spinner("Generating response..."):
            try:
                response, citations, cited_documents, updated_chat_history = run_chatbot(
                    ingestion, 
                    user_input, 
                    st.session_state.chat_history
                )
                
                # Display AI response
                st.chat_message("assistant").write(response)

                # Update chat history
                st.session_state.chat_history = updated_chat_history

                # Optionally display citations and documents
                with st.expander("Citations and Sources"):
                    if citations:
                        st.subheader("Citations")
                        for citation in citations:
                            st.write(citation)
                    
                    if cited_documents:
                        st.subheader("Cited Documents")
                        for doc in cited_documents:
                            st.write(doc)

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
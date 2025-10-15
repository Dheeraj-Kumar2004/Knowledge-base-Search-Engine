# my_rag.py
import os
import logging
from typing_extensions import TypedDict
from typing import List
from dotenv import load_dotenv

# LangChain / document handling
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

# LangGraph
from langgraph.graph import START, StateGraph

# LangChain message types
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import MessagesState
from langgraph.checkpoint.memory import MemorySaver

# Google GenAI SDK
import google.generativeai as genai
# LangChain Google GenAI embeddings integration
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

logger = logging.getLogger(_name_)
load_dotenv()

class MyRAGAgent:
    """
    A RAG (Retrieval-Augmented Generation) agent that processes PDF documents
    and provides intelligent chat responses using Google Gemini (GenAI) for
    generation and embeddings.
    """
    
    def _init_(self, gemini_api_key: str):
        """
        Initialize the RAG agent with Gemini API key and set up components.
        """
        self.gemini_api_key = gemini_api_key
        if not self.gemini_api_key:
            raise EnvironmentError("GEMINI_API_KEY is required to initialize MyRAGAgent")
        
        try:
            # Configure genai SDK
            genai.configure(api_key=self.gemini_api_key)
            logger.info("GenAI SDK configured")
            
            # Create a generative model instance (you can change model_name if needed)
            self.gen_model = genai.GenerativeModel(
                model_name="gemini-2.5-flash",
                # you can pass a default generation_config here or per-call
            )
            logger.info("Generative model instance created")
            
            # Initialize embeddings using LangChain Google GenAI wrapper
            # The constructor accepts google_api_key kwarg
            self.embeddings = GoogleGenerativeAIEmbeddings(google_api_key=self.gemini_api_key)
            logger.info("Embeddings model initialized (Google Generative AI embeddings)")
            
            # Initialize in-memory vector store
            self.vector_store = InMemoryVectorStore(self.embeddings)
            logger.info("Vector store initialized successfully")
            
            # Initialize text splitter for document chunking
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            logger.info("Text splitter initialized successfully")
            
            # Initialize chat history
            self.chat_history = []
            
            # Setup the conversation graph
            self._setup_graph()
            logger.info("RAG Agent initialized successfully with Gemini")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG Agent: {e}")
            raise Exception(f"RAG Agent initialization failed: {e}")

    def load_documents(self, pdf_paths: List[str]) -> None:
        """
        Load and process PDF documents into the vector store.
        """
        if not pdf_paths:
            logger.warning("No PDF paths provided for loading")
            return
            
        docs = []
        
        for pdf_path in pdf_paths:
            try:
                if not os.path.exists(pdf_path):
                    logger.error(f"PDF file not found: {pdf_path}")
                    continue
                    
                if not pdf_path.lower().endswith('.pdf'):
                    logger.error(f"File is not a PDF: {pdf_path}")
                    continue
                
                logger.info(f"Loading PDF from {pdf_path}")
                loader = PyPDFLoader(pdf_path)
                loaded_docs = loader.load()
                
                if not loaded_docs:
                    logger.warning(f"No content loaded from {pdf_path}")
                    continue
                
                docs.extend(loaded_docs)
                logger.info(f"Successfully loaded {len(loaded_docs)} pages from {pdf_path}")
                
            except Exception as e:
                logger.error(f"Error loading PDF {pdf_path}: {e}")
                continue
        
        if not docs:
            logger.error("No documents were successfully loaded")
            raise Exception("Failed to load any documents")
        
        try:
            # Split documents into chunks
            all_splits = self.text_splitter.split_documents(docs)
            logger.info(f"Split documents into {len(all_splits)} chunks")
            
            if not all_splits:
                raise Exception("Document splitting resulted in no chunks")
            
            # Add documents to vector store
            self.vector_store.add_documents(all_splits)
            logger.info(f"Added {len(all_splits)} document chunks to vector store")
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            raise Exception(f"Document processing failed: {e}")

    def ask(self, question: str) -> str:
        """
        Ask a question to the RAG system and get a response.
        """
        if not question.strip():
            raise ValueError("Question cannot be empty")
        
        try:
            state = {
                "messages": self.chat_history.copy(),
                "question": question.strip(),
                "context": [],
                "answer": ""
            }
            
            config = {"configurable": {"thread_id": "session-1"}}
            response = self.graph.invoke(state, config)
            
            # Update chat history with new messages
            if "messages" in response and response["messages"]:
                self.chat_history.extend(response["messages"])
                # Keep chat history manageable (last 20 messages)
                if len(self.chat_history) > 20:
                    self.chat_history = self.chat_history[-20:]
            
            answer = response.get("answer", "I apologize, but I couldn't generate a response.")
            logger.info(f"Generated response for question: {question[:50]}...")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise Exception(f"Failed to generate response: {e}")

    def _setup_graph(self):
        """Setup the conversation graph for RAG processing."""
        try:
            class State(TypedDict):
                messages: MessagesState
                question: str
                context: List[Document]
                answer: str

            def retrieve(state: State):
                """Retrieve relevant documents based on the question."""
                try:
                    question = state.get("question", "")
                    if not question:
                        return {"context": []}
                    
                    # Perform similarity search (k=4)
                    retrieved_docs = self.vector_store.similarity_search(
                        question, 
                        k=4
                    )
                    
                    logger.info(f"Retrieved {len(retrieved_docs)} documents for question")
                    return {"context": retrieved_docs}
                    
                except Exception as e:
                    logger.error(f"Error in retrieve step: {e}")
                    return {"context": []}

            def generate(state: State):
                """Generate response based on retrieved context and chat history."""
                try:
                    question = state.get("question", "")
                    context_docs = state.get("context", [])
                    messages = state.get("messages", [])
                    
                    # Prepare context from retrieved documents
                    docs_content = ""
                    if context_docs:
                        docs_content = "\n\n".join(
                            f"Document {i+1}:\n{doc.page_content[:800]}..." 
                            if len(doc.page_content) > 800 
                            else f"Document {i+1}:\n{doc.page_content}"
                            for i, doc in enumerate(context_docs)
                        )
                    
                    # Prepare conversation history (text)
                    history_text = ""
                    if messages:
                        # messages are expected to be objects with .type and .content in your graph usage
                        history_text = "\n".join(
                            f"{getattr(msg, 'type', 'Human')}: {getattr(msg, 'content', str(msg))}"
                            for msg in messages[-10:]
                        )
                    
                    # Create comprehensive prompt
                    context_section = f"Relevant Information:\n{docs_content}\n\n" if docs_content else ""
                    history_section = f"Recent Conversation:\n{history_text}\n\n" if history_text else ""
                    
                    prompt_text = f"""You are a helpful AI assistant that answers questions based on provided documents and conversation context.

{context_section}{history_section}Current Question: {question}

Instructions:
- Answer the question clearly and helpfully
- Use information from the provided documents when relevant
- If the documents don't contain relevant information, say so clearly
- Be concise but comprehensive
- Maintain conversation context when appropriate

Answer:"""

                    # Call Gemini (GenAI) model to generate content
                    # We call generate_content with a single text prompt; adjust generation_config as needed
                    response = self.gen_model.generate_content(prompt_text)
                    # response.text contains generated text per SDK
                    gen_text = ""
                    # different SDK versions may expose text or .text or other attribute
                    if hasattr(response, "text") and response.text:
                        gen_text = response.text
                    elif hasattr(response, "content") and response.content:
                        gen_text = response.content
                    else:
                        # fallback to string representation
                        gen_text = str(response)

                    # Build messages for chat history
                    human_msg = HumanMessage(content=question)
                    ai_msg = AIMessage(content=gen_text)

                    return {
                        "answer": gen_text,
                        "messages": [human_msg, ai_msg]
                    }
                    
                except Exception as e:
                    logger.error(f"Error in generate step: {e}")
                    error_message = "I apologize, but I encountered an error while generating a response. Please try again."
                    return {
                        "answer": error_message,
                        "messages": [HumanMessage(content=question), AIMessage(content=error_message)]
                    }

            # Build the conversation graph
            graph_builder = StateGraph(State).add_sequence([retrieve, generate])
            graph_builder.add_edge(START, "retrieve")
            
            # Add memory checkpointer for conversation persistence
            checkpointer = MemorySaver()
            self.graph = graph_builder.compile(checkpointer=checkpointer)
            
            logger.info("Conversation graph setup completed successfully")
            
        except Exception as e:
            logger.error(f"Error setting up conversation graph: {e}")
            raise Exception(f"Graph setup failed: {e}")

    def reset_chat_history(self):
        """Reset the chat history."""
        self.chat_history = []
        logger.info("Chat history reset")

    def get_document_count(self) -> int:
        """Get the number of documents in the vector store."""
        try:
            return len(self.vector_store.get_documents())
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0

    def reset(self):
        """Reset the RAG agent state."""
        try:
            # Clear the vector store
            self.vector_store = InMemoryVectorStore(self.embeddings)
            self.chat_history = []
            logger.info("RAG agent state has been reset")
        except Exception as e:
            logger.error(f"Error resetting RAG agent: {e}")
            raise Exception(f"Failed to reset RAG agent: {e}")
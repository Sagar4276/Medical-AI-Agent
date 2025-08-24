from typing import List, Dict, Any, Optional, Union, Callable
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain, LLMChain

from utils.logger import get_logger
from rag.retriever import EnhancedMedicalRetriever

logger = get_logger(__name__)

# Medical-specific system prompt for better responses
MEDICAL_SYSTEM_PROMPT = """You are an advanced medical AI assistant trained to provide accurate, helpful information on medical topics. 
Your purpose is to assist with medical information and education only.

When responding to queries:
- Provide accurate medical information based on current medical knowledge
- Cite your sources whenever possible
- Be clear about limitations and uncertainties in medical knowledge
- Avoid making definitive diagnoses
- Explain complex medical concepts in clear, understandable language
- Always recommend consulting healthcare professionals for personal medical advice
- Do not prescribe medications or recommend specific treatments
- Present balanced information on controversial topics
- Always mention if a topic is outside of your expertise
- Use the retrieved context to inform your responses, but don't reference the context directly

**FORMATTING REQUIREMENTS:**
- Format your response using proper markdown syntax
- Use **bold text** for important terms, conditions, and key concepts
- Use bullet points (-) for lists and enumerations
- Use headers (## or ###) for different sections when appropriate
- Use *italics* for emphasis on less critical information
- Make your response well-structured and visually appealing

RETRIEVED CONTEXT:
{context}

Remember that your responses may impact health decisions, so accuracy and clarity are essential.
"""

class MedicalResponseGenerator:
    """
    Generates medical responses using a retrieval-augmented generation approach
    """
    
    def __init__(
        self,
        retriever: EnhancedMedicalRetriever,
        model_name: str = "llama-3.1-8b-instant",
        provider: str = "groq",
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        system_prompt: str = MEDICAL_SYSTEM_PROMPT,
        max_tokens: int = 1024,
        top_p: float = 0.95
    ):
        """
        Initialize the medical response generator
        
        Args:
            retriever: Document retriever
            model_name: Name of the model to use
            provider: Model provider (groq, ollama, etc.)
            temperature: Temperature for generation
            api_key: API key for the model provider
            system_prompt: System prompt for the model
            max_tokens: Maximum tokens to generate
            top_p: Top-p for generation
        """
        self.retriever = retriever
        self.system_prompt = system_prompt
        self.model_name = model_name
        
        # Get API key from environment if not provided
        if api_key is None and provider == "groq":
            import os
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables")
        
        # Initialize LLM based on provider
        if provider == "groq":
            self.llm = ChatGroq(
                model_name=model_name,
                temperature=temperature,
                groq_api_key=api_key,
                max_tokens=max_tokens,
                top_p=top_p
            )
        elif provider == "ollama":
            from langchain_community.llms import Ollama
            self.llm = Ollama(
                model=model_name,
                temperature=temperature,
                num_ctx=4096
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Initialize the RAG pipeline
        self._init_rag_pipeline()
    
    def _format_docs(self, docs: List[Document]) -> str:
        """Format retrieved documents into a string for context"""
        return "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
    
    def _init_rag_pipeline(self):
        """Initialize the RAG pipeline"""
        # Create the prompt template
        prompt = ChatPromptTemplate.from_template(
            template="""
            Answer the following medical question based on the provided context.
            If the context doesn't contain relevant information, acknowledge that and provide general medical information.
            
            IMPORTANT: Format your response using proper markdown syntax:
            - Use **bold text** for medical terms, conditions, and important concepts
            - Use bullet points (-) for lists
            - Use headers (## or ###) for sections
            - Use *italics* for emphasis
            - Make the response well-structured and visually appealing
            
            Question: {question}
            """,
            system_message=self.system_prompt
        )
        
        # Create the chain
        self.chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Create conversational chain for multi-turn conversations
        self.conversational_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="stuff",
            verbose=True
        )
    
    def generate_response(self, query: str) -> str:
        """
        Generate a response to a medical query
        
        Args:
            query: Medical question
            
        Returns:
            Generated response
        """
        logger.info(f"Generating response for query: {query[:30]}...")
        
        try:
            response = self.chain.invoke(query)
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while processing your query. Please try again with a different question."
    
    def chat(self, query: str, chat_history: List = None) -> str:
        """
        Generate a response in a conversational context
        
        Args:
            query: Current query
            chat_history: Chat history as list of (query, response) tuples
            
        Returns:
            Generated response
        """
        chat_history = chat_history or []
        logger.info(f"Generating conversational response for query: {query[:30]}...")
        
        try:
            result = self.conversational_chain({"question": query, "chat_history": chat_history})
            return result["answer"]
        except Exception as e:
            logger.error(f"Error generating conversational response: {e}")
            return f"I apologize, but I encountered an error while processing your query in conversation. Please try again."
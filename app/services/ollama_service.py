from typing import Dict, List
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

class OllamaService:
    def __init__(self, model_name: str = "mxbai-embed-large"):
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
        )

    async def get_embedding(self, text: str) -> List[float]:
        """Get embeddings for a single text using Ollama."""
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            raise Exception(f"Error getting embeddings: {str(e)}")

    async def get_event_embedding(self, event: Dict) -> List[float]:
        """Get embeddings for an event by converting it to a searchable string format."""
        # Convert event to a searchable string format
        event_text = f"Event: {event.get('title', '')} "
        event_text += f"Date: {event.get('date', '')} "
        if event.get('description'):
            event_text += f"Description: {event.get('description')} "
        if event.get('location'):
            event_text += f"Location: {event.get('location')} "
        if event.get('participants'):
            event_text += f"Participants: {', '.join(event.get('participants'))} "
        
        # Split text into chunks if needed
        chunks = self.text_splitter.split_text(event_text)
        # For events, we'll just use the first chunk since they're typically small
        return await self.get_embedding(chunks[0]) 
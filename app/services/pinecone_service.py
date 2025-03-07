import uuid
from pinecone import Pinecone, ServerlessSpec
from typing import Dict, List, Optional
from app.core.config import get_settings

settings = get_settings()

class PineconeService:
    def __init__(self):
        # Initialize Pinecone
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        
        # Get or create index
        self.index_name = "events"
        if not self.pc.has_index(self.index_name):
            self.pc.create_index(
                name=self.index_name,
                dimension=768, 
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        self.index = self.pc.Index(self.index_name)

    async def store_event(self, user_id: str, event: Dict, embedding: List[float]):
        """Store an event with its embedding in Pinecone."""
        # Create a unique ID for the event using the user_id and a generated UUID
        event_id = f"{user_id}_{uuid.uuid4()}"
        
        # Create an event summary from the entire event dict
        event_summary = f"Event details: {event}"
        
        # Store the event with its embedding and full metadata
        self.index.upsert(
            vectors=[(
                event_id,
                embedding,
                {
                    'user_id': user_id,
                    'content': event_summary,
                    **event
                }
            )]
        )

    async def search_events(self, user_id: str, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Search for events using the query embedding."""
        results = self.index.query(
            vector=query_embedding,
            filter={'user_id': user_id},
            top_k=top_k,
            include_metadata=True
        )
        
        # Extract and return the events from the results
        events = []
        for match in results.get('matches', []):
            if match.get('metadata'):
                events.append(match['metadata'])
        
        return events

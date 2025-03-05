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
                dimension=1024,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        self.index = self.pc.Index(self.index_name)

    async def store_event(self, user_id: str, event: Dict, embedding: List[float]):
        """Store an event with its embedding in Pinecone."""
        # Create a unique ID for the event
        event_id = f"{user_id}_{event.get('title')}_{event.get('date')}"
        
        # Store the event with its embedding
        self.index.upsert(
            vectors=[(
                event_id,
                embedding,
                {
                    'user_id': user_id,
                    'content': f"Event: {event.get('title')} on {event.get('date')}",
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
        for match in results['matches']:
            if match['metadata']:
                events.append(match['metadata'])
        
        return events 
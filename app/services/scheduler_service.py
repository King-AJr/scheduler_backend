from typing import Dict, List, Optional
from datetime import datetime
import json
from app.services.groq_service import GroqService
from app.services.ollama_service import OllamaService
from app.services.pinecone_service import PineconeService

class SchedulerService:
    def __init__(self):
        self.groq = GroqService()
        self.ollama = OllamaService()
        self.pinecone = PineconeService()
        self.conversation_history = []

    async def process_message(self, user_id: str, message: str) -> str:
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": message})
        
        # First determine if this is a query or an event addition
        is_query = await self._is_schedule_query(message)
        
        # Get today's date for context
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Combine message with today's date
        message_with_date = f"{message} (context: today is {today})"
        
        if is_query:
            # Handle event query
            query_embedding = await self.ollama.get_embedding(message_with_date)
            events = await self.pinecone.search_events(user_id, query_embedding, top_k=20)
            
            # Generate response with event context
            system_prompt = """You are a helpful scheduling assistant. Based on the user's query and the events provided, 
            give a natural and informative response. Include relevant event details in a conversational way.
            If no events are found, let the user know in a friendly manner."""
            
            prompt = [
                {"role": "system", "content": system_prompt}
            ]
            
            if events:
                prompt.append({
                    "role": "system",
                    "content": f"Today's date: {today}\nRelevant events: {json.dumps(events, default=str)}"
                })
            else:
                prompt.append({
                    "role": "system",
                    "content": f"Today's date: {today}\nNo relevant events found in your schedule."
                })
            
            prompt.append({"role": "user", "content": message})
            
        else:
            embedding = await self.ollama.get_embedding(message_with_date)
            await self.pinecone.store_event(user_id, {"content": message}, embedding)
            
            # Generate confirmation response
            system_prompt = """You are a helpful scheduling assistant. The user's message has been saved.
            Confirm this naturally and acknowledge what they said."""
            
            prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]
        
        # Generate and store response
        response = await self.groq.generate(prompt)
        self.conversation_history.append({"role": "assistant", "content": response})
        return response


    async def _is_schedule_query(self, message: str) -> bool:
        prompt = [
            {"role": "system", "content": """Determine if the user is asking about schedules, events, or calendar information. 
            Return only 'true' or 'false'."""},
            {"role": "user", "content": message}
        ]
        
        response = await self.groq.generate(prompt)
        return 'true' in response.lower() 
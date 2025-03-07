from typing import List
import json
from datetime import datetime
from app.models.schedule import Event, ScheduleResponse
from app.services.groq_service import GroqService
from app.services.ollama_service import OllamaService
from app.services.pinecone_service import PineconeService
from app.services.firestore_service import FirestoreService

class ScheduleService:
    def __init__(self):
        self.groq = GroqService()
        self.ollama = OllamaService()
        self.pinecone = PineconeService()
        self.firestore = FirestoreService()

    async def get_schedule_from_query(self, user_id: str, query: str) -> ScheduleResponse:
        """
        Process the schedule query and return formatted events
        """
        print(f"Query: {query}")
        # Get today's date for context
        today = datetime.now().strftime("%Y-%m-%d")
        message_with_date = f"{query} (context: today is {today})"
        
        # Get embedding for the query
        query_embedding = await self.ollama.get_embedding(message_with_date)
        
        # Search for relevant events
        events_data = await self.pinecone.search_events(user_id, query_embedding, top_k=20)
        
        # Prepare prompt for LLM
        prompt = [
            {"role": "system", "content": """
            You are a scheduling assistant. Based on the events provided and the user's query,
            return only events that are within the date range specified by the user
            format the events into a clear response. Return ONLY a raw JSON object with 
            a list of 'events' with fields: title, time, date, priority, tag, venue. Nothing else just the list of events.
            
            If there's no events within the date range specified by the user, return an empty JSON object.
            Format dates consistently and ensure all required fields are present.
            Do not include markdown code block markers in your response.
            """},
            {"role": "user", "content": f"Query: {query}\nEvents found: {json.dumps(events_data, default=str)}"}
        ]

        # Generate response using Groq
        llm_response = await self.groq.generate(prompt)
        
        # Parse and validate the response
        response_data = self._parse_llm_response(llm_response)
        
        
        return response_data

    def _parse_llm_response(self, llm_response: str) -> ScheduleResponse:
        """
        Parse the LLM response and convert it to ScheduleResponse
        """
        try:
            start = llm_response.find('{')
            end = llm_response.rfind('}') + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON object found in response")
            
            json_str = llm_response[start:end]
            response_data = json.loads(json_str)
            events = [Event(**event) for event in response_data.get('events', [])]
            return ScheduleResponse(events=events)
        except Exception as e:
            raise ValueError(f"Failed to parse events: {str(e)}") 
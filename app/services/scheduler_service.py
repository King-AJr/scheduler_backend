from typing import Dict, List
from datetime import datetime
import json
from app.services.groq_service import GroqService
from app.services.ollama_service import OllamaService
from app.services.pinecone_service import PineconeService
from app.services.firestore_service import FirestoreService

class SchedulerService:
    def __init__(self):
        self.groq = GroqService()
        self.ollama = OllamaService()
        self.pinecone = PineconeService()
        self.firestore = FirestoreService()

    async def process_message(self, user_id: str, message: str) -> str:
        today = datetime.now().strftime("%Y-%m-%d")
        message_with_date = f"{message} (context: today is {today})"

        # Fetch recent conversation history
        recent_conversations = await self.firestore.get_user_conversations(user_id, limit=10)
        conversation_context = []
        for conv in recent_conversations:
            conversation_context.extend([
                {"role": "user", "content": conv['message']},
                {"role": "assistant", "content": conv['response']}
            ])

        is_query = await self._is_schedule_query(message)

        if is_query:
            print('Processing schedule query...')
            query_embedding = await self.ollama.get_embedding(message_with_date)
            events = await self.pinecone.search_events(user_id, query_embedding, top_k=10)

            system_prompt = (
                "Hi there! I'm your personal scheduling assistant. "
                "It seems you might be asking about your schedule or interacting with me. "
                "I can help review your upcoming events, provide details, or even set reminders based on your needs."
            )

            prompt = [{"role": "system", "content": system_prompt}]
            # Add conversation history
            prompt.extend(conversation_context)
            
            if events:
                prompt.append({
                    "role": "system", "content": (
                        f"Today is {today}. I found the following events on your calendar: {json.dumps(events, default=str)}. "
                        "You can ask for more details, set reminders, or make changes to any event if needed. But if the user is not asking about an event respond appropriately"
                    )
                })
            else:
                prompt.append({
                    "role": "system", "content": (
                        f"Today is {today}. I couldn't locate any events in your calendar. "
                        "Would you like to add a new event, or do you have another query I can assist with?"
                    )
                })
            prompt.append({"role": "user", "content": message})
        
        else:
            print('Processing event addition...')
            system_prompt = (
                "Hi there! I'm here to help you add a new event to your calendar. "
                "Please ensure you include all necessary details such as title, date, time, and location."
            )
            embedding = await self.ollama.get_embedding(message_with_date)
            event_data = {"content": message_with_date, "status": "complete", "timestamp": today}
            await self.pinecone.store_event(user_id, event_data, embedding)
            event_completion_prompt = [
                {"role": "system", "content": system_prompt},
                # Add conversation history
                *conversation_context,
                {"role": "user", "content": message}
            ]

            event_response = await self.groq.generate(event_completion_prompt)
            event_details = await self._extract_event_details(event_response)
            
            if event_details:
                embedding = await self.ollama.get_embedding(message_with_date)
                await self.pinecone.store_event(user_id, event_details, embedding)
                response_text = (
                    f"Your event has been successfully saved with the following details: {event_details}. "
                    "If any information is missing or you need changes, please let me know!"
                )
            else:
                response_text = (
                    "I need a bit more information to add your event. "
                    "Could you please provide the title, date, time, and location?"
                )
            
            prompt = [
                {"role": "system", "content": response_text},
                {"role": "user", "content": message}
            ]
        
        prompt.extend(conversation_context)
        response = await self.groq.generate(prompt)
        await self.firestore.store_conversation(user_id, message, response)
        
        return response

    async def _is_schedule_query(self, message: str) -> bool:
        prompt = [
            {"role": "system", "content": (
                "You are a classifier that determines whether a message is asking about reviewing schedules or general calendar queries, "
                "versus storing/adding a new event. If the message is about storing or adding an event, return 'false'. "
                "If it is a query about the schedule or a general interaction, return 'true'. "
                "Return ONLY 'true' or 'false'."
            )},
            {"role": "user", "content": message}
        ]
        response = await self.groq.generate(prompt)
        print(f"Response: {response}")
        return 'true' in response.lower()

    async def _extract_event_details(self, response: str) -> Dict:
        prompt = [
            {"role": "system", "content": (
                "Extract event details from the following response. "
                "Ensure that the details include 'title', 'date', 'time', and 'location'. "
                "If any detail is missing, return an empty dictionary. "
                "Please provide the output as structured JSON."
            )},
            {"role": "user", "content": response}
        ]
        extracted_details = await self.groq.generate(prompt)
        try:
            return json.loads(extracted_details)
        except json.JSONDecodeError:
            return {}

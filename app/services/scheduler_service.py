from typing import Dict, List, Optional
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
        # Get today's date for context
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Combine message with today's date for better embedding context
        message_with_date = f"{message} (context: today is {today})"
        
        # First determine if this is a schedule query or an event addition
        is_query = await self._is_schedule_query(message)
        
        if is_query:
            # --- Handling Schedule Queries ---
            print('Processing schedule query...')
            query_embedding = await self.ollama.get_embedding(message_with_date)
            events = await self.pinecone.search_events(user_id, query_embedding, top_k=20)
            
            # Construct a conversational system prompt
            system_prompt = (
                "Hi there! I'm your scheduling assistant. Let me check your calendar for you. "
                "I'll provide you with the details of your upcoming events and even offer some suggestions if needed."
            )
            
            # Prepare the prompt with context
            prompt = [{"role": "system", "content": system_prompt}]
            if events:
                # Include the event details in a friendly, conversational style
                prompt.append({
                    "role": "system",
                    "content": (
                        f"Today is {today}. Here are some events I found: {json.dumps(events, default=str)}. "
                        "I noticed a few events that might need a reminder or additional details. "
                        "Would you like me to help set a reminder or get more details on any of these?"
                    )
                })
            else:
                prompt.append({
                    "role": "system",
                    "content": (
                        f"Today is {today}. I couldn't find any events on your schedule. "
                        "Would you like to add a new event or perhaps check again later?"
                    )
                })
            prompt.append({"role": "user", "content": message})
        
        else:
            # --- Handling Event Additions ---
            is_event = await self._is_event_addition(message)
            
            if is_event:
                # Check if the event details are complete using a new classifier
                is_complete = await self._is_event_complete(message)
                if not is_complete:
                    # If incomplete, work with a draft event and ask for more details
                    print('Processing partial event addition...')
                    draft_event = await self.firestore.get_draft_event(user_id)
                    if draft_event:
                        updated_event = self._merge_event_details(draft_event, message)
                        await self.firestore.update_draft_event(user_id, updated_event)
                        response_text = (
                            "I've updated your draft event with what you just mentioned. "
                            "Could you please share more details—like the exact time, location, or who will be joining? "
                            "This will help me set it up perfectly for you."
                        )
                    else:
                        # Create a new draft if none exists
                        draft_event = {"content": message, "status": "draft", "timestamp": today}
                        await self.firestore.create_draft_event(user_id, draft_event)
                        response_text = (
                            "I noticed you mentioned an event, but I need a bit more info. "
                            "Could you please tell me the time, location, or any other details so I can add it properly?"
                        )
                else:
                    # For complete event details, save the event and clear any draft
                    print('Processing complete event addition...')
                    embedding = await self.ollama.get_embedding(message_with_date)
                    event_data = {"content": message, "status": "complete", "timestamp": today}
                    await self.pinecone.store_event(user_id, event_data, embedding)
                    await self.firestore.delete_draft_event(user_id)
                    response_text = (
                        "Great! I've saved your event. Is there anything else you'd like to add or any questions about your schedule?"
                    )
                
                # Use a conversational system prompt for event additions
                system_prompt = (
                    "Hi! I'm your scheduling assistant. " + response_text
                )
                prompt = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ]
            else:
                # --- Handling General Conversation ---
                print('Processing general chat...')
                system_prompt = (
                    "Hello! I'm here to help you with your schedule and any questions you might have about your calendar. "
                    "Feel free to ask me anything or let me know how I can assist you today."
                )
                prompt = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ]
        
        # Generate a response using the conversational prompt
        response = await self.groq.generate(prompt)
        
        # Save the conversation for future context and to improve suggestions
        await self.firestore.store_conversation(user_id, message, response)
        
        return response

    async def _is_schedule_query(self, message: str) -> bool:
        prompt = [
            {"role": "system", "content": (
                "You are a classifier that determines if a message is specifically asking about schedules, events, or calendar information.\n"
                "Return ONLY 'true' or 'false'.\n\n"
                "Return 'true' ONLY if the message:\n"
                "- Explicitly asks about upcoming events, appointments, or meetings\n"
                "- Asks about what's on the schedule for a specific time/date\n"
                "- Inquires about calendar availability\n"
                "- Asks about event details or timing\n\n"
                "Return 'false' for general or casual mentions."
            )},
            {"role": "user", "content": message}
        ]
        response = await self.groq.generate(prompt)
        return 'true' in response.lower()

    async def _is_event_addition(self, message: str) -> bool:
        prompt = [
            {"role": "system", "content": (
                "You are a classifier that determines if a message is explicitly adding a new event or appointment to a schedule.\n"
                "Return ONLY 'true' or 'false'.\n\n"
                "Return 'true' ONLY if the message:\n"
                "- Clearly states adding a new event or appointment\n"
                "- Contains specific event details (time, date, description)\n"
                "- Uses scheduling-related verbs (schedule, add, book, set up)\n"
                "- Includes both what the event is and when it occurs\n\n"
                "Return 'false' otherwise."
            )},
            {"role": "user", "content": message}
        ]
        response = await self.groq.generate(prompt)
        return 'true' in response.lower()

    async def _is_event_complete(self, message: str) -> bool:
        # Classifier to check if the event message includes complete details
        prompt = [
            {"role": "system", "content": (
                "You are a classifier that checks whether an event message includes all necessary details.\n"
                "Return ONLY 'true' or 'false'.\n\n"
                "Return 'true' ONLY if the message includes:\n"
                "- A specific date and time\n"
                "- Details about the event (e.g., meeting purpose, participants, location)\n\n"
                "Return 'false' if any of these details are missing."
            )},
            {"role": "user", "content": message}
        ]
        response = await self.groq.generate(prompt)
        return 'true' in response.lower()

    def _merge_event_details(self, existing_event: Dict, new_message: str) -> Dict:
        # Merge new details with the existing draft content in a conversational manner
        merged_content = f"{existing_event.get('content', '')} {new_message}"
        existing_event['content'] = merged_content.strip()
        return existing_event

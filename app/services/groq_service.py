from groq import Groq
from app.core.config import get_settings

settings = get_settings()

class GroqService:
    def __init__(self):
        self.client = Groq(
            api_key=settings.GROQ_API_KEY,
        )
        self.model = settings.MODEL_NAME

    async def generate(self, messages):
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
            )
            return completion.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}") 
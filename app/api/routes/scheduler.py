from fastapi import APIRouter, Depends
from app.services.auth_service import AuthService
from app.services.scheduler_service import SchedulerService
from app.models.schemas import ChatMessage, ChatResponse

router = APIRouter()
scheduler_service = SchedulerService()

@router.post("/chat", response_model=ChatResponse)
async def chat(
    message: ChatMessage,
    token = Depends(AuthService.verify_token)
):
    user_id = token["uid"]
    response = await scheduler_service.process_message(user_id, message.content)
    return ChatResponse(message=response) 
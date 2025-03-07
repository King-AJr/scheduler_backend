from fastapi import APIRouter, Depends
from app.services.firestore_service import FirestoreService

router = APIRouter()

@router.get("/chat/history/{user_id}")
async def get_chat_history(user_id: str, firestore_service: FirestoreService = Depends()):
    """
    Retrieve the last 20 conversation pairs for a user.
    
    Args:
        user_id (str): The user's unique identifier
        
    Returns:
        List[Dict]: List of conversation pairs containing messages and responses
    """
    conversations = await firestore_service.get_user_conversations(user_id, limit=20)
    return conversations 
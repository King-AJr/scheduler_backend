from fastapi import APIRouter, HTTPException, Depends
from app.models.schedule import ScheduleQuery, ScheduleResponse
from app.services.schedule_service import ScheduleService
import logging


router = APIRouter()

@router.post("/schedule", response_model=ScheduleResponse)
async def get_schedule(
    query: ScheduleQuery,
    schedule_service: ScheduleService = Depends()
):
    try:
        response = await schedule_service.get_schedule_from_query(
            user_id=query.user_id,
            query=query.query
        )
        return response
    except Exception as e:
        print(f"Error: {str(e)}")  # For development
        logging.error(f"Schedule API error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 
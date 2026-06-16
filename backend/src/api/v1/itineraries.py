"""Itinerary v1 API endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends, Query

from api.deps import get_current_user, get_itinerary_service
from api.v1.schemas import (
    CreateItineraryRequest,
    ItineraryResponse,
    UpdateFavoriteRequest,
)
from core.exceptions import NotFoundException
from core.responses import success_response
from models import User
from services import ItineraryService

router = APIRouter(prefix="/itineraries", tags=["itineraries"])


@router.get("")
async def list_itineraries(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_current_user),
    service: ItineraryService = Depends(get_itinerary_service),
):
    """List itineraries for the current user."""
    itineraries = await service.list_by_user(user.id, limit=limit, offset=offset)
    return success_response(
        data=[ItineraryResponse.model_validate(i).model_dump() for i in itineraries],
        meta={"limit": limit, "offset": offset},
    )


@router.post("")
async def create_itinerary(
    body: CreateItineraryRequest,
    user: User = Depends(get_current_user),
    service: ItineraryService = Depends(get_itinerary_service),
):
    """Create an itinerary for a conversation."""
    itinerary = await service.create(
        user_id=user.id,
        conversation_id=body.conversation_id,
        destination=body.destination,
        days=body.days,
        content=body.content,
        proposal_text=body.proposal_text,
        job_id=body.job_id,
    )
    return success_response(
        data=ItineraryResponse.model_validate(itinerary).model_dump(),
        status_code=201,
    )


@router.post("/{itinerary_id}/favorite")
async def set_favorite(
    itinerary_id: UUID,
    body: UpdateFavoriteRequest,
    user: User = Depends(get_current_user),
    service: ItineraryService = Depends(get_itinerary_service),
):
    """Mark an itinerary as favorite or not."""
    itinerary = await service.repo.get_by_id(itinerary_id)
    if itinerary is None or itinerary.user_id != user.id:
        raise NotFoundException("Itinerary", itinerary_id)
    await service.set_favorite(itinerary_id, body.is_favorite)
    return success_response(message="Favorite status updated")

"""
预订模拟接口 — Layer 6 工具调用层。

提供机票/酒店/门票/餐厅 4 类预订 API，返回模拟数据。
LLM 通过 Function Calling 调用，标注 source=mock。
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from api.deps import get_current_user
from core.responses import success_response
from models import User

router = APIRouter(prefix="/bookings", tags=["bookings"])

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class FlightSearchRequest(BaseModel):
    origin: str = Field(..., description="出发城市")
    dest: str = Field(..., description="目的地")
    date: str = Field(..., description="日期 YYYY-MM-DD")


class FlightResult(BaseModel):
    flight_no: str
    departure: str
    arrival: str
    duration: str
    price: float
    airline: str
    source: str = "mock"


class HotelSearchRequest(BaseModel):
    city: str
    checkin: str
    checkout: str
    budget_per_night: Optional[float] = None
    guests: int = 1


class HotelResult(BaseModel):
    name: str
    district: str
    price_per_night: float
    rating: float
    has_breakfast: bool
    has_parking: bool
    distance_to_center: str
    source: str = "mock"


class TicketCheckRequest(BaseModel):
    poi_name: str
    date: str


class TicketResult(BaseModel):
    poi_name: str
    ticket_price: float
    available: bool
    need_reservation: bool
    source: str = "mock"


class RestaurantReserveRequest(BaseModel):
    restaurant: str
    date: str
    time: str
    persons: int


class ReserveResult(BaseModel):
    restaurant: str
    reservation_id: str
    status: str = "confirmed"
    source: str = "mock"


# ---------------------------------------------------------------------------
# Mock data generators
# ---------------------------------------------------------------------------

AIRLINES = ["中国国航 CA", "东方航空 MU", "南方航空 CZ", "海南航空 HU", "春秋航空 9C"]
HOTEL_CHAINS = ["如家", "汉庭", "全季", "亚朵", "希尔顿欢朋", "万豪", "洲际"]

MOCK_FLIGHTS: dict[str, list[dict]] = {
    "北京-成都": [
        {"no": "CA4101", "dep": "07:30", "arr": "10:15", "dur": "2h45m", "price": 680},
        {"no": "MU5210", "dep": "14:00", "arr": "16:45", "dur": "2h45m", "price": 520},
        {"no": "CZ8842", "dep": "19:30", "arr": "22:15", "dur": "2h45m", "price": 380},
    ],
}

MOCK_HOTELS: dict[str, list[dict]] = {
    "成都": [
        {"name": "春熙路亚朵酒店", "district": "锦江区", "price": 350, "rating": 4.7, "breakfast": True, "parking": True},
        {"name": "宽窄巷子全季酒店", "district": "青羊区", "price": 280, "rating": 4.5, "breakfast": True, "parking": False},
        {"name": "天府广场汉庭酒店", "district": "锦江区", "price": 180, "rating": 4.2, "breakfast": False, "parking": False},
        {"name": "成都希尔顿酒店", "district": "高新区", "price": 680, "rating": 4.9, "breakfast": True, "parking": True},
    ],
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/flights/search")
async def search_flights(
    body: FlightSearchRequest,
    user: User = Depends(get_current_user),
):
    """搜索机票 — 模拟数据。"""
    key = f"{body.origin}-{body.dest}"
    flights = MOCK_FLIGHTS.get(key, [
        {"no": f"CA{random.randint(1000,9999)}", "dep": "08:00", "arr": "11:00", "dur": "3h", "price": round(random.uniform(300, 900), 0)},
        {"no": f"MU{random.randint(1000,9999)}", "dep": "14:00", "arr": "17:00", "dur": "3h", "price": round(random.uniform(250, 700), 0)},
    ])

    results = [
        FlightResult(
            flight_no=f["no"],
            departure=f"{body.date}T{f['dep']}:00",
            arrival=f"{body.date}T{f['arr']}:00",
            duration=f["dur"],
            price=f["price"],
            airline=random.choice(AIRLINES),
        )
        for f in flights
    ]

    return success_response(
        data={"flights": [r.model_dump() for r in results]},
        message=f"找到 {len(results)} 个航班"
    )


@router.post("/hotels/search")
async def search_hotels(
    body: HotelSearchRequest,
    user: User = Depends(get_current_user),
):
    """搜索酒店 — 模拟数据。"""
    hotels = MOCK_HOTELS.get(body.city, [
        {"name": f"{body.city}{random.choice(HOTEL_CHAINS)}", "district": "市中心", "price": round(random.uniform(150, 600), 0), "rating": round(random.uniform(4.0, 4.8), 1), "breakfast": random.choice([True, False]), "parking": random.choice([True, False])}
        for _ in range(3)
    ])

    if body.budget_per_night:
        hotels = [h for h in hotels if h["price"] <= body.budget_per_night]

    results = [
        HotelResult(
            name=h["name"],
            district=h["district"],
            price_per_night=h["price"],
            rating=h["rating"],
            has_breakfast=h["breakfast"],
            has_parking=h["parking"],
            distance_to_center=f"{random.randint(1, 5)}km",
        )
        for h in hotels
    ]

    nights = max(1, (datetime.strptime(body.checkout, "%Y-%m-%d") - datetime.strptime(body.checkin, "%Y-%m-%d")).days)
    total = sum(r.price_per_night for r in results[:1]) * nights

    return success_response(
        data={
            "hotels": [r.model_dump() for r in results],
            "nights": nights,
            "estimated_total": total,
        },
        message=f"找到 {len(results)} 家酒店"
    )


@router.post("/attractions/tickets")
async def check_tickets(
    body: TicketCheckRequest,
    user: User = Depends(get_current_user),
):
    """查询门票 — 模拟数据。"""
    return success_response(
        data=TicketResult(
            poi_name=body.poi_name,
            ticket_price=round(random.uniform(20, 150), 0),
            available=random.random() > 0.1,
            need_reservation=random.random() > 0.5,
        ).model_dump(),
        message="门票信息"
    )


@router.post("/restaurants/reserve")
async def reserve_restaurant(
    body: RestaurantReserveRequest,
    user: User = Depends(get_current_user),
):
    """预订餐厅 — 模拟数据。"""
    reservation_id = f"RSV{random.randint(10000, 99999)}"
    return success_response(
        data=ReserveResult(
            restaurant=body.restaurant,
            reservation_id=reservation_id,
        ).model_dump(),
        message=f"已预订 {body.restaurant}，编号 {reservation_id}"
    )

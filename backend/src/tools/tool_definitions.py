"""OpenAI-compatible tool schema definitions for the travel agent.

Defines 11 tool schemas used by the LangGraph tool-calling layer. Each schema
follows the function-calling format expected by LLM providers and LangChain
ToolNode:

    {
        "type": "function",
        "function": {
            "name": "...",
            "description": "...",
            "parameters": <JSON Schema>
        }
    }
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class WeatherParams(BaseModel):
    city: str = Field(..., description="城市名称，例如：北京")
    date: str | None = Field(None, description="日期，格式 YYYY-MM-DD，默认当天")


class ReservationParams(BaseModel):
    poi_name: str = Field(..., description="景点或场馆名称")
    city: str | None = Field(None, description="城市名称")
    date: str | None = Field(None, description="计划游玩日期 YYYY-MM-DD")


class RouteParams(BaseModel):
    origin: str = Field(..., description="起点名称或坐标")
    destination: str = Field(..., description="终点名称或坐标")
    city: str | None = Field(None, description="城市名称")
    mode: str = Field("transit", description="交通方式：transit/drive/walk/taxi")


class RestaurantParams(BaseModel):
    city: str = Field(..., description="城市名称")
    area: str | None = Field(None, description="商圈或区域")
    cuisine: str | None = Field(None, description="菜系偏好")
    budget_per_person: float | None = Field(None, description="人均预算（元）")


class HotelParams(BaseModel):
    city: str = Field(..., description="城市名称")
    area: str | None = Field(None, description="商圈或区域")
    check_in: str | None = Field(None, description="入住日期 YYYY-MM-DD")
    check_out: str | None = Field(None, description="离店日期 YYYY-MM-DD")
    budget_per_night: float | None = Field(None, description="每晚预算（元）")


class QueueTimeParams(BaseModel):
    poi_name: str = Field(..., description="景点或场馆名称")
    city: str = Field(..., description="城市名称")
    date: str | None = Field(None, description="日期 YYYY-MM-DD")


class TicketLinkParams(BaseModel):
    poi_name: str = Field(..., description="景点或场馆名称")
    city: str | None = Field(None, description="城市名称")


class LocalEventsParams(BaseModel):
    city: str = Field(..., description="城市名称")
    date: str | None = Field(None, description="日期 YYYY-MM-DD")


class EmergencyServicesParams(BaseModel):
    city: str = Field(..., description="城市名称")
    area: str | None = Field(None, description="区域或地址")


class POIDetailParams(BaseModel):
    poi_name: str = Field(..., description="POI 名称")
    city: str | None = Field(None, description="城市名称")


class UpdateProfileParams(BaseModel):
    key: str = Field(..., description="画像字段名，例如：budget_per_day")
    value: str | int | float | bool = Field(..., description="字段值")


def _schema(model: type[BaseModel]) -> dict[str, Any]:
    """Build a JSON schema dict and strip the title for compactness."""
    schema = model.model_json_schema()
    schema.pop("title", None)
    return schema


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市未来天气，用于行程衣物和室内/室外景点建议。",
            "parameters": _schema(WeatherParams),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_reservation",
            "description": "查询景点、博物馆或演出是否需要提前预约，以及当前可预约时段。",
            "parameters": _schema(ReservationParams),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_route",
            "description": "查询两点之间的推荐交通路线与通勤时间。",
            "parameters": _schema(RouteParams),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_restaurants",
            "description": "根据城市、区域、菜系和预算推荐餐厅。",
            "parameters": _schema(RestaurantParams),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_hotels",
            "description": "根据城市、区域、入住日期和预算推荐酒店。",
            "parameters": _schema(HotelParams),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_queue_time",
            "description": "查询景点当前或预计排队时间。",
            "parameters": _schema(QueueTimeParams),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ticket_link",
            "description": "获取景点官方购票或预约链接。",
            "parameters": _schema(TicketLinkParams),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_local_events",
            "description": "查询城市近期活动、展览、演出或节庆。",
            "parameters": _schema(LocalEventsParams),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_emergency_services",
            "description": "查询城市紧急服务信息：医院、派出所、领事馆等。",
            "parameters": _schema(EmergencyServicesParams),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_poi_detail",
            "description": "获取 POI 详细信息：开放时间、门票价格、建议游玩时长。",
            "parameters": _schema(POIDetailParams),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_user_profile",
            "description": "更新用户画像中的单个字段。",
            "parameters": _schema(UpdateProfileParams),
        },
    },
]

# Map name -> schema for quick lookup.
TOOL_NAME_TO_SCHEMA: dict[str, dict[str, Any]] = {tool["function"]["name"]: tool for tool in TOOLS}
TOOL_NAME_TO_MODEL: dict[str, type[BaseModel]] = {
    "get_weather": WeatherParams,
    "check_reservation": ReservationParams,
    "get_route": RouteParams,
    "find_restaurants": RestaurantParams,
    "find_hotels": HotelParams,
    "get_queue_time": QueueTimeParams,
    "get_ticket_link": TicketLinkParams,
    "get_local_events": LocalEventsParams,
    "get_emergency_services": EmergencyServicesParams,
    "get_poi_detail": POIDetailParams,
    "update_user_profile": UpdateProfileParams,
}

__all__ = ["TOOLS", "TOOL_NAME_TO_MODEL", "TOOL_NAME_TO_SCHEMA"]

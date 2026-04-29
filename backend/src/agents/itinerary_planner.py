"""Itinerary Planner Agent - LLM-driven direct planning from raw POI data."""
import math
import random
from difflib import SequenceMatcher
from typing import Optional


from core.llm_client import llm
from schemas import (
    UserProfile, ScoredPOI, WeatherDay, DayPlan, Activity, Location, TravelContext,
)
from skills.route_calculation import RouteCalculationSkill


class ItineraryPlannerAgent:
    """Plan itinerary using LLM-driven planning from raw POI data."""

    # City-specific fallback attractions when POI search returns empty
    # All POIs include location (lat/lng) and ticket_price for route + budget calc
    CITY_DEFAULTS: dict[str, list[ScoredPOI]] = {
        "上海": [
            ScoredPOI(name="外滩", category="attraction", score=0.9, description="万国建筑博览群，上海地标", tags=["夜景", "历史", "拍照"], best_time="晚上", recommended_hours="1-2小时", area="外滩", indoor_outdoor="outdoor", location=Location(lat=31.2397, lng=121.4998), ticket_price=0),
            ScoredPOI(name="东方明珠", category="attraction", score=0.9, description="上海标志性建筑，俯瞰全城", tags=["地标", "观景", "拍照"], best_time="全天", recommended_hours="2-3小时", area="陆家嘴", indoor_outdoor="mixed", location=Location(lat=31.2397, lng=121.4998), ticket_price=199),
            ScoredPOI(name="豫园", category="attraction", score=0.85, description="明代园林，江南古典园林代表", tags=["园林", "历史", "文化"], best_time="上午", recommended_hours="2小时", area="城隍庙", indoor_outdoor="mixed", location=Location(lat=31.2273, lng=121.4920), ticket_price=40),
            ScoredPOI(name="南京路步行街", category="attraction", score=0.8, description="中华商业第一街", tags=["购物", "美食", "逛街"], best_time="下午", recommended_hours="2小时", area="南京东路", indoor_outdoor="outdoor", location=Location(lat=31.2346, lng=121.4750), ticket_price=0),
            ScoredPOI(name="田子坊", category="attraction", score=0.8, description="石库门里弄，创意艺术区", tags=["文艺", "拍照", "小巷"], best_time="下午", recommended_hours="1-2小时", area="打浦桥", indoor_outdoor="mixed", location=Location(lat=31.2106, lng=121.4692), ticket_price=0),
            ScoredPOI(name="新天地", category="attraction", score=0.8, description="石库门与现代融合的时尚地标", tags=["夜生活", "餐饮", "文化"], best_time="晚上", recommended_hours="2小时", area="新天地", indoor_outdoor="mixed", location=Location(lat=31.2225, lng=121.4753), ticket_price=0),
            ScoredPOI(name="陆家嘴", category="attraction", score=0.85, description="金融中心，摩天大楼群", tags=["地标", "现代", "观景"], best_time="全天", recommended_hours="2-3小时", area="陆家嘴", indoor_outdoor="mixed", location=Location(lat=31.2397, lng=121.4998), ticket_price=0),
            ScoredPOI(name="上海博物馆", category="attraction", score=0.8, description="中国古代艺术博物馆", tags=["博物馆", "历史", "文化"], best_time="上午", recommended_hours="2-3小时", area="人民广场", indoor_outdoor="indoor", location=Location(lat=31.2283, lng=121.4453), ticket_price=0),
            ScoredPOI(name="城隍庙", category="attraction", score=0.75, description="道教庙宇，特色小吃街", tags=["美食", "文化", "传统"], best_time="下午", recommended_hours="1-2小时", area="城隍庙", indoor_outdoor="mixed", location=Location(lat=31.2273, lng=121.4920), ticket_price=10),
            ScoredPOI(name="武康路", category="attraction", score=0.8, description="历史风貌街区，梧桐洋房", tags=["历史", "拍照", "文艺"], best_time="上午", recommended_hours="1-2小时", area="徐汇区", indoor_outdoor="outdoor", location=Location(lat=31.2165, lng=121.4375), ticket_price=0),
            ScoredPOI(name="1933老场坊", category="attraction", score=0.7, description="创意园区，工业风建筑", tags=["建筑", "拍照", "艺术"], best_time="下午", recommended_hours="1-2小时", area="虹口区", indoor_outdoor="mixed", location=Location(lat=31.2578, lng=121.4856), ticket_price=0),
            ScoredPOI(name="上海迪士尼", category="attraction", score=0.9, description="中国大陆首座迪士尼乐园", tags=["亲子", "娱乐", "主题公园"], best_time="全天", recommended_hours="全天", area="浦东新区", indoor_outdoor="mixed", location=Location(lat=31.1413, lng=121.6618), ticket_price=475),
            ScoredPOI(name="思南公馆", category="attraction", score=0.75, description="花园洋房建筑群", tags=["历史", "建筑", "休闲"], best_time="下午", recommended_hours="1-2小时", area="复兴中路", indoor_outdoor="mixed", location=Location(lat=31.2183, lng=121.4692), ticket_price=0),
            ScoredPOI(name="M50创意园", category="attraction", score=0.7, description="艺术创意园区", tags=["艺术", "画廊", "文艺"], best_time="下午", recommended_hours="1-2小时", area="莫干山路", indoor_outdoor="mixed", location=Location(lat=31.2469, lng=121.4453), ticket_price=0),
            ScoredPOI(name="七宝老街", category="attraction", score=0.7, description="古镇风貌美食街", tags=["美食", "古镇", "传统"], best_time="下午", recommended_hours="1-2小时", area="闵行区", indoor_outdoor="outdoor", location=Location(lat=31.1575, lng=121.3531), ticket_price=0),
            ScoredPOI(name="南翔馒头店", category="restaurant", score=0.75, description="百年老字号小笼包", tags=["小笼包", "老字号", "必吃"], best_time="午餐", recommended_hours="1小时", area="城隍庙", indoor_outdoor="indoor", location=Location(lat=31.2273, lng=121.4920), ticket_price=0),
            ScoredPOI(name="小杨生煎", category="restaurant", score=0.75, description="上海人气生煎连锁", tags=["生煎", "小吃", "人气"], best_time="午餐", recommended_hours="1小时", area="多家分店", indoor_outdoor="indoor", location=Location(lat=31.2346, lng=121.4750), ticket_price=0),
            ScoredPOI(name="老正兴菜馆", category="restaurant", score=0.7, description="上海本帮菜老字号", tags=["本帮菜", "老字号", "晚餐"], best_time="晚餐", recommended_hours="1.5小时", area="黄浦区", indoor_outdoor="indoor", location=Location(lat=31.2346, lng=121.4750), ticket_price=0),
            ScoredPOI(name="云南路美食街", category="restaurant", score=0.7, description="上海传统美食一条街", tags=["美食街", "小吃", "老字号"], best_time="晚餐", recommended_hours="2小时", area="黄浦区", indoor_outdoor="outdoor", location=Location(lat=31.2346, lng=121.4750), ticket_price=0),
        ],
        "北京": [
            ScoredPOI(name="故宫", category="attraction", score=0.95, description="明清皇宫，世界文化遗产", tags=["历史", "文化", "皇家"], best_time="上午", recommended_hours="半天", area="东城区", indoor_outdoor="mixed", location=Location(lat=39.916345, lng=116.397155), ticket_price=60),
            ScoredPOI(name="长城", category="attraction", score=0.95, description="世界文化遗产，中华民族象征", tags=["历史", "登山", "观景"], best_time="上午", recommended_hours="半天", area="延庆区", indoor_outdoor="outdoor", location=Location(lat=40.359580, lng=116.019967), ticket_price=40),
            ScoredPOI(name="天坛", category="attraction", score=0.9, description="明清祭天建筑", tags=["历史", "建筑", "文化"], best_time="上午", recommended_hours="2-3小时", area="东城区", indoor_outdoor="mixed", location=Location(lat=39.883455, lng=116.406588), ticket_price=34),
            ScoredPOI(name="颐和园", category="attraction", score=0.9, description="皇家园林，中国园林巅峰", tags=["园林", "湖景", "皇家"], best_time="下午", recommended_hours="半天", area="海淀区", indoor_outdoor="mixed", location=Location(lat=39.999982, lng=116.275461), ticket_price=30),
            ScoredPOI(name="王府井", category="attraction", score=0.8, description="著名商业街区", tags=["购物", "美食", "逛街"], best_time="晚上", recommended_hours="2小时", area="东城区", indoor_outdoor="mixed", location=Location(lat=39.911057, lng=116.410876), ticket_price=0),
            ScoredPOI(name="798艺术区", category="attraction", score=0.8, description="当代艺术中心", tags=["艺术", "画廊", "文艺"], best_time="下午", recommended_hours="2-3小时", area="朝阳区", indoor_outdoor="mixed", location=Location(lat=39.985326, lng=116.497772), ticket_price=0),
            ScoredPOI(name="南锣鼓巷", category="attraction", score=0.75, description="老北京胡同文化", tags=["胡同", "小吃", "文化"], best_time="下午", recommended_hours="1-2小时", area="东城区", indoor_outdoor="outdoor", location=Location(lat=39.937243, lng=116.403076), ticket_price=0),
            ScoredPOI(name="鸟巢/水立方", category="attraction", score=0.85, description="奥运场馆地标", tags=["地标", "建筑", "现代"], best_time="晚上", recommended_hours="1-2小时", area="朝阳区", indoor_outdoor="outdoor", location=Location(lat=39.992928, lng=116.396531), ticket_price=50),
            ScoredPOI(name="什刹海", category="attraction", score=0.8, description="老北京风貌区，酒吧街", tags=["湖景", "夜生活", "胡同"], best_time="晚上", recommended_hours="2小时", area="西城区", indoor_outdoor="mixed", location=Location(lat=39.940430, lng=116.385331), ticket_price=0),
            ScoredPOI(name="国家博物馆", category="attraction", score=0.85, description="中国历史文物展览", tags=["博物馆", "历史", "文化"], best_time="上午", recommended_hours="2-3小时", area="天安门", indoor_outdoor="indoor", location=Location(lat=39.905490, lng=116.397650), ticket_price=0),
            ScoredPOI(name="景山公园", category="attraction", score=0.8, description="俯瞰故宫全景最佳位置", tags=["观景", "园林", "拍照"], best_time="傍晚", recommended_hours="1小时", area="西城区", indoor_outdoor="outdoor", location=Location(lat=39.924418, lng=116.397087), ticket_price=2),
            ScoredPOI(name="北海公园", category="attraction", score=0.75, description="皇家园林，白塔", tags=["园林", "湖景", "休闲"], best_time="下午", recommended_hours="2小时", area="西城区", indoor_outdoor="mixed", location=Location(lat=39.924175, lng=116.388862), ticket_price=10),
            ScoredPOI(name="前门大街", category="attraction", score=0.75, description="老北京商业街", tags=["购物", "老字号", "文化"], best_time="下午", recommended_hours="1-2小时", area="前门", indoor_outdoor="outdoor", location=Location(lat=39.899841, lng=116.398211), ticket_price=0),
            ScoredPOI(name="雍和宫", category="attraction", score=0.75, description="藏传佛教寺院", tags=["宗教", "文化", "建筑"], best_time="上午", recommended_hours="1-2小时", area="东城区", indoor_outdoor="mixed", location=Location(lat=39.947200, lng=116.417755), ticket_price=25),
            ScoredPOI(name="恭王府", category="attraction", score=0.8, description="清代王府建筑", tags=["历史", "建筑", "园林"], best_time="下午", recommended_hours="2小时", area="西城区", indoor_outdoor="mixed", location=Location(lat=39.936185, lng=116.386460), ticket_price=40),
            ScoredPOI(name="全聚德", category="restaurant", score=0.8, description="北京烤鸭老字号", tags=["烤鸭", "老字号", "必吃"], best_time="晚餐", recommended_hours="1.5小时", area="前门", indoor_outdoor="indoor", location=Location(lat=39.899841, lng=116.398211), ticket_price=0),
            ScoredPOI(name="四季民福", category="restaurant", score=0.8, description="人气北京烤鸭店", tags=["烤鸭", "人气", "晚餐"], best_time="晚餐", recommended_hours="1.5小时", area="东城区", indoor_outdoor="indoor", location=Location(lat=39.916345, lng=116.397155), ticket_price=0),
            ScoredPOI(name="护国寺小吃", category="restaurant", score=0.75, description="北京传统小吃集合", tags=["小吃", "传统", "早餐"], best_time="早餐", recommended_hours="1小时", area="西城区", indoor_outdoor="indoor", location=Location(lat=39.940430, lng=116.385331), ticket_price=0),
            ScoredPOI(name="方砖厂炸酱面", category="restaurant", score=0.75, description="老北京炸酱面名店", tags=["炸酱面", "老北京", "午餐"], best_time="午餐", recommended_hours="1小时", area="东城区", indoor_outdoor="indoor", location=Location(lat=39.937243, lng=116.403076), ticket_price=0),
            ScoredPOI(name="姚记炒肝", category="restaurant", score=0.75, description="北京传统炒肝店", tags=["炒肝", "老北京", "早餐"], best_time="早餐", recommended_hours="1小时", area="东城区", indoor_outdoor="indoor", location=Location(lat=39.937243, lng=116.403076), ticket_price=0),
        ],
        "广州": [
            ScoredPOI(name="广州塔", category="attraction", score=0.9, description="小蛮腰，广州地标", tags=["地标", "观景", "夜景"], best_time="晚上", recommended_hours="2-3小时", area="海珠区", indoor_outdoor="mixed", location=Location(lat=23.1065, lng=113.3255), ticket_price=150),
            ScoredPOI(name="陈家祠", category="attraction", score=0.85, description="岭南建筑艺术明珠", tags=["建筑", "文化", "历史"], best_time="上午", recommended_hours="1-2小时", area="荔湾区", indoor_outdoor="mixed", location=Location(lat=23.1288, lng=113.2458), ticket_price=10),
            ScoredPOI(name="沙面", category="attraction", score=0.85, description="欧陆风情建筑群", tags=["建筑", "拍照", "历史"], best_time="下午", recommended_hours="1-2小时", area="荔湾区", indoor_outdoor="outdoor", location=Location(lat=23.1092, lng=113.2447), ticket_price=0),
            ScoredPOI(name="北京路步行街", category="attraction", score=0.8, description="千年古道商业街", tags=["购物", "美食", "逛街"], best_time="下午", recommended_hours="2小时", area="越秀区", indoor_outdoor="outdoor", location=Location(lat=23.1247, lng=113.2669), ticket_price=0),
            ScoredPOI(name="上下九步行街", category="attraction", score=0.75, description="西关风情商业街", tags=["美食", "购物", "传统"], best_time="下午", recommended_hours="2小时", area="荔湾区", indoor_outdoor="outdoor", location=Location(lat=23.1175, lng=113.2458), ticket_price=0),
            ScoredPOI(name="越秀公园", category="attraction", score=0.8, description="城市绿肺，五羊雕像", tags=["公园", "休闲", "地标"], best_time="上午", recommended_hours="2小时", area="越秀区", indoor_outdoor="outdoor", location=Location(lat=23.1408, lng=113.2597), ticket_price=0),
            ScoredPOI(name="珠江夜游", category="attraction", score=0.85, description="珠江两岸夜景", tags=["夜景", "游船", "浪漫"], best_time="晚上", recommended_hours="1-2小时", area="珠江", indoor_outdoor="outdoor", location=Location(lat=23.1065, lng=113.3255), ticket_price=78),
            ScoredPOI(name="广东省博物馆", category="attraction", score=0.8, description="广东历史文化展览", tags=["博物馆", "历史", "文化"], best_time="上午", recommended_hours="2-3小时", area="珠江新城", indoor_outdoor="indoor", location=Location(lat=23.1108, lng=113.3242), ticket_price=0),
            ScoredPOI(name="白云山", category="attraction", score=0.8, description="羊城第一秀", tags=["登山", "自然", "观景"], best_time="上午", recommended_hours="半天", area="白云区", indoor_outdoor="outdoor", location=Location(lat=23.1833, lng=113.3000), ticket_price=5),
            ScoredPOI(name="长隆欢乐世界", category="attraction", score=0.85, description="大型主题乐园", tags=["亲子", "娱乐", "刺激"], best_time="全天", recommended_hours="全天", area="番禺区", indoor_outdoor="mixed", location=Location(lat=22.9986, lng=113.3242), ticket_price=250),
            ScoredPOI(name="石室圣心大教堂", category="attraction", score=0.8, description="哥特式建筑，全石结构", tags=["建筑", "宗教", "拍照"], best_time="上午", recommended_hours="1小时", area="越秀区", indoor_outdoor="mixed", location=Location(lat=23.1175, lng=113.2597), ticket_price=0),
            ScoredPOI(name="红砖厂", category="attraction", score=0.75, description="创意艺术园区", tags=["艺术", "文艺", "拍照"], best_time="下午", recommended_hours="1-2小时", area="天河区", indoor_outdoor="mixed", location=Location(lat=23.1065, lng=113.3255), ticket_price=0),
            ScoredPOI(name="永庆坊", category="attraction", score=0.8, description="西关风情文创街区", tags=["文化", "建筑", "美食"], best_time="下午", recommended_hours="2小时", area="荔湾区", indoor_outdoor="mixed", location=Location(lat=23.1175, lng=113.2458), ticket_price=0),
            ScoredPOI(name="花城广场", category="attraction", score=0.8, description="广州新中轴线", tags=["现代", "地标", "夜景"], best_time="晚上", recommended_hours="1-2小时", area="珠江新城", indoor_outdoor="outdoor", location=Location(lat=23.1208, lng=113.3255), ticket_price=0),
            ScoredPOI(name="华南植物园", category="attraction", score=0.75, description="大型植物园", tags=["自然", "植物", "休闲"], best_time="上午", recommended_hours="2-3小时", area="天河区", indoor_outdoor="outdoor", location=Location(lat=23.1833, lng=113.3667), ticket_price=20),
            ScoredPOI(name="点都德", category="restaurant", score=0.8, description="广式早茶连锁名店", tags=["早茶", "虾饺", "必吃"], best_time="早餐", recommended_hours="1.5小时", area="荔湾区", indoor_outdoor="indoor", location=Location(lat=23.1288, lng=113.2458), ticket_price=0),
            ScoredPOI(name="陶陶居", category="restaurant", score=0.8, description="广州早茶老字号", tags=["早茶", "老字号", "点心"], best_time="早餐", recommended_hours="1.5小时", area="荔湾区", indoor_outdoor="indoor", location=Location(lat=23.1175, lng=113.2458), ticket_price=0),
            ScoredPOI(name="广州酒家", category="restaurant", score=0.75, description="粤菜老字号", tags=["粤菜", "老字号", "晚餐"], best_time="晚餐", recommended_hours="1.5小时", area="荔湾区", indoor_outdoor="indoor", location=Location(lat=23.1288, lng=113.2458), ticket_price=0),
            ScoredPOI(name="银记肠粉", category="restaurant", score=0.75, description="广州肠粉名店", tags=["肠粉", "小吃", "早餐"], best_time="早餐", recommended_hours="1小时", area="荔湾区", indoor_outdoor="indoor", location=Location(lat=23.1175, lng=113.2458), ticket_price=0),
            ScoredPOI(name="陈添记", category="restaurant", score=0.75, description="广州传统鱼皮", tags=["鱼皮", "小吃", "老字号"], best_time="午餐", recommended_hours="1小时", area="荔湾区", indoor_outdoor="indoor", location=Location(lat=23.1175, lng=113.2458), ticket_price=0),
        ],
        "成都": [
            ScoredPOI(name="宽窄巷子", category="attraction", score=0.9, description="老成都生活体验", tags=["文化", "美食", "历史"], best_time="下午", recommended_hours="2小时", area="青羊区", indoor_outdoor="outdoor", location=Location(lat=30.6636, lng=104.0556), ticket_price=0),
            ScoredPOI(name="锦里", category="attraction", score=0.9, description="西蜀第一街", tags=["美食", "文化", "夜景"], best_time="晚上", recommended_hours="2小时", area="武侯祠", indoor_outdoor="outdoor", location=Location(lat=30.6458, lng=104.0458), ticket_price=0),
            ScoredPOI(name="大熊猫繁育基地", category="attraction", score=0.95, description="国宝大熊猫", tags=["亲子", "自然", "动物"], best_time="上午", recommended_hours="半天", area="新都区", indoor_outdoor="outdoor", location=Location(lat=30.7333, lng=104.1500), ticket_price=55),
            ScoredPOI(name="武侯祠", category="attraction", score=0.85, description="三国文化圣地", tags=["历史", "文化", "三国"], best_time="上午", recommended_hours="2小时", area="武侯区", indoor_outdoor="mixed", location=Location(lat=30.6417, lng=104.0458), ticket_price=50),
            ScoredPOI(name="杜甫草堂", category="attraction", score=0.85, description="唐代诗人故居", tags=["历史", "文化", "诗歌"], best_time="上午", recommended_hours="2小时", area="青羊区", indoor_outdoor="mixed", location=Location(lat=30.6597, lng=104.0289), ticket_price=50),
            ScoredPOI(name="春熙路", category="attraction", score=0.8, description="成都最繁华商业街", tags=["购物", "美食", "逛街"], best_time="下午", recommended_hours="2小时", area="锦江区", indoor_outdoor="outdoor", location=Location(lat=30.6569, lng=104.0814), ticket_price=0),
            ScoredPOI(name="青城山", category="attraction", score=0.85, description="道教名山，避暑胜地", tags=["登山", "道教", "自然"], best_time="全天", recommended_hours="半天", area="都江堰", indoor_outdoor="outdoor", location=Location(lat=30.9083, lng=103.5667), ticket_price=80),
            ScoredPOI(name="都江堰", category="attraction", score=0.9, description="世界文化遗产水利工程", tags=["历史", "工程", "文化"], best_time="全天", recommended_hours="半天", area="都江堰", indoor_outdoor="outdoor", location=Location(lat=31.0000, lng=103.6167), ticket_price=80),
            ScoredPOI(name="文殊院", category="attraction", score=0.8, description="佛教寺院，素斋", tags=["宗教", "文化", "美食"], best_time="上午", recommended_hours="1-2小时", area="青羊区", indoor_outdoor="mixed", location=Location(lat=30.6750, lng=104.0667), ticket_price=0),
            ScoredPOI(name="人民公园", category="attraction", score=0.8, description="老成都茶馆文化", tags=["休闲", "茶馆", "生活"], best_time="下午", recommended_hours="1-2小时", area="青羊区", indoor_outdoor="outdoor", location=Location(lat=30.6625, lng=104.0569), ticket_price=0),
            ScoredPOI(name="九眼桥", category="attraction", score=0.8, description="成都夜生活地标", tags=["夜生活", "酒吧", "夜景"], best_time="晚上", recommended_hours="2小时", area="武侯区", indoor_outdoor="outdoor", location=Location(lat=30.6375, lng=104.0875), ticket_price=0),
            ScoredPOI(name="金沙遗址", category="attraction", score=0.85, description="古蜀文明遗址", tags=["博物馆", "历史", "考古"], best_time="上午", recommended_hours="2-3小时", area="青羊区", indoor_outdoor="mixed", location=Location(lat=30.6833, lng=104.0167), ticket_price=70),
            ScoredPOI(name="太古里", category="attraction", score=0.8, description="时尚购物街区", tags=["购物", "时尚", "建筑"], best_time="下午", recommended_hours="2小时", area="锦江区", indoor_outdoor="outdoor", location=Location(lat=30.6569, lng=104.0814), ticket_price=0),
            ScoredPOI(name="黄龙溪古镇", category="attraction", score=0.75, description="千年水乡古镇", tags=["古镇", "水景", "休闲"], best_time="下午", recommended_hours="2-3小时", area="双流区", indoor_outdoor="outdoor", location=Location(lat=30.3167, lng=103.9833), ticket_price=0),
            ScoredPOI(name="东郊记忆", category="attraction", score=0.75, description="工业风创意园区", tags=["艺术", "音乐", "文艺"], best_time="下午", recommended_hours="1-2小时", area="成华区", indoor_outdoor="mixed", location=Location(lat=30.6750, lng=104.1167), ticket_price=0),
            ScoredPOI(name="蜀大侠火锅", category="restaurant", score=0.8, description="成都网红火锅", tags=["火锅", "麻辣", "晚餐"], best_time="晚餐", recommended_hours="2小时", area="锦江区", indoor_outdoor="indoor", location=Location(lat=30.6569, lng=104.0814), ticket_price=0),
            ScoredPOI(name="小龙坎", category="restaurant", score=0.8, description="成都火锅连锁", tags=["火锅", "麻辣", "人气"], best_time="晚餐", recommended_hours="2小时", area="锦江区", indoor_outdoor="indoor", location=Location(lat=30.6569, lng=104.0814), ticket_price=0),
            ScoredPOI(name="陈麻婆豆腐", category="restaurant", score=0.75, description="麻婆豆腐发源地", tags=["川菜", "老字号", "午餐"], best_time="午餐", recommended_hours="1.5小时", area="青羊区", indoor_outdoor="indoor", location=Location(lat=30.6750, lng=104.0667), ticket_price=0),
            ScoredPOI(name="钟水饺", category="restaurant", score=0.75, description="成都传统水饺", tags=["水饺", "小吃", "老字号"], best_time="午餐", recommended_hours="1小时", area="青羊区", indoor_outdoor="indoor", location=Location(lat=30.6750, lng=104.0667), ticket_price=0),
            ScoredPOI(name="龙抄手", category="restaurant", score=0.75, description="成都传统抄手", tags=["抄手", "小吃", "老字号"], best_time="午餐", recommended_hours="1小时", area="锦江区", indoor_outdoor="indoor", location=Location(lat=30.6569, lng=104.0814), ticket_price=0),
        ],
        "杭州": [
            ScoredPOI(name="西湖", category="attraction", score=0.95, description="世界文化遗产，杭州名片", tags=["湖景", "文化", "拍照"], best_time="全天", recommended_hours="半天", area="西湖区", indoor_outdoor="outdoor", location=Location(lat=30.2456, lng=120.1407), ticket_price=0),
            ScoredPOI(name="灵隐寺", category="attraction", score=0.85, description="千年古刹，佛教圣地", tags=["佛教", "历史", "文化"], best_time="上午", recommended_hours="2-3小时", area="西湖区", indoor_outdoor="mixed", location=Location(lat=30.2408, lng=120.0983), ticket_price=75),
            ScoredPOI(name="宋城", category="attraction", score=0.8, description="大型宋文化主题公园", tags=["演出", "亲子", "文化"], best_time="下午", recommended_hours="半天", area="西湖区", indoor_outdoor="mixed", location=Location(lat=30.1886, lng=120.1124), ticket_price=320),
            ScoredPOI(name="西溪湿地", category="attraction", score=0.8, description="城市湿地，非诚勿扰取景地", tags=["自然", "湿地", "休闲"], best_time="上午", recommended_hours="3-4小时", area="西湖区", indoor_outdoor="outdoor", location=Location(lat=30.2715, lng=120.0632), ticket_price=80),
            ScoredPOI(name="河坊街", category="attraction", score=0.75, description="杭州老街，传统手工艺品", tags=["古街", "购物", "美食"], best_time="下午", recommended_hours="2小时", area="上城区", indoor_outdoor="outdoor", location=Location(lat=30.2430, lng=120.1686), ticket_price=0),
            ScoredPOI(name="千岛湖", category="attraction", score=0.85, description="天下第一秀水", tags=["湖景", "自然", "休闲"], best_time="全天", recommended_hours="全天", area="淳安县", indoor_outdoor="outdoor", location=Location(lat=29.6041, lng=118.9907), ticket_price=150),
            ScoredPOI(name="龙井村", category="attraction", score=0.8, description="龙井茶原产地，茶文化体验", tags=["茶文化", "自然", "乡村"], best_time="上午", recommended_hours="2小时", area="西湖区", indoor_outdoor="outdoor", location=Location(lat=30.2214, lng=120.1244), ticket_price=0),
            ScoredPOI(name="断桥残雪", category="attraction", score=0.85, description="西湖十景之一，白娘子传说", tags=["传说", "湖景", "拍照"], best_time="傍晚", recommended_hours="1小时", area="西湖区", indoor_outdoor="outdoor", location=Location(lat=30.2596, lng=120.1478), ticket_price=0),
            ScoredPOI(name="雷峰塔", category="attraction", score=0.8, description="西湖标志性建筑", tags=["地标", "历史", "湖景"], best_time="下午", recommended_hours="1-2小时", area="西湖区", indoor_outdoor="mixed", location=Location(lat=30.2314, lng=120.1493), ticket_price=40),
            ScoredPOI(name="楼外楼", category="restaurant", score=0.75, description="西湖醋鱼发源地，百年老店", tags=["杭帮菜", "老字号", "湖景"], best_time="午餐", recommended_hours="1.5小时", area="西湖区", indoor_outdoor="indoor", location=Location(lat=30.2586, lng=120.1432), ticket_price=0),
            ScoredPOI(name="知味小笼", category="restaurant", score=0.75, description="杭州小笼包名店", tags=["小笼包", "小吃", "早餐"], best_time="早餐", recommended_hours="1小时", area="上城区", indoor_outdoor="indoor", location=Location(lat=30.2580, lng=120.1650), ticket_price=0),
            ScoredPOI(name="外婆家", category="restaurant", score=0.75, description="杭帮菜连锁名店", tags=["杭帮菜", "人气", "晚餐"], best_time="晚餐", recommended_hours="1.5小时", area="下城区", indoor_outdoor="indoor", location=Location(lat=30.2700, lng=120.1600), ticket_price=0),
        ],
        "西安": [
            ScoredPOI(name="兵马俑", category="attraction", score=0.95, description="世界第八大奇迹", tags=["历史", "考古", "世界遗产"], best_time="上午", recommended_hours="半天", area="临潼区", indoor_outdoor="mixed", location=Location(lat=34.3841, lng=109.2785), ticket_price=120),
            ScoredPOI(name="大雁塔", category="attraction", score=0.9, description="唐代佛教建筑，西安地标", tags=["佛教", "历史", "地标"], best_time="下午", recommended_hours="2小时", area="雁塔区", indoor_outdoor="mixed", location=Location(lat=34.2204, lng=108.9685), ticket_price=50),
            ScoredPOI(name="古城墙", category="attraction", score=0.9, description="中国现存最完整古城墙", tags=["历史", "骑行", "观景"], best_time="傍晚", recommended_hours="2-3小时", area="碑林区", indoor_outdoor="outdoor", location=Location(lat=34.2566, lng=108.9607), ticket_price=54),
            ScoredPOI(name="回民街", category="attraction", score=0.8, description="西安美食文化街区", tags=["美食", "文化", "夜市"], best_time="晚上", recommended_hours="2小时", area="莲湖区", indoor_outdoor="outdoor", location=Location(lat=34.2624, lng=108.9467), ticket_price=0),
            ScoredPOI(name="华清宫", category="attraction", score=0.85, description="唐玄宗与杨贵妃爱情故事", tags=["历史", "温泉", "演出"], best_time="下午", recommended_hours="3小时", area="临潼区", indoor_outdoor="mixed", location=Location(lat=34.3633, lng=109.2124), ticket_price=120),
            ScoredPOI(name="大唐芙蓉园", category="attraction", score=0.85, description="盛唐皇家园林", tags=["园林", "夜景", "文化"], best_time="晚上", recommended_hours="3小时", area="雁塔区", indoor_outdoor="mixed", location=Location(lat=34.2187, lng=108.9695), ticket_price=120),
            ScoredPOI(name="陕西历史博物馆", category="attraction", score=0.9, description="华夏珍宝库", tags=["博物馆", "历史", "文物"], best_time="上午", recommended_hours="3小时", area="雁塔区", indoor_outdoor="indoor", location=Location(lat=34.2264, lng=108.9604), ticket_price=0),
            ScoredPOI(name="钟楼", category="attraction", score=0.85, description="西安市中心地标", tags=["地标", "历史", "建筑"], best_time="晚上", recommended_hours="1小时", area="碑林区", indoor_outdoor="mixed", location=Location(lat=34.2611, lng=108.9425), ticket_price=30),
            ScoredPOI(name="鼓楼", category="attraction", score=0.8, description="中国古代最大鼓楼", tags=["地标", "历史", "建筑"], best_time="晚上", recommended_hours="1小时", area="莲湖区", indoor_outdoor="mixed", location=Location(lat=34.2617, lng=108.9391), ticket_price=30),
            ScoredPOI(name="肉夹馍", category="restaurant", score=0.8, description="西安代表性小吃", tags=["小吃", "必吃", "传统"], best_time="午餐", recommended_hours="1小时", area="莲湖区", indoor_outdoor="indoor", location=Location(lat=34.2600, lng=108.9400), ticket_price=0),
            ScoredPOI(name="羊肉泡馍", category="restaurant", score=0.8, description="陕西传统名吃", tags=["传统", "必吃", "午餐"], best_time="午餐", recommended_hours="1小时", area="莲湖区", indoor_outdoor="indoor", location=Location(lat=34.2620, lng=108.9450), ticket_price=0),
            ScoredPOI(name="凉皮", category="restaurant", score=0.75, description="西安夏日解暑小吃", tags=["小吃", "夏日", "传统"], best_time="午餐", recommended_hours="1小时", area="莲湖区", indoor_outdoor="indoor", location=Location(lat=34.2610, lng=108.9440), ticket_price=0),
        ],
        "重庆": [
            ScoredPOI(name="洪崖洞", category="attraction", score=0.9, description="现实版千与千寻", tags=["夜景", "拍照", "地标"], best_time="晚上", recommended_hours="2小时", area="渝中区", indoor_outdoor="mixed", location=Location(lat=29.5632, lng=106.5803), ticket_price=0),
            ScoredPOI(name="解放碑", category="attraction", score=0.85, description="重庆地标商业步行街", tags=["地标", "购物", "美食"], best_time="下午", recommended_hours="2小时", area="渝中区", indoor_outdoor="outdoor", location=Location(lat=29.5630, lng=106.5516), ticket_price=0),
            ScoredPOI(name="磁器口", category="attraction", score=0.85, description="千年古镇，巴渝文化", tags=["古镇", "文化", "美食"], best_time="上午", recommended_hours="3小时", area="沙坪坝区", indoor_outdoor="outdoor", location=Location(lat=29.5823, lng=106.4485), ticket_price=0),
            ScoredPOI(name="长江索道", category="attraction", score=0.8, description="空中看重庆", tags=["观景", "交通", "体验"], best_time="傍晚", recommended_hours="1小时", area="渝中区", indoor_outdoor="outdoor", location=Location(lat=29.5565, lng=106.5869), ticket_price=30),
            ScoredPOI(name="武隆天坑", category="attraction", score=0.9, description="世界自然遗产，喀斯特地貌", tags=["自然", "地质", "世界遗产"], best_time="全天", recommended_hours="全天", area="武隆区", indoor_outdoor="outdoor", location=Location(lat=29.4276, lng=107.7188), ticket_price=125),
            ScoredPOI(name="朝天门", category="attraction", score=0.8, description="两江交汇处", tags=["江景", "地标", "码头"], best_time="晚上", recommended_hours="1-2小时", area="渝中区", indoor_outdoor="outdoor", location=Location(lat=29.5685, lng=106.5855), ticket_price=0),
            ScoredPOI(name="鹅岭二厂", category="attraction", score=0.75, description="文创园区，文艺青年聚集地", tags=["文创", "拍照", "艺术"], best_time="下午", recommended_hours="2小时", area="渝中区", indoor_outdoor="mixed", location=Location(lat=29.5538, lng=106.5406), ticket_price=0),
            ScoredPOI(name="南山一棵树", category="attraction", score=0.8, description="俯瞰重庆夜景最佳地点", tags=["夜景", "观景", "拍照"], best_time="晚上", recommended_hours="1-2小时", area="南岸区", indoor_outdoor="outdoor", location=Location(lat=29.5562, lng=106.6054), ticket_price=30),
            ScoredPOI(name="四川美术学院", category="attraction", score=0.75, description="涂鸦街，艺术氛围浓厚", tags=["艺术", "涂鸦", "拍照"], best_time="下午", recommended_hours="2小时", area="九龙坡区", indoor_outdoor="outdoor", location=Location(lat=29.6036, lng=106.3014), ticket_price=0),
            ScoredPOI(name="重庆火锅", category="restaurant", score=0.9, description="麻辣鲜香，重庆名片", tags=["火锅", "麻辣", "必吃"], best_time="晚餐", recommended_hours="2小时", area="渝中区", indoor_outdoor="indoor", location=Location(lat=29.5630, lng=106.5516), ticket_price=0),
            ScoredPOI(name="小面", category="restaurant", score=0.8, description="重庆人的早餐灵魂", tags=["面食", "早餐", "人气"], best_time="早餐", recommended_hours="1小时", area="渝中区", indoor_outdoor="indoor", location=Location(lat=29.5630, lng=106.5516), ticket_price=0),
            ScoredPOI(name="酸辣粉", category="restaurant", score=0.75, description="重庆街头经典小吃", tags=["小吃", "酸辣", "传统"], best_time="午餐", recommended_hours="1小时", area="渝中区", indoor_outdoor="indoor", location=Location(lat=29.5630, lng=106.5516), ticket_price=0),
        ],
        "深圳": [
            ScoredPOI(name="世界之窗", category="attraction", score=0.85, description="微缩景观主题公园", tags=["亲子", "地标", "娱乐"], best_time="全天", recommended_hours="半天", area="南山区", indoor_outdoor="mixed", location=Location(lat=22.5358, lng=113.9804), ticket_price=220),
            ScoredPOI(name="欢乐谷", category="attraction", score=0.85, description="大型主题乐园", tags=["亲子", "刺激", "娱乐"], best_time="全天", recommended_hours="全天", area="南山区", indoor_outdoor="mixed", location=Location(lat=22.5404, lng=113.9836), ticket_price=230),
            ScoredPOI(name="大梅沙", category="attraction", score=0.8, description="深圳最美海滩", tags=["海滩", "度假", "亲子"], best_time="下午", recommended_hours="半天", area="盐田区", indoor_outdoor="outdoor", location=Location(lat=22.5947, lng=114.3061), ticket_price=0),
            ScoredPOI(name="华侨城", category="attraction", score=0.8, description="文化旅游度假区", tags=["文化", "休闲", "艺术"], best_time="下午", recommended_hours="3小时", area="南山区", indoor_outdoor="mixed", location=Location(lat=22.5388, lng=113.9817), ticket_price=0),
            ScoredPOI(name="深圳湾公园", category="attraction", score=0.8, description="观鸟看海，城市绿肺", tags=["自然", "休闲", "海景"], best_time="傍晚", recommended_hours="2小时", area="南山区", indoor_outdoor="outdoor", location=Location(lat=22.5180, lng=113.9441), ticket_price=0),
            ScoredPOI(name="莲花山公园", category="attraction", score=0.75, description="市民休闲首选", tags=["公园", "登山", "休闲"], best_time="上午", recommended_hours="2小时", area="福田区", indoor_outdoor="outdoor", location=Location(lat=22.5538, lng=114.0511), ticket_price=0),
            ScoredPOI(name="东门老街", category="attraction", score=0.75, description="深圳最老牌商业街", tags=["购物", "美食", "老街"], best_time="下午", recommended_hours="2小时", area="罗湖区", indoor_outdoor="outdoor", location=Location(lat=22.5456, lng=114.0592), ticket_price=0),
            ScoredPOI(name="海上世界", category="attraction", score=0.8, description="滨海休闲综合体", tags=["海景", "餐饮", "夜景"], best_time="晚上", recommended_hours="2小时", area="南山区", indoor_outdoor="mixed", location=Location(lat=22.4856, lng=113.9136), ticket_price=0),
            ScoredPOI(name="大鹏所城", category="attraction", score=0.8, description="明清海防要塞", tags=["历史", "古迹", "文化"], best_time="上午", recommended_hours="2小时", area="大鹏新区", indoor_outdoor="outdoor", location=Location(lat=22.6000, lng=114.5500), ticket_price=0),
            ScoredPOI(name="潮汕牛肉火锅", category="restaurant", score=0.8, description="深圳热门美食", tags=["火锅", "牛肉", "人气"], best_time="晚餐", recommended_hours="2小时", area="福田区", indoor_outdoor="indoor", location=Location(lat=22.5431, lng=114.0579), ticket_price=0),
            ScoredPOI(name="椰子鸡", category="restaurant", score=0.8, description="深圳原创特色美食", tags=["特色", "清淡", "人气"], best_time="晚餐", recommended_hours="1.5小时", area="福田区", indoor_outdoor="indoor", location=Location(lat=22.5431, lng=114.0579), ticket_price=0),
            ScoredPOI(name="肠粉", category="restaurant", score=0.75, description="广式早餐经典", tags=["早餐", "广式", "传统"], best_time="早餐", recommended_hours="1小时", area="罗湖区", indoor_outdoor="indoor", location=Location(lat=22.5431, lng=114.0579), ticket_price=0),
        ],
        "南京": [
            ScoredPOI(name="中山陵", category="attraction", score=0.9, description="孙中山先生陵寝", tags=["历史", "纪念", "园林"], best_time="上午", recommended_hours="3小时", area="玄武区", indoor_outdoor="outdoor", location=Location(lat=32.0558, lng=118.8493), ticket_price=0),
            ScoredPOI(name="夫子庙", category="attraction", score=0.85, description="秦淮河风光带核心", tags=["古街", "文化", "夜景"], best_time="晚上", recommended_hours="2小时", area="秦淮区", indoor_outdoor="mixed", location=Location(lat=32.0227, lng=118.7943), ticket_price=0),
            ScoredPOI(name="秦淮河", category="attraction", score=0.85, description="十里秦淮，金陵风月", tags=["夜景", "游船", "文化"], best_time="晚上", recommended_hours="2小时", area="秦淮区", indoor_outdoor="outdoor", location=Location(lat=32.0220, lng=118.7940), ticket_price=0),
            ScoredPOI(name="明孝陵", category="attraction", score=0.85, description="明朝开国皇帝陵寝", tags=["历史", "世界遗产", "神道"], best_time="上午", recommended_hours="3小时", area="玄武区", indoor_outdoor="outdoor", location=Location(lat=32.0572, lng=118.8466), ticket_price=70),
            ScoredPOI(name="总统府", category="attraction", score=0.85, description="近代历史遗址", tags=["历史", "建筑", "民国"], best_time="下午", recommended_hours="2-3小时", area="玄武区", indoor_outdoor="mixed", location=Location(lat=32.0438, lng=118.7981), ticket_price=35),
            ScoredPOI(name="鸡鸣寺", category="attraction", score=0.8, description="南京最古老的佛寺", tags=["佛教", "樱花", "历史"], best_time="上午", recommended_hours="1-2小时", area="玄武区", indoor_outdoor="mixed", location=Location(lat=32.0576, lng=118.7444), ticket_price=10),
            ScoredPOI(name="玄武湖", category="attraction", score=0.8, description="江南三大名湖之一", tags=["湖景", "休闲", "皇家园林"], best_time="下午", recommended_hours="2小时", area="玄武区", indoor_outdoor="outdoor", location=Location(lat=32.0605, lng=118.7977), ticket_price=0),
            ScoredPOI(name="南京博物院", category="attraction", score=0.9, description="中国三大博物馆之一", tags=["博物馆", "历史", "文物"], best_time="上午", recommended_hours="3小时", area="玄武区", indoor_outdoor="indoor", location=Location(lat=32.0426, lng=118.8209), ticket_price=0),
            ScoredPOI(name="老门东", category="attraction", score=0.8, description="南京传统民居聚集地", tags=["古街", "美食", "文化"], best_time="下午", recommended_hours="2小时", area="秦淮区", indoor_outdoor="outdoor", location=Location(lat=32.0156, lng=118.7923), ticket_price=0),
            ScoredPOI(name="鸭血粉丝汤", category="restaurant", score=0.85, description="南京代表性小吃", tags=["小吃", "必吃", "传统"], best_time="午餐", recommended_hours="1小时", area="秦淮区", indoor_outdoor="indoor", location=Location(lat=32.0600, lng=118.7800), ticket_price=0),
            ScoredPOI(name="盐水鸭", category="restaurant", score=0.8, description="南京特产，金陵名菜", tags=["特产", "传统", "必吃"], best_time="午餐", recommended_hours="1小时", area="秦淮区", indoor_outdoor="indoor", location=Location(lat=32.0600, lng=118.7800), ticket_price=0),
            ScoredPOI(name="小笼包", category="restaurant", score=0.75, description="江南经典点心", tags=["点心", "早餐", "传统"], best_time="早餐", recommended_hours="1小时", area="秦淮区", indoor_outdoor="indoor", location=Location(lat=32.0600, lng=118.7800), ticket_price=0),
        ],
        "厦门": [
            ScoredPOI(name="鼓浪屿", category="attraction", score=0.9, description="海上花园，万国建筑", tags=["海岛", "建筑", "文艺"], best_time="全天", recommended_hours="全天", area="思明区", indoor_outdoor="mixed", location=Location(lat=24.4482, lng=118.0823), ticket_price=35),
            ScoredPOI(name="南普陀寺", category="attraction", score=0.85, description="闽南佛教圣地", tags=["佛教", "素食", "海景"], best_time="上午", recommended_hours="2小时", area="思明区", indoor_outdoor="mixed", location=Location(lat=24.4396, lng=118.0982), ticket_price=0),
            ScoredPOI(name="厦门大学", category="attraction", score=0.85, description="中国最美大学", tags=["校园", "建筑", "海景"], best_time="下午", recommended_hours="2小时", area="思明区", indoor_outdoor="outdoor", location=Location(lat=24.4393, lng=118.0973), ticket_price=0),
            ScoredPOI(name="环岛路", category="attraction", score=0.8, description="黄金海岸线", tags=["海景", "骑行", "休闲"], best_time="傍晚", recommended_hours="2小时", area="思明区", indoor_outdoor="outdoor", location=Location(lat=24.4625, lng=118.1331), ticket_price=0),
            ScoredPOI(name="曾厝垵", category="attraction", score=0.8, description="文艺渔村", tags=["文创", "美食", "民宿"], best_time="下午", recommended_hours="2小时", area="思明区", indoor_outdoor="mixed", location=Location(lat=24.4317, lng=118.1271), ticket_price=0),
            ScoredPOI(name="中山路步行街", category="attraction", score=0.8, description="厦门最老牌商业街", tags=["购物", "美食", "骑楼"], best_time="晚上", recommended_hours="2小时", area="思明区", indoor_outdoor="outdoor", location=Location(lat=24.4624, lng=118.0834), ticket_price=0),
            ScoredPOI(name="胡里山炮台", category="attraction", score=0.75, description="海防历史遗址", tags=["历史", "海景", "古迹"], best_time="上午", recommended_hours="1-2小时", area="思明区", indoor_outdoor="outdoor", location=Location(lat=24.4391, lng=118.1028), ticket_price=25),
            ScoredPOI(name="园林植物园", category="attraction", score=0.8, description="网红雨林喷雾", tags=["自然", "拍照", "植物"], best_time="上午", recommended_hours="3小时", area="思明区", indoor_outdoor="outdoor", location=Location(lat=24.4450, lng=118.1110), ticket_price=30),
            ScoredPOI(name="沙坡尾", category="attraction", score=0.8, description="厦门港发源地，艺术西区", tags=["艺术", "文创", "老港口"], best_time="下午", recommended_hours="2小时", area="思明区", indoor_outdoor="mixed", location=Location(lat=24.4400, lng=118.1050), ticket_price=0),
            ScoredPOI(name="沙茶面", category="restaurant", score=0.85, description="厦门代表性面食", tags=["面食", "特色小吃", "必吃"], best_time="午餐", recommended_hours="1小时", area="思明区", indoor_outdoor="indoor", location=Location(lat=24.4600, lng=118.0800), ticket_price=0),
            ScoredPOI(name="海蛎煎", category="restaurant", score=0.8, description="闽南传统小吃", tags=["小吃", "海鲜", "传统"], best_time="午餐", recommended_hours="1小时", area="思明区", indoor_outdoor="indoor", location=Location(lat=24.4600, lng=118.0800), ticket_price=0),
            ScoredPOI(name="花生汤", category="restaurant", score=0.75, description="厦门传统甜汤", tags=["甜品", "传统", "早餐"], best_time="早餐", recommended_hours="1小时", area="思明区", indoor_outdoor="indoor", location=Location(lat=24.4600, lng=118.0800), ticket_price=0),
        ],
        "青岛": [
            ScoredPOI(name="栈桥", category="attraction", score=0.85, description="青岛地标，百年老桥", tags=["地标", "海景", "历史"], best_time="上午", recommended_hours="1小时", area="市南区", indoor_outdoor="outdoor", location=Location(lat=36.0596, lng=120.3177), ticket_price=0),
            ScoredPOI(name="八大关", category="attraction", score=0.85, description="万国建筑博览会", tags=["建筑", "历史", "拍照"], best_time="下午", recommended_hours="2小时", area="市南区", indoor_outdoor="outdoor", location=Location(lat=36.0538, lng=120.3508), ticket_price=0),
            ScoredPOI(name="崂山", category="attraction", score=0.9, description="海上第一名山", tags=["山海", "道教", "自然"], best_time="全天", recommended_hours="全天", area="崂山区", indoor_outdoor="outdoor", location=Location(lat=36.1968, lng=120.6237), ticket_price=130),
            ScoredPOI(name="五四广场", category="attraction", score=0.8, description="青岛市中心地标", tags=["地标", "夜景", "广场"], best_time="晚上", recommended_hours="1-2小时", area="市南区", indoor_outdoor="outdoor", location=Location(lat=36.0600, lng=120.3850), ticket_price=0),
            ScoredPOI(name="青岛啤酒博物馆", category="attraction", score=0.85, description="百年啤酒文化", tags=["博物馆", "工业", "体验"], best_time="下午", recommended_hours="2小时", area="市北区", indoor_outdoor="mixed", location=Location(lat=36.0894, lng=120.4110), ticket_price=60),
            ScoredPOI(name="小鱼山", category="attraction", score=0.8, description="俯瞰青岛全景", tags=["观景", "园林", "拍照"], best_time="下午", recommended_hours="1小时", area="市南区", indoor_outdoor="outdoor", location=Location(lat=36.0580, lng=120.3320), ticket_price=10),
            ScoredPOI(name="信号山公园", category="attraction", score=0.8, description="红瓦绿树最佳观景点", tags=["观景", "园林", "旋转楼"], best_time="下午", recommended_hours="1-2小时", area="市南区", indoor_outdoor="outdoor", location=Location(lat=36.0700, lng=120.3400), ticket_price=15),
            ScoredPOI(name="金沙滩", category="attraction", score=0.8, description="亚洲最佳沙滩之一", tags=["海滩", "度假", "亲子"], best_time="下午", recommended_hours="半天", area="黄岛区", indoor_outdoor="outdoor", location=Location(lat=36.0100, lng=120.2700), ticket_price=0),
            ScoredPOI(name="劈柴院", category="attraction", score=0.75, description="青岛老街，特色小吃", tags=["古街", "小吃", "传统"], best_time="下午", recommended_hours="1-2小时", area="市南区", indoor_outdoor="outdoor", location=Location(lat=36.0660, lng=120.3180), ticket_price=0),
            ScoredPOI(name="青岛啤酒", category="restaurant", score=0.8, description="青岛啤酒文化体验", tags=["啤酒", "特色", "体验"], best_time="晚餐", recommended_hours="2小时", area="市北区", indoor_outdoor="indoor", location=Location(lat=36.0894, lng=120.4110), ticket_price=0),
            ScoredPOI(name="海鲜大咖", category="restaurant", score=0.8, description="青岛海鲜盛宴", tags=["海鲜", "聚餐", "人气"], best_time="晚餐", recommended_hours="2小时", area="市南区", indoor_outdoor="indoor", location=Location(lat=36.0600, lng=120.3800), ticket_price=0),
            ScoredPOI(name="鲅鱼饺子", category="restaurant", score=0.8, description="青岛特色水饺", tags=["饺子", "海鲜", "传统"], best_time="午餐", recommended_hours="1小时", area="市南区", indoor_outdoor="indoor", location=Location(lat=36.0600, lng=120.3800), ticket_price=0),
        ],
    }

    def __init__(self):
        self.route_skill = RouteCalculationSkill()

    async def plan(
        self,
        pois: list[ScoredPOI],
        weather: list[WeatherDay],
        profile: UserProfile,
        travel_context: Optional[dict] = None,
    ) -> list[DayPlan]:
        """Plan itinerary using LLM-driven planning from raw POI data."""
        if not profile.travel_days:
            return []

        # Use fallback POIs if search returned empty
        if not pois:
            city = profile.destination or ""
            pois = list(self.CITY_DEFAULTS.get(city, self.CITY_DEFAULTS.get("上海", [])))

        # Primary path: LLM direct planning
        try:
            return await self._plan_with_llm(pois, weather, profile, travel_context)
        except Exception:
            # Fallback to algorithm if LLM fails
            return await self._plan_with_algorithm(pois, weather, profile)

    async def _plan_with_llm(
        self,
        pois: list[ScoredPOI],
        weather: list[WeatherDay],
        profile: UserProfile,
        travel_context: Optional[dict] = None,
    ) -> list[DayPlan]:
        """Use LLM to plan itinerary directly from raw POI data."""

        # Build rich POI context for LLM
        poi_context = self._build_poi_context(pois, profile)
        weather_context = self._build_weather_context(weather)
        profile_context = self._build_profile_context(profile)
        travel_context_text = self._build_travel_context_section(travel_context)

        system_prompt = """你是一位中国资深旅行规划专家 + 独行/小众旅行顾问，拥有15年以上全国带团和自由行规划经验。

你的核心使命：为用户设计高性价比、舒适、不累、体验完整的国内旅行行程，让用户真正"玩得值"而不是"赶路打卡"。

【必须严格遵守的铁律】
1. 时间分配必须合理：高价值/高票价景点（如迪士尼、故宫、长隆、景区核心区等）必须给足够时间（至少6-8小时或全天），绝不允许只安排2-3小时。
2. 路线必须高效：尽量区域集中、不走回头路、减少无效跨区移动。以交通枢纽为中心规划。
3. 必须考虑季节与天气：主动利用当月/当季限定活动（花海、音乐节、美食季、避暑/避寒等），并给出天气应对方案。
4. 必须平衡：经典打卡 + 小众/本地生活 + 休息/缓冲时间。每天步行/移动量控制合理。
5. 必须实用：包含交通方式、预约提醒、避坑建议、美食推荐、灵活调整方案。
6. 优先推荐公共交通（高铁、地铁、公交），打车作为补充。考虑行李、独行安全等因素。
7. 绝不擅自删除用户想去的核心/高票价景点：用户明确提到的景点（如"想去迪士尼""一定要看故宫"）必须在行程中出现且给足时间。如确实无法安排，须在recommendation_reason中说明理由并提供替代方案，绝不可默默删除或弱化。
8. 必须保留目的地Top级经典文化地标：博物馆、标志性建筑、世界遗产、代表性古迹等文化地标的价值高于网红打卡点，不可为追求"小众"而遗漏。这类景点即使不在用户兴趣列表中，也应优先安排至少1-2个。
9. 必须将季节限定活动和近期节日作为行程亮点融入：【当地实用信息】中的upcoming_events必须在行程中体现，有活动时优先安排到对应日期，并在recommendation_reason中明确提及活动名称和看点，绝不可忽略。"""

        user_prompt = f"""请严格按照以下思考步骤设计行程，然后输出JSON格式的完整方案。

【思考步骤】
Step 1: 分析用户需求（天数、偏好、节奏：慢游/深度/打卡等）
Step 2: 了解目的地当前季节特点和限定活动
Step 3: 设计整体路线框架（枢纽 + 区域集中 + 不回头路）
Step 4: 检查致命错误（时间分配不足、路线混乱、忽略季节、过于赶路等）
Step 5: 优化每天节奏和体验亮点
Step 6: 输出最终高质量方案

【用户画像】
{profile_context}

【候选景点/餐厅列表】（共{len(pois)}个，按相关度排序）
{poi_context}

【天气预报】
{weather_context}

{travel_context_text}

【规划要求】
1. 共规划{profile.travel_days}天行程，每天安排3-5个活动（景点+餐厅）
2. 每个景点的duration_min必须根据recommended_hours设置，不要统一用120分钟：
   - "1小时" = 60分钟
   - "1-2小时" = 90分钟
   - "2小时" = 120分钟
   - "2-3小时" = 150分钟
   - "半天" = 240分钟
   - "全天" = 360分钟
3. 餐厅(category=restaurant)duration_min固定为90分钟
4. 每天09:00开始，21:00前结束
5. 考虑地理位置(area字段)：同一区域的景点尽量安排在同一天或相邻时段
6. 考虑最佳时间(best_time)：
   - "上午"的安排在09:00-12:00
   - "下午"的安排在13:00-17:00
   - "晚上"的安排在18:00后
   - "傍晚"的安排在17:00-19:00
7. 考虑天气：雨天优先indoor景点，炎热天气避免正午outdoor活动
8. 每天设定一个theme（主题），如"历史文化之旅"、"美食探索日"
9. 景点之间插入合理的步行/交通时间（约30分钟），体现在start_time/end_time中
10. 午餐安排在11:30-13:30之间，晚餐在17:30-19:30之间
11. 每个activity的recommendation_reason写2-3句推荐理由，必须实用、有画面感
12. 为每一天的day.date填入对应的日期（从travel_dates解析）
13. 必须结合【当地实用信息】中的季节限定和近期活动安排行程，有活动时优先安排
14. 在recommendation_reason中融入【当地实用信息】的美食推荐和避坑提醒
15. 高价值景点（迪士尼/故宫/长隆等）必须安排全天或至少6小时，不可敷衍
16. 尽量区域集中，避免同一天内在城市两端来回奔波
17. 用户兴趣/需求匹配【匹配兴趣】【匹配需求】标记的景点必须优先安排，不可遗漏
18. 【文化地标】标记的景点（博物馆、世界遗产、标志性建筑等）必须至少安排1-2个，不可全部为网红打卡点

【输出格式】
只输出JSON对象，不要输出任何解释、分析、思考过程、Markdown代码块或分隔符。JSON格式如下：
{{
  "days": [
    {{
      "day_number": 1,
      "date": "2026-05-01",
      "theme": "...",
      "activities": [
        {{
          "poi_name": "景点名称",
          "category": "attraction/restaurant",
          "start_time": "09:00",
          "end_time": "11:00",
          "duration_min": 120,
          "ticket_price": 50,
          "meal_cost": 0,
          "recommendation_reason": "推荐理由",
          "time_constraint": "flexible",
          "tags": ["标签1", "标签2"]
        }}
      ]
    }}
  ]
}}"""

        itinerary_data = await llm.json_chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=8192,
        )
        if not itinerary_data or "days" not in itinerary_data:
            raise ValueError("LLM response missing 'days' field")

        # Convert to DayPlan objects with location enrichment
        days = itinerary_data["days"]
        schedule = self._build_day_plans(days, pois, profile)
        return schedule

    def _build_poi_context(
        self, pois: list[ScoredPOI], profile: UserProfile
    ) -> str:
        """Serialize POI list into rich text for LLM prompt, with importance markers."""
        interests = set(profile.interests)
        special_reqs = " ".join(profile.special_requests)

        # Landmark detection keywords
        landmark_tags = {"博物馆", "世界遗产", "地标", "历史", "文化", "古迹", "标志性建筑", "全国重点文物保护单位"}
        full_day_keywords = {"迪士尼", "环球", "长隆", "欢乐世界", "主题乐园", "故宫", "兵马俑", "九寨沟", "布达拉宫"}

        lines = []
        for i, poi in enumerate(pois[:35], 1):  # Top 35 POIs
            markers: list[str] = []

            # User interest match
            if interests and set(poi.tags) & interests:
                markers.append("【匹配兴趣】")

            # User special request match
            if special_reqs:
                req_match = any(
                    kw in special_reqs for kw in [poi.name] + poi.tags
                )
                if req_match:
                    markers.append("【匹配需求】")

            # Cultural landmark
            if poi.category == "museum" or (set(poi.tags) & landmark_tags):
                markers.append("【文化地标】")

            # High-value / full-day attraction
            if any(kw in poi.name for kw in full_day_keywords) or poi.recommended_hours == "全天":
                markers.append("【需全天/高价值】")

            marker_str = "".join(markers)
            area_info = f"【区域:{poi.area}】" if poi.area else ""
            hours_info = f"【建议时长:{poi.recommended_hours}】" if poi.recommended_hours else ""
            best_info = f"【最佳时间:{poi.best_time}】" if poi.best_time else ""
            io_info = f"【类型:{poi.indoor_outdoor}】" if poi.indoor_outdoor else ""
            ticket_info = f"【门票:{poi.ticket_price}元】" if poi.ticket_price is not None else ""
            open_info = f"【开放:{poi.open_time}】" if poi.open_time else ""
            tags_str = ",".join(poi.tags) if poi.tags else ""

            line = (
                f"{i}. {marker_str}{poi.name} ({poi.category}) {area_info}{hours_info}{best_info}{io_info}{ticket_info}{open_info}\n"
                f"   简介：{poi.description or '暂无描述'}\n"
                f"   标签：{tags_str}"
            )
            lines.append(line)
        return "\n\n".join(lines)

    def _build_weather_context(self, weather: list[WeatherDay]) -> str:
        if not weather:
            return "暂无天气数据"
        lines = []
        for w in weather:
            line = f"  {w.date}: {w.condition} {w.temp_low}°C-{w.temp_high}°C"
            if w.precipitation_chance is not None:
                line += f" 降水概率{w.precipitation_chance}%"
            if w.recommendation:
                line += f" ({w.recommendation})"
            lines.append(line)
        return "\n".join(lines)

    def _build_profile_context(self, profile: UserProfile) -> str:
        lines = [
            f"目的地：{profile.destination or '未指定'}",
            f"旅行天数：{profile.travel_days}天",
            f"旅行日期：{profile.travel_dates or '未指定'}",
            f"人数：{profile.travelers_count}人",
            f"类型：{profile.travelers_type or '未指定'}",
            f"预算：{profile.budget_range or '未指定'}元",
            f"饮食偏好：{', '.join(profile.food_preferences) if profile.food_preferences else '无'}",
            f"兴趣：{', '.join(profile.interests) if profile.interests else '无'}",
            f"节奏：{profile.pace}",
            f"住宿偏好：{profile.accommodation_preference or '未指定'}",
            f"特殊要求：{', '.join(profile.special_requests) if profile.special_requests else '无'}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _build_travel_context_section(travel_context: Optional[dict]) -> str:
        """Build the travel context section for the planner prompt."""
        if not travel_context:
            return "【当地实用信息】\n暂无"
        tc = TravelContext.from_dict(travel_context)
        return f"【当地实用信息】\n{tc.to_prompt_text()}"

    def _build_day_plans(
        self,
        days_data: list[dict],
        original_pois: list[ScoredPOI],
        profile: UserProfile,
    ) -> list[DayPlan]:
        """Convert LLM output JSON into DayPlan objects, enriching with original POI data."""

        schedule = []
        for day_data in days_data:
            day = DayPlan(
                day_number=day_data.get("day_number", len(schedule) + 1),
                date=day_data.get("date"),
                theme=day_data.get("theme"),
            )

            activities_data = day_data.get("activities", [])
            for act_data in activities_data:
                poi_name = act_data.get("poi_name", "")
                original = self._find_original_poi(poi_name, original_pois)

                activity = Activity(
                    poi_name=poi_name,
                    category=act_data.get("category", "attraction"),
                    start_time=act_data.get("start_time"),
                    end_time=act_data.get("end_time"),
                    duration_min=act_data.get("duration_min", 120),
                    ticket_price=act_data.get("ticket_price"),
                    meal_cost=act_data.get("meal_cost"),
                    recommendation_reason=act_data.get("recommendation_reason", ""),
                    time_constraint=act_data.get("time_constraint", "flexible"),
                    tags=act_data.get("tags", []),
                )

                # Enrich with original POI data
                if original:
                    if original.location:
                        activity.location = original.location
                    if activity.ticket_price is None and original.ticket_price is not None:
                        activity.ticket_price = original.ticket_price
                    if not activity.tags and original.tags:
                        activity.tags = original.tags

                day.activities.append(activity)

            # Calculate day totals
            day.total_cost = sum(
                (a.ticket_price or 0) + (a.meal_cost or 0)
                for a in day.activities
            )
            schedule.append(day)

        return schedule

    @staticmethod
    def _find_original_poi(poi_name: str, original_pois: list[ScoredPOI]) -> Optional[ScoredPOI]:
        """Fuzzy match POI name against original POI list with confidence threshold."""
        if not poi_name:
            return None

        name_lower = poi_name.lower().strip()

        # Exact match first
        for poi in original_pois:
            if poi.name.lower().strip() == name_lower:
                return poi

        # Substring match with length filter
        candidates = []
        for poi in original_pois:
            orig_lower = poi.name.lower().strip()
            if orig_lower in name_lower or name_lower in orig_lower:
                # Require minimum length for substring match
                min_len = min(len(orig_lower), len(name_lower))
                if min_len >= 3:
                    candidates.append(poi)

        if candidates:
            # Return the one with highest similarity ratio
            best = max(
                candidates,
                key=lambda p: SequenceMatcher(None, name_lower, p.name.lower().strip()).ratio(),
            )
            return best

        # Fuzzy match with threshold
        best_match = None
        best_ratio = 0.0
        for poi in original_pois:
            ratio = SequenceMatcher(None, name_lower, poi.name.lower().strip()).ratio()
            if ratio > best_ratio and ratio >= 0.6:
                best_ratio = ratio
                best_match = poi

        return best_match

    # ===== Fallback algorithm (original 6-layer) =====

    async def _plan_with_algorithm(
        self,
        pois: list[ScoredPOI],
        weather: list[WeatherDay],
        profile: UserProfile,
    ) -> list[DayPlan]:
        """Original 6-layer algorithm as fallback."""
        scored = self._score_pois(pois, profile)
        groups = self._group_pois_by_area(scored)
        constrained = self._mark_time_constraints(scored)
        day_assignments = self._assign_days(constrained, groups, profile.travel_days)
        optimized = self._optimize_daily_routes(day_assignments)
        schedule = self._build_schedule(optimized, weather, profile)
        return schedule

    def _score_pois(self, pois: list[ScoredPOI], profile: UserProfile) -> list[ScoredPOI]:
        """Step 1: Score POIs by preference match."""
        interests = set(profile.interests)
        food_prefs = set(profile.food_preferences)

        for poi in pois:
            base = poi.score
            interest_match = len(set(poi.tags) & interests) * 0.2
            food_match = len(set(poi.tags) & food_prefs) * 0.3
            pace_bonus = 0.1 if profile.pace != "intensive" else 0.0
            poi.score = min(base + interest_match + food_match + pace_bonus, 1.0)

        pois.sort(key=lambda p: p.score, reverse=True)
        return pois

    def _group_pois_by_area(
        self, pois: list[ScoredPOI]
    ) -> dict[str, list[ScoredPOI]]:
        """Step 2: Group POIs by area/region for efficient daily planning."""
        groups: dict[str, list[ScoredPOI]] = {}
        for poi in pois:
            area = poi.area or "其他"
            groups.setdefault(area, []).append(poi)
        # Sort groups by size descending so largest areas get dedicated days first
        return dict(sorted(groups.items(), key=lambda x: len(x[1]), reverse=True))

    def _mark_time_constraints(
        self, pois: list[ScoredPOI]
    ) -> list[ScoredPOI]:
        """Step 3: Mark hard time constraints."""
        for poi in pois:
            if poi.category == "attraction" and "夜景" in poi.tags:
                poi.time_constraint = "evening_only"
            elif poi.category == "restaurant" and "早茶" in poi.tags:
                poi.time_constraint = "morning_only"
            else:
                poi.time_constraint = "flexible"
        return pois

    def _assign_days(
        self,
        pois: list[ScoredPOI],
        groups: dict[str, list[ScoredPOI]],
        travel_days: int,
    ) -> list[list[ScoredPOI]]:
        """Step 4: Assign POIs to days by area group."""
        days: list[list[ScoredPOI]] = [[] for _ in range(travel_days)]
        assigned = set()

        sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
        for day_idx, (_, group_pois) in enumerate(sorted_groups[:travel_days]):
            for poi in group_pois:
                if poi.name not in assigned:
                    days[day_idx].append(poi)
                    assigned.add(poi.name)

        remaining = [p for p in pois if p.name not in assigned]
        day_idx = 0
        for poi in remaining:
            while len(days[day_idx]) >= 5:
                day_idx = (day_idx + 1) % travel_days
            days[day_idx].append(poi)
            assigned.add(poi.name)
            day_idx = (day_idx + 1) % travel_days

        return days

    def _optimize_daily_routes(
        self, day_assignments: list[list[ScoredPOI]]
    ) -> list[list[ScoredPOI]]:
        """Step 5: 2-opt route optimization per day."""
        optimized = []
        for day_pois in day_assignments:
            if len(day_pois) <= 2:
                optimized.append(day_pois)
                continue
            ordered = self._nearest_neighbor(day_pois)
            optimized.append(ordered)
        return optimized

    def _nearest_neighbor(self, pois: list[ScoredPOI]) -> list[ScoredPOI]:
        """Greedy nearest neighbor ordering."""
        if not pois:
            return []

        unvisited = set(range(len(pois)))
        route = [0]
        unvisited.remove(0)

        while unvisited:
            last = route[-1]
            last_loc = pois[last].location
            nearest = min(
                unvisited,
                key=lambda i: self._distance(last_loc, pois[i].location),
            )
            route.append(nearest)
            unvisited.remove(nearest)

        return [pois[i] for i in route]

    def _distance(self, a: Optional[Location], b: Optional[Location]) -> float:
        if not a or not b:
            return float("inf")
        return self.route_skill._haversine(a.lat, a.lng, b.lat, b.lng)

    def _build_schedule(
        self,
        day_pois: list[list[ScoredPOI]],
        weather: list[WeatherDay],
        profile: UserProfile,
    ) -> list[DayPlan]:
        """Step 6: Build daily schedule with meal insertion."""
        schedule = []
        start_time = 9 * 60

        for day_idx, pois in enumerate(day_pois):
            day = DayPlan(day_number=day_idx + 1)

            if day_idx < len(weather):
                day.date = weather[day_idx].date

            current_time = start_time
            last_meal_time = -1000

            for poi in pois:
                if current_time >= 11 * 60 + 30 and current_time <= 13 * 60 + 30:
                    if current_time - last_meal_time >= 3.5 * 60:
                        meal = self._create_meal_activity(day_idx, "lunch", profile)
                        day.activities.append(meal)
                        current_time += 90
                        last_meal_time = current_time

                if current_time >= 17 * 60 + 30 and current_time <= 19 * 60 + 30:
                    if current_time - last_meal_time >= 3.5 * 60:
                        meal = self._create_meal_activity(day_idx, "dinner", profile)
                        day.activities.append(meal)
                        current_time += 90
                        last_meal_time = current_time

                duration = 120 if poi.category == "attraction" else 60
                activity = Activity(
                    poi_name=poi.name,
                    poi_id=poi.name,
                    category=poi.category,
                    start_time=self._min_to_time(current_time),
                    end_time=self._min_to_time(current_time + duration),
                    duration_min=duration,
                    location=poi.location,
                    recommendation_reason=poi.description or f"推荐游览{poi.name}",
                    ticket_price=poi.ticket_price,
                    time_constraint=poi.time_constraint,
                    tags=poi.tags,
                )
                day.activities.append(activity)
                current_time += duration
                current_time += 30

            day.total_cost = sum(
                (a.ticket_price or 0) + (a.meal_cost or 0)
                for a in day.activities
            )
            schedule.append(day)

        return schedule

    def _create_meal_activity(
        self, day_idx: int, meal_type: str, profile: UserProfile
    ) -> Activity:
        """Create a meal activity."""
        food_hint = f"（偏好：{','.join(profile.food_preferences)}）" if profile.food_preferences else ""
        return Activity(
            poi_name=f"{meal_type.capitalize()} {food_hint}",
            category="restaurant",
            duration_min=90,
            meal_cost=80,
            recommendation_reason=f"在附近找一家{'辣' if '辣' in profile.food_preferences else '口碑好'}的餐厅",
        )

    @staticmethod
    def _min_to_time(minutes: int) -> str:
        h = minutes // 60
        m = minutes % 60
        return f"{h:02d}:{m:02d}"

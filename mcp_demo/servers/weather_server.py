#!/usr/bin/env python3
"""
天气MCP服务器

这个服务器展示了如何使用MCP暴露天气信息查询功能。
演示了Resources、Tools和Prompts，以及模拟API调用和缓存机制。
"""

import asyncio
import json
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import time

# 复用前面的MCPServer基础类
from filesystem_server import MCPServer

class WeatherServer:
    def __init__(self, api_timeout: int = 30, cache_duration: int = 300, default_units: str = "celsius"):
        """
        初始化天气服务器
        
        Args:
            api_timeout: API超时时间（秒）
            cache_duration: 缓存持续时间（秒）
            default_units: 默认温度单位
        """
        self.server = MCPServer("weather-server", "1.0.0")
        self.api_timeout = api_timeout
        self.cache_duration = cache_duration
        self.default_units = default_units
        
        # 简单的内存缓存
        self.cache = {}
        
        # 模拟城市数据库
        self.cities = {
            "北京": {"lat": 39.9042, "lon": 116.4074, "timezone": "Asia/Shanghai"},
            "上海": {"lat": 31.2304, "lon": 121.4737, "timezone": "Asia/Shanghai"},
            "广州": {"lat": 23.1291, "lon": 113.2644, "timezone": "Asia/Shanghai"},
            "深圳": {"lat": 22.3193, "lon": 114.1694, "timezone": "Asia/Shanghai"},
            "纽约": {"lat": 40.7128, "lon": -74.0060, "timezone": "America/New_York"},
            "伦敦": {"lat": 51.5074, "lon": -0.1278, "timezone": "Europe/London"},
            "东京": {"lat": 35.6762, "lon": 139.6503, "timezone": "Asia/Tokyo"},
            "悉尼": {"lat": -33.8688, "lon": 151.2093, "timezone": "Australia/Sydney"}
        }
        
        self._setup_capabilities()
    
    def _setup_capabilities(self):
        """设置服务器能力"""
        # 添加工具
        self.server.add_tool(
            "get_current_weather",
            "获取指定城市的当前天气信息",
            {
                "city": {"type": "string", "description": "城市名称"},
                "units": {
                    "type": "string", 
                    "enum": ["celsius", "fahrenheit", "kelvin"],
                    "description": "温度单位",
                    "default": self.default_units
                }
            }
        )
        
        self.server.add_tool(
            "get_weather_forecast",
            "获取指定城市的天气预报",
            {
                "city": {"type": "string", "description": "城市名称"},
                "days": {"type": "integer", "description": "预报天数（1-7）", "default": 3},
                "units": {
                    "type": "string", 
                    "enum": ["celsius", "fahrenheit", "kelvin"],
                    "description": "温度单位",
                    "default": self.default_units
                }
            }
        )
        
        self.server.add_tool(
            "search_cities",
            "搜索城市",
            {
                "query": {"type": "string", "description": "城市名称查询"}
            }
        )
        
        self.server.add_tool(
            "get_weather_alerts",
            "获取天气预警信息",
            {
                "city": {"type": "string", "description": "城市名称"}
            }
        )
        
        self.server.add_tool(
            "compare_weather",
            "比较多个城市的天气",
            {
                "cities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "城市列表"
                },
                "units": {
                    "type": "string", 
                    "enum": ["celsius", "fahrenheit", "kelvin"],
                    "description": "温度单位",
                    "default": self.default_units
                }
            }
        )
        
        # 添加提示
        self.server.add_prompt(
            "weather_report",
            "生成天气报告",
            [
                {"name": "city", "description": "城市名称", "required": True},
                {"name": "report_type", "description": "报告类型：current, forecast, detailed", "required": False}
            ]
        )
        
        self.server.add_prompt(
            "travel_weather_advice",
            "旅行天气建议",
            [
                {"name": "cities", "description": "旅行城市列表", "required": True},
                {"name": "travel_dates", "description": "旅行日期", "required": False}
            ]
        )
    
    def _get_cache_key(self, city: str, data_type: str, **kwargs) -> str:
        """生成缓存键"""
        key_parts = [city.lower(), data_type]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}:{v}")
        return "|".join(key_parts)
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """检查缓存是否有效"""
        return time.time() - timestamp < self.cache_duration
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """从缓存获取数据"""
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if self._is_cache_valid(timestamp):
                return data
            else:
                del self.cache[cache_key]
        return None
    
    def _save_to_cache(self, cache_key: str, data: Dict):
        """保存数据到缓存"""
        self.cache[cache_key] = (data, time.time())
    
    def _generate_mock_weather(self, city: str, units: str = "celsius") -> Dict:
        """生成模拟天气数据"""
        # 根据城市生成相对真实的随机天气数据
        city_info = self.cities.get(city, self.cities["北京"])
        
        # 基于纬度生成合理的温度范围
        lat = city_info["lat"]
        if lat > 45:  # 高纬度
            base_temp = random.randint(-10, 15)
        elif lat > 25:  # 中纬度
            base_temp = random.randint(5, 25)
        elif lat > 0:  # 低纬度
            base_temp = random.randint(15, 35)
        else:  # 南半球
            base_temp = random.randint(0, 20)
        
        # 转换温度单位
        if units == "fahrenheit":
            temperature = base_temp * 9/5 + 32
            temp_unit = "°F"
        elif units == "kelvin":
            temperature = base_temp + 273.15
            temp_unit = "K"
        else:
            temperature = base_temp
            temp_unit = "°C"
        
        conditions = ["晴朗", "多云", "阴天", "小雨", "中雨", "大雨", "雷雨", "雪", "雾"]
        condition = random.choice(conditions)
        
        return {
            "city": city,
            "temperature": round(temperature, 1),
            "temperature_unit": temp_unit,
            "condition": condition,
            "humidity": random.randint(30, 90),
            "wind_speed": round(random.uniform(0, 20), 1),
            "wind_direction": random.choice(["北", "东北", "东", "东南", "南", "西南", "西", "西北"]),
            "pressure": random.randint(980, 1030),
            "visibility": round(random.uniform(1, 15), 1),
            "uv_index": random.randint(0, 11),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": city_info["timezone"]
        }
    
    def _generate_mock_forecast(self, city: str, days: int, units: str = "celsius") -> List[Dict]:
        """生成模拟预报数据"""
        forecast = []
        base_weather = self._generate_mock_weather(city, units)
        
        for i in range(days):
            date = datetime.now() + timedelta(days=i)
            day_weather = base_weather.copy()
            
            # 每天的温度有小幅变化
            temp_variation = random.randint(-5, 5)
            day_weather["temperature"] += temp_variation
            day_weather["date"] = date.strftime("%Y-%m-%d")
            day_weather["day"] = date.strftime("%A")
            
            # 添加最高最低温度
            if units == "celsius":
                day_weather["temp_max"] = day_weather["temperature"] + random.randint(2, 8)
                day_weather["temp_min"] = day_weather["temperature"] - random.randint(2, 8)
            elif units == "fahrenheit":
                day_weather["temp_max"] = day_weather["temperature"] + random.randint(4, 15)
                day_weather["temp_min"] = day_weather["temperature"] - random.randint(4, 15)
            else:  # kelvin
                day_weather["temp_max"] = day_weather["temperature"] + random.randint(2, 8)
                day_weather["temp_min"] = day_weather["temperature"] - random.randint(2, 8)
            
            # 随机改变天气条件
            conditions = ["晴朗", "多云", "阴天", "小雨", "中雨"]
            day_weather["condition"] = random.choice(conditions)
            
            forecast.append(day_weather)
        
        return forecast
    
    async def call_tool(self, name: str, arguments: Dict) -> Dict:
        """调用工具"""
        if name == "get_current_weather":
            return await self._get_current_weather(
                arguments["city"], 
                arguments.get("units", self.default_units)
            )
        elif name == "get_weather_forecast":
            return await self._get_weather_forecast(
                arguments["city"],
                arguments.get("days", 3),
                arguments.get("units", self.default_units)
            )
        elif name == "search_cities":
            return await self._search_cities(arguments["query"])
        elif name == "get_weather_alerts":
            return await self._get_weather_alerts(arguments["city"])
        elif name == "compare_weather":
            return await self._compare_weather(
                arguments["cities"],
                arguments.get("units", self.default_units)
            )
        else:
            raise ValueError(f"未知工具: {name}")
    
    async def _get_current_weather(self, city: str, units: str) -> Dict:
        """获取当前天气"""
        try:
            # 检查城市是否存在
            if city not in self.cities:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"错误: 未找到城市 '{city}'。支持的城市: {', '.join(self.cities.keys())}"
                    }]
                }
            
            # 检查缓存
            cache_key = self._get_cache_key(city, "current", units=units)
            cached_data = self._get_from_cache(cache_key)
            
            if cached_data:
                weather_data = cached_data
                source_note = " (缓存数据)"
            else:
                # 模拟API调用延迟
                await asyncio.sleep(0.1)
                weather_data = self._generate_mock_weather(city, units)
                self._save_to_cache(cache_key, weather_data)
                source_note = " (实时数据)"
            
            result_text = f"当前天气{source_note}:\n"
            result_text += f"城市: {weather_data['city']}\n"
            result_text += f"时间: {weather_data['timestamp']}\n"
            result_text += f"温度: {weather_data['temperature']}{weather_data['temperature_unit']}\n"
            result_text += f"天气: {weather_data['condition']}\n"
            result_text += f"湿度: {weather_data['humidity']}%\n"
            result_text += f"风速: {weather_data['wind_speed']} m/s\n"
            result_text += f"风向: {weather_data['wind_direction']}\n"
            result_text += f"气压: {weather_data['pressure']} hPa\n"
            result_text += f"能见度: {weather_data['visibility']} km\n"
            result_text += f"紫外线指数: {weather_data['uv_index']}"
            
            return {
                "content": [{
                    "type": "text",
                    "text": result_text
                }]
            }
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"获取天气信息失败: {str(e)}"
                }]
            }
    
    async def _get_weather_forecast(self, city: str, days: int, units: str) -> Dict:
        """获取天气预报"""
        try:
            if city not in self.cities:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"错误: 未找到城市 '{city}'。支持的城市: {', '.join(self.cities.keys())}"
                    }]
                }
            
            if not 1 <= days <= 7:
                return {
                    "content": [{
                        "type": "text",
                        "text": "错误: 预报天数必须在1-7之间"
                    }]
                }
            
            # 检查缓存
            cache_key = self._get_cache_key(city, "forecast", days=days, units=units)
            cached_data = self._get_from_cache(cache_key)
            
            if cached_data:
                forecast_data = cached_data
                source_note = " (缓存数据)"
            else:
                # 模拟API调用延迟
                await asyncio.sleep(0.2)
                forecast_data = self._generate_mock_forecast(city, days, units)
                self._save_to_cache(cache_key, forecast_data)
                source_note = " (实时数据)"
            
            result_text = f"{city} {days}天天气预报{source_note}:\n\n"
            
            for day_data in forecast_data:
                result_text += f"日期: {day_data['date']} ({day_data['day']})\n"
                result_text += f"温度: {day_data['temp_min']}-{day_data['temp_max']}{day_data['temperature_unit']}\n"
                result_text += f"天气: {day_data['condition']}\n"
                result_text += f"湿度: {day_data['humidity']}%\n"
                result_text += f"风速: {day_data['wind_speed']} m/s\n\n"
            
            return {
                "content": [{
                    "type": "text",
                    "text": result_text
                }]
            }
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"获取天气预报失败: {str(e)}"
                }]
            }
    
    async def _search_cities(self, query: str) -> Dict:
        """搜索城市"""
        query = query.lower()
        found_cities = []
        
        for city, info in self.cities.items():
            if query in city.lower():
                found_cities.append({
                    "name": city,
                    "latitude": info["lat"],
                    "longitude": info["lon"],
                    "timezone": info["timezone"]
                })
        
        if found_cities:
            result_text = f"找到 {len(found_cities)} 个匹配的城市:\n\n"
            for city in found_cities:
                result_text += f"城市: {city['name']}\n"
                result_text += f"经纬度: {city['latitude']}, {city['longitude']}\n"
                result_text += f"时区: {city['timezone']}\n\n"
        else:
            result_text = f"未找到匹配 '{query}' 的城市\n"
            result_text += f"支持的城市: {', '.join(self.cities.keys())}"
        
        return {
            "content": [{
                "type": "text",
                "text": result_text
            }]
        }
    
    async def _get_weather_alerts(self, city: str) -> Dict:
        """获取天气预警"""
        try:
            if city not in self.cities:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"错误: 未找到城市 '{city}'"
                    }]
                }
            
            # 模拟随机预警（实际应该从API获取）
            alerts = []
            alert_types = ["高温", "暴雨", "大风", "雷电", "雾霾", "寒潮"]
            
            # 30%概率有预警
            if random.random() < 0.3:
                alert_type = random.choice(alert_types)
                severity = random.choice(["黄色", "橙色", "红色"])
                alerts.append({
                    "type": alert_type,
                    "severity": severity,
                    "description": f"{alert_type}预警信号",
                    "issue_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "effective_time": (datetime.now() + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M"),
                    "expires_time": (datetime.now() + timedelta(hours=6)).strftime("%Y-%m-%d %H:%M")
                })
            
            if alerts:
                result_text = f"{city} 天气预警信息:\n\n"
                for alert in alerts:
                    result_text += f"预警类型: {alert['type']}预警\n"
                    result_text += f"预警级别: {alert['severity']}\n"
                    result_text += f"发布时间: {alert['issue_time']}\n"
                    result_text += f"生效时间: {alert['effective_time']}\n"
                    result_text += f"过期时间: {alert['expires_time']}\n"
                    result_text += f"描述: {alert['description']}\n\n"
            else:
                result_text = f"{city} 目前没有天气预警信息"
            
            return {
                "content": [{
                    "type": "text",
                    "text": result_text
                }]
            }
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"获取预警信息失败: {str(e)}"
                }]
            }
    
    async def _compare_weather(self, cities: List[str], units: str) -> Dict:
        """比较多个城市的天气"""
        try:
            if len(cities) < 2:
                return {
                    "content": [{
                        "type": "text",
                        "text": "错误: 至少需要比较2个城市"
                    }]
                }
            
            if len(cities) > 5:
                return {
                    "content": [{
                        "type": "text",
                        "text": "错误: 最多只能比较5个城市"
                    }]
                }
            
            weather_data = []
            for city in cities:
                if city not in self.cities:
                    return {
                        "content": [{
                            "type": "text",
                            "text": f"错误: 未找到城市 '{city}'"
                        }]
                    }
                
                # 获取天气数据
                cache_key = self._get_cache_key(city, "current", units=units)
                cached_data = self._get_from_cache(cache_key)
                
                if cached_data:
                    city_weather = cached_data
                else:
                    await asyncio.sleep(0.05)  # 模拟API调用
                    city_weather = self._generate_mock_weather(city, units)
                    self._save_to_cache(cache_key, city_weather)
                
                weather_data.append(city_weather)
            
            # 生成比较报告
            result_text = f"城市天气对比 ({len(cities)}个城市):\n\n"
            
            # 表格头
            result_text += f"{'城市':8} {'温度':8} {'天气':8} {'湿度':6} {'风速':8}\n"
            result_text += "-" * 50 + "\n"
            
            # 数据行
            for data in weather_data:
                result_text += f"{data['city']:8} {data['temperature']:6.1f}{data['temperature_unit']:2} "
                result_text += f"{data['condition']:8} {data['humidity']:3}% {data['wind_speed']:6.1f}m/s\n"
            
            # 统计信息
            temperatures = [data['temperature'] for data in weather_data]
            result_text += f"\n温度统计:\n"
            result_text += f"最高: {max(temperatures):.1f}{weather_data[0]['temperature_unit']}\n"
            result_text += f"最低: {min(temperatures):.1f}{weather_data[0]['temperature_unit']}\n"
            result_text += f"平均: {sum(temperatures)/len(temperatures):.1f}{weather_data[0]['temperature_unit']}\n"
            
            return {
                "content": [{
                    "type": "text",
                    "text": result_text
                }]
            }
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"天气比较失败: {str(e)}"
                }]
            }
    
    async def list_resources(self) -> List[Dict]:
        """列出可用资源"""
        resources = []
        
        # 为每个支持的城市创建资源
        for city in self.cities.keys():
            resources.append({
                "uri": f"weather://{city}/current",
                "name": f"{city}当前天气",
                "description": f"{city}的实时天气信息",
                "mimeType": "application/json"
            })
            
            resources.append({
                "uri": f"weather://{city}/forecast",
                "name": f"{city}天气预报",
                "description": f"{city}的天气预报信息",
                "mimeType": "application/json"
            })
        
        return resources
    
    async def read_resource(self, uri: str) -> Dict:
        """读取资源内容"""
        if not uri.startswith("weather://"):
            raise ValueError("不支持的URI格式")
        
        # 解析URI: weather://城市/类型
        parts = uri[10:].split("/")
        if len(parts) != 2:
            raise ValueError("无效的URI格式")
        
        city, data_type = parts
        
        if city not in self.cities:
            raise ValueError(f"不支持的城市: {city}")
        
        if data_type == "current":
            weather_data = self._generate_mock_weather(city, self.default_units)
            return {
                "contents": [{
                    "uri": uri,
                    "text": json.dumps(weather_data, ensure_ascii=False, indent=2),
                    "mimeType": "application/json"
                }]
            }
        elif data_type == "forecast":
            forecast_data = self._generate_mock_forecast(city, 3, self.default_units)
            return {
                "contents": [{
                    "uri": uri,
                    "text": json.dumps(forecast_data, ensure_ascii=False, indent=2),
                    "mimeType": "application/json"
                }]
            }
        else:
            raise ValueError(f"不支持的数据类型: {data_type}")
    
    async def get_prompt(self, name: str, arguments: Dict) -> Dict:
        """获取提示"""
        if name == "weather_report":
            city = arguments["city"]
            report_type = arguments.get("report_type", "current")
            return {
                "description": f"生成{city}的天气报告",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": f"请为{city}生成一份{report_type}天气报告。请使用天气工具获取最新数据，并提供详细的分析和建议。"
                        }
                    }
                ]
            }
        elif name == "travel_weather_advice":
            cities = arguments["cities"]
            travel_dates = arguments.get("travel_dates", "近期")
            return {
                "description": f"旅行天气建议",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": f"我计划{travel_dates}去这些城市旅行: {', '.join(cities)}。请帮我分析各地的天气情况，并提供旅行建议，包括穿衣、携带物品等。"
                        }
                    }
                ]
            }
        else:
            raise ValueError(f"未知提示: {name}")

# 服务器运行逻辑
async def run_server():
    """运行天气服务器"""
    server = WeatherServer(
        api_timeout=30,
        cache_duration=300,
        default_units="celsius"
    )
    
    print("天气MCP服务器已启动")
    print("支持的功能:")
    print("- Resources: 城市天气资源")
    print("- Tools: get_current_weather, get_weather_forecast, search_cities, get_weather_alerts, compare_weather")
    print("- Prompts: weather_report, travel_weather_advice")
    print(f"- 支持城市: {', '.join(server.cities.keys())}")
    print(f"- 缓存时间: {server.cache_duration} 秒")
    print("\n等待客户端连接...")
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n服务器停止")

if __name__ == "__main__":
    asyncio.run(run_server()) 
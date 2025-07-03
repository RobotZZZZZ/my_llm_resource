"""
MCP Demo服务器包

包含多个示例MCP服务器：
- filesystem_server: 文件系统服务器
- calculator_server: 计算器服务器 
- weather_server: 天气服务器
"""

__version__ = "1.0.0"
__author__ = "MCP Demo"

# 导出主要服务器类
from .filesystem_server import FilesystemServer
from .calculator_server import CalculatorServer
from .weather_server import WeatherServer

__all__ = [
    "FilesystemServer",
    "CalculatorServer", 
    "WeatherServer"
] 
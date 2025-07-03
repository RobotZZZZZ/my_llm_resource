"""
MCP Demo客户端包

包含多个示例MCP客户端：
- simple_client: 简单客户端示例
- chat_client: 聊天客户端示例
"""

__version__ = "1.0.0"
__author__ = "MCP Demo"

# 导出主要客户端类
from .simple_client import SimpleClient
from .chat_client import ChatClient

__all__ = [
    "SimpleClient",
    "ChatClient"
] 
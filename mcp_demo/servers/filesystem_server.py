#!/usr/bin/env python3
"""
文件系统MCP服务器

这个服务器展示了如何使用MCP暴露文件系统资源和文件操作工具。
它演示了MCP的三个核心原语：Resources、Tools和Prompts。
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import mimetypes
import stat
from datetime import datetime

# 简化的MCP实现（在实际项目中应该使用官方MCP SDK）
class MCPServer:
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self.capabilities = {
            "resources": {},
            "tools": {},
            "prompts": {}
        }
        self.resources = {}
        self.tools = {}
        self.prompts = {}
    
    def add_resource(self, uri: str, name: str, description: str = ""):
        """添加资源"""
        self.resources[uri] = {
            "uri": uri,
            "name": name,
            "description": description
        }
    
    def add_tool(self, name: str, description: str, parameters: Dict):
        """添加工具"""
        self.tools[name] = {
            "name": name,
            "description": description,
            "inputSchema": {
                "type": "object",
                "properties": parameters
            }
        }
    
    def add_prompt(self, name: str, description: str, arguments: List[Dict]):
        """添加提示"""
        self.prompts[name] = {
            "name": name,
            "description": description,
            "arguments": arguments
        }

class FilesystemServer:
    def __init__(self, allowed_paths: List[str] = None, max_file_size: int = 10*1024*1024):
        """
        初始化文件系统服务器
        
        Args:
            allowed_paths: 允许访问的路径列表
            max_file_size: 最大文件大小（字节）
        """
        self.server = MCPServer("filesystem-server", "1.0.0")
        self.allowed_paths = [Path(p).resolve() for p in (allowed_paths or ["./test_data"])]
        self.max_file_size = max_file_size
        
        # 确保允许的路径存在
        for path in self.allowed_paths:
            path.mkdir(parents=True, exist_ok=True)
        
        self._setup_capabilities()
    
    def _setup_capabilities(self):
        """设置服务器能力"""
        # 添加工具
        self.server.add_tool(
            "list_files",
            "列出指定目录中的文件和文件夹",
            {
                "path": {"type": "string", "description": "目录路径"}
            }
        )
        
        self.server.add_tool(
            "read_file",
            "读取文件内容",
            {
                "path": {"type": "string", "description": "文件路径"}
            }
        )
        
        self.server.add_tool(
            "write_file",
            "写入文件内容",
            {
                "path": {"type": "string", "description": "文件路径"},
                "content": {"type": "string", "description": "文件内容"}
            }
        )
        
        self.server.add_tool(
            "get_file_info",
            "获取文件信息",
            {
                "path": {"type": "string", "description": "文件路径"}
            }
        )
        
        # 添加提示
        self.server.add_prompt(
            "analyze_file",
            "分析文件内容",
            [
                {"name": "file_path", "description": "要分析的文件路径", "required": True}
            ]
        )
        
        self.server.add_prompt(
            "summarize_directory",
            "总结目录内容",
            [
                {"name": "directory_path", "description": "要总结的目录路径", "required": True}
            ]
        )
    
    def _is_path_allowed(self, path: Union[str, Path]) -> bool:
        """检查路径是否在允许的范围内"""
        try:
            path = Path(path).resolve()
            return any(
                path == allowed_path or allowed_path in path.parents
                for allowed_path in self.allowed_paths
            )
        except:
            return False
    
    async def list_resources(self) -> List[Dict]:
        """列出所有可用资源"""
        resources = []
        
        for allowed_path in self.allowed_paths:
            if allowed_path.exists():
                for file_path in allowed_path.rglob("*"):
                    if file_path.is_file():
                        uri = f"file://{file_path.as_posix()}"
                        resources.append({
                            "uri": uri,
                            "name": file_path.name,
                            "description": f"文件: {file_path.relative_to(allowed_path)}",
                            "mimeType": mimetypes.guess_type(str(file_path))[0] or "text/plain"
                        })
        
        return resources
    
    async def read_resource(self, uri: str) -> Dict:
        """读取资源内容"""
        if not uri.startswith("file://"):
            raise ValueError("不支持的URI格式")
        
        file_path = Path(uri[7:])  # 移除 "file://" 前缀
        
        if not self._is_path_allowed(file_path):
            raise PermissionError("路径不在允许范围内")
        
        if not file_path.exists():
            raise FileNotFoundError("文件不存在")
        
        if file_path.stat().st_size > self.max_file_size:
            raise ValueError("文件太大")
        
        try:
            # 尝试以文本模式读取
            content = file_path.read_text(encoding='utf-8')
            return {
                "contents": [{
                    "uri": uri,
                    "text": content,
                    "mimeType": "text/plain"
                }]
            }
        except UnicodeDecodeError:
            # 如果不是文本文件，返回二进制内容的信息
            return {
                "contents": [{
                    "uri": uri,
                    "text": f"[二进制文件: {file_path.name}, 大小: {file_path.stat().st_size} 字节]",
                    "mimeType": "application/octet-stream"
                }]
            }
    
    async def call_tool(self, name: str, arguments: Dict) -> Dict:
        """调用工具"""
        if name == "list_files":
            return await self._list_files(arguments.get("path", "."))
        elif name == "read_file":
            return await self._read_file(arguments["path"])
        elif name == "write_file":
            return await self._write_file(arguments["path"], arguments["content"])
        elif name == "get_file_info":
            return await self._get_file_info(arguments["path"])
        else:
            raise ValueError(f"未知工具: {name}")
    
    async def _list_files(self, path: str) -> Dict:
        """列出文件"""
        target_path = Path(path).resolve()
        
        if not self._is_path_allowed(target_path):
            return {
                "content": [{
                    "type": "text",
                    "text": f"错误: 路径 '{path}' 不在允许范围内"
                }]
            }
        
        if not target_path.exists():
            return {
                "content": [{
                    "type": "text", 
                    "text": f"错误: 路径 '{path}' 不存在"
                }]
            }
        
        if not target_path.is_dir():
            return {
                "content": [{
                    "type": "text",
                    "text": f"错误: '{path}' 不是一个目录"
                }]
            }
        
        items = []
        for item in target_path.iterdir():
            item_type = "目录" if item.is_dir() else "文件"
            size = item.stat().st_size if item.is_file() else "-"
            modified = datetime.fromtimestamp(item.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            
            items.append(f"{item_type:4} {item.name:30} {str(size):>10} {modified}")
        
        result = f"目录内容: {target_path}\n\n"
        result += f"{'类型':4} {'名称':30} {'大小':>10} {'修改时间'}\n"
        result += "-" * 70 + "\n"
        result += "\n".join(items)
        
        return {
            "content": [{
                "type": "text",
                "text": result
            }]
        }
    
    async def _read_file(self, path: str) -> Dict:
        """读取文件"""
        file_path = Path(path).resolve()
        
        if not self._is_path_allowed(file_path):
            return {
                "content": [{
                    "type": "text",
                    "text": f"错误: 路径 '{path}' 不在允许范围内"
                }]
            }
        
        if not file_path.exists():
            return {
                "content": [{
                    "type": "text",
                    "text": f"错误: 文件 '{path}' 不存在"
                }]
            }
        
        if not file_path.is_file():
            return {
                "content": [{
                    "type": "text",
                    "text": f"错误: '{path}' 不是一个文件"
                }]
            }
        
        if file_path.stat().st_size > self.max_file_size:
            return {
                "content": [{
                    "type": "text",
                    "text": f"错误: 文件 '{path}' 太大 (>{self.max_file_size} 字节)"
                }]
            }
        
        try:
            content = file_path.read_text(encoding='utf-8')
            return {
                "content": [{
                    "type": "text",
                    "text": f"文件内容: {file_path}\n\n{content}"
                }]
            }
        except UnicodeDecodeError:
            return {
                "content": [{
                    "type": "text",
                    "text": f"错误: 文件 '{path}' 不是文本文件"
                }]
            }
    
    async def _write_file(self, path: str, content: str) -> Dict:
        """写入文件"""
        file_path = Path(path).resolve()
        
        if not self._is_path_allowed(file_path):
            return {
                "content": [{
                    "type": "text",
                    "text": f"错误: 路径 '{path}' 不在允许范围内"
                }]
            }
        
        try:
            # 创建父目录（如果不存在）
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 写入文件
            file_path.write_text(content, encoding='utf-8')
            
            return {
                "content": [{
                    "type": "text",
                    "text": f"成功写入文件: {file_path}\n大小: {len(content)} 字符"
                }]
            }
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"错误: 写入文件失败 - {str(e)}"
                }]
            }
    
    async def _get_file_info(self, path: str) -> Dict:
        """获取文件信息"""
        file_path = Path(path).resolve()
        
        if not self._is_path_allowed(file_path):
            return {
                "content": [{
                    "type": "text",
                    "text": f"错误: 路径 '{path}' 不在允许范围内"
                }]
            }
        
        if not file_path.exists():
            return {
                "content": [{
                    "type": "text",
                    "text": f"错误: 路径 '{path}' 不存在"
                }]
            }
        
        stat_info = file_path.stat()
        info = {
            "路径": str(file_path),
            "名称": file_path.name,
            "类型": "目录" if file_path.is_dir() else "文件",
            "大小": f"{stat_info.st_size} 字节",
            "创建时间": datetime.fromtimestamp(stat_info.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
            "修改时间": datetime.fromtimestamp(stat_info.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "访问时间": datetime.fromtimestamp(stat_info.st_atime).strftime("%Y-%m-%d %H:%M:%S"),
            "权限": oct(stat_info.st_mode)[-3:],
        }
        
        if file_path.is_file():
            info["MIME类型"] = mimetypes.guess_type(str(file_path))[0] or "未知"
        
        result = "文件信息:\n\n"
        for key, value in info.items():
            result += f"{key:8}: {value}\n"
        
        return {
            "content": [{
                "type": "text",
                "text": result
            }]
        }
    
    async def get_prompt(self, name: str, arguments: Dict) -> Dict:
        """获取提示"""
        if name == "analyze_file":
            file_path = arguments["file_path"]
            return {
                "description": f"分析文件: {file_path}",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": f"请分析以下文件的内容，包括文件类型、结构、主要内容和任何值得注意的特点:\n\n文件路径: {file_path}"
                        }
                    }
                ]
            }
        elif name == "summarize_directory":
            directory_path = arguments["directory_path"] 
            return {
                "description": f"总结目录: {directory_path}",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": f"请总结以下目录的内容，包括文件数量、文件类型分布、目录结构和整体用途:\n\n目录路径: {directory_path}"
                        }
                    }
                ]
            }
        else:
            raise ValueError(f"未知提示: {name}")

# 简化的服务器运行逻辑（实际应该使用官方MCP协议）
async def run_server():
    """运行文件系统服务器"""
    server = FilesystemServer(allowed_paths=["./test_data", "./examples"])
    
    print("文件系统MCP服务器已启动")
    print("支持的功能:")
    print("- Resources: 文件系统资源访问")
    print("- Tools: list_files, read_file, write_file, get_file_info")
    print("- Prompts: analyze_file, summarize_directory")
    print("\n等待客户端连接...")
    
    # 在实际实现中，这里应该启动MCP协议的监听器
    # 现在只是保持服务器运行
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n服务器停止")

if __name__ == "__main__":
    asyncio.run(run_server()) 
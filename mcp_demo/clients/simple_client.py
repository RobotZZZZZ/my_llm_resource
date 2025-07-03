#!/usr/bin/env python3
"""
简单MCP客户端

这个客户端展示了如何连接到MCP服务器并使用其功能。
包含基础的资源访问、工具调用和提示使用示例。
"""

import asyncio
import json
import sys
import os
from typing import Dict, List, Any
from pathlib import Path

# 添加服务器路径以便导入
sys.path.append(str(Path(__file__).parent.parent / "servers"))

from filesystem_server import FilesystemServer
from calculator_server import CalculatorServer
from weather_server import WeatherServer

class MCPClient:
    """简化的MCP客户端"""
    
    def __init__(self, name: str):
        self.name = name
        self.connected_servers = {}
    
    def connect_server(self, server_name: str, server_instance):
        """连接到服务器"""
        self.connected_servers[server_name] = server_instance
        print(f"✓ 已连接到服务器: {server_name}")
    
    async def list_resources(self, server_name: str = None) -> List[Dict]:
        """列出资源"""
        if server_name:
            if server_name not in self.connected_servers:
                raise ValueError(f"未连接到服务器: {server_name}")
            server = self.connected_servers[server_name]
            if hasattr(server, 'list_resources'):
                return await server.list_resources()
        else:
            # 列出所有服务器的资源
            all_resources = []
            for name, server in self.connected_servers.items():
                if hasattr(server, 'list_resources'):
                    resources = await server.list_resources()
                    for resource in resources:
                        resource['server'] = name
                    all_resources.extend(resources)
            return all_resources
        return []
    
    async def read_resource(self, server_name: str, uri: str) -> Dict:
        """读取资源"""
        if server_name not in self.connected_servers:
            raise ValueError(f"未连接到服务器: {server_name}")
        
        server = self.connected_servers[server_name]
        if hasattr(server, 'read_resource'):
            return await server.read_resource(uri)
        else:
            raise ValueError(f"服务器 {server_name} 不支持资源读取")
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict) -> Dict:
        """调用工具"""
        if server_name not in self.connected_servers:
            raise ValueError(f"未连接到服务器: {server_name}")
        
        server = self.connected_servers[server_name]
        if hasattr(server, 'call_tool'):
            return await server.call_tool(tool_name, arguments)
        else:
            raise ValueError(f"服务器 {server_name} 不支持工具调用")
    
    async def get_prompt(self, server_name: str, prompt_name: str, arguments: Dict) -> Dict:
        """获取提示"""
        if server_name not in self.connected_servers:
            raise ValueError(f"未连接到服务器: {server_name}")
        
        server = self.connected_servers[server_name]
        if hasattr(server, 'get_prompt'):
            return await server.get_prompt(prompt_name, arguments)
        else:
            raise ValueError(f"服务器 {server_name} 不支持提示")

class SimpleClient:
    """简单客户端实现"""
    
    def __init__(self):
        self.client = MCPClient("SimpleClient")
        self.setup_servers()
    
    def setup_servers(self):
        """设置服务器连接"""
        # 连接到文件系统服务器
        fs_server = FilesystemServer(
            allowed_paths=["./test_data", "./examples"],
            max_file_size=1024*1024  # 1MB
        )
        self.client.connect_server("filesystem", fs_server)
        
        # 连接到计算器服务器
        calc_server = CalculatorServer(
            precision=6,
            max_operations=100
        )
        self.client.connect_server("calculator", calc_server)
        
        # 连接到天气服务器
        weather_server = WeatherServer(
            cache_duration=60,  # 1分钟缓存
            default_units="celsius"
        )
        self.client.connect_server("weather", weather_server)
    
    async def demo_resources(self):
        """演示资源功能"""
        print("\n" + "="*50)
        print("🗂️  资源演示 (Resources Demo)")
        print("="*50)
        
        try:
            # 列出所有资源
            print("📋 列出所有可用资源:")
            resources = await self.client.list_resources()
            for i, resource in enumerate(resources[:5], 1):  # 只显示前5个
                print(f"  {i}. {resource.get('name', 'Unknown')} ({resource.get('server', 'Unknown')})")
                print(f"     URI: {resource.get('uri', 'N/A')}")
                print(f"     描述: {resource.get('description', 'N/A')}")
                print()
            
            if len(resources) > 5:
                print(f"     ... 还有 {len(resources) - 5} 个资源")
            
            # 读取一个文件系统资源示例
            print("📖 读取文件系统资源示例:")
            try:
                # 创建一个测试文件
                test_file_path = Path("./test_data/demo.txt")
                test_file_path.parent.mkdir(parents=True, exist_ok=True)
                test_file_path.write_text("这是一个MCP演示文件\n包含一些示例内容。", encoding='utf-8')
                
                # 将文件路径转换为URI格式 (--> 绝对路径 --> POSIX格式)
                file_uri = f"file://{test_file_path.resolve().as_posix()}"
                # 读取文件
                result = await self.client.read_resource("filesystem", file_uri)
                
                if result and 'contents' in result:
                    content = result['contents'][0]
                    print(f"  文件URI: {content.get('uri', 'N/A')}")
                    print(f"  内容预览: {content.get('text', 'N/A')[:100]}...")
                
            except Exception as e:
                print(f"  ❌ 读取文件资源失败: {str(e)}")
            
            # 读取天气资源示例
            print("🌤️  读取天气资源示例:")
            try:
                weather_uri = "weather://北京/current"
                result = await self.client.read_resource("weather", weather_uri)
                
                if result and 'contents' in result:
                    content = result['contents'][0]
                    weather_data = json.loads(content.get('text', '{}'))
                    print(f"  城市: {weather_data.get('city', 'N/A')}")
                    print(f"  温度: {weather_data.get('temperature', 'N/A')}{weather_data.get('temperature_unit', '')}")
                    print(f"  天气: {weather_data.get('condition', 'N/A')}")
                
            except Exception as e:
                print(f"  ❌ 读取天气资源失败: {str(e)}")
                
        except Exception as e:
            print(f"❌ 资源演示失败: {str(e)}")
    
    async def demo_tools(self):
        """演示工具功能"""
        print("\n" + "="*50)
        print("🔧 工具演示 (Tools Demo)")
        print("="*50)
        
        # 文件系统工具演示
        print("📁 文件系统工具演示:")
        try:
            # 列出文件
            result = await self.client.call_tool("filesystem", "list_files", {"path": "./test_data"})
            if result and 'content' in result:
                print("  📝 文件列表:")
                print("  " + result['content'][0]['text'][:200] + "...")
            
            # 文件信息
            result = await self.client.call_tool("filesystem", "get_file_info", {"path": "./test_data/demo.txt"})
            if result and 'content' in result:
                print("  ℹ️  文件信息:")
                print("  " + result['content'][0]['text'][:200] + "...")
                
        except Exception as e:
            print(f"  ❌ 文件系统工具演示失败: {str(e)}")
        
        # 计算器工具演示
        print("\n🧮 计算器工具演示:")
        try:
            # 基础计算
            result = await self.client.call_tool("calculator", "calculate", {"expression": "sqrt(16) + 2 * 3"})
            if result and 'content' in result:
                print("  🔢 计算结果:")
                print("  " + result['content'][0]['text'])
            
            # 统计分析
            numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            result = await self.client.call_tool("calculator", "statistics", {
                "numbers": numbers,
                "operations": ["mean", "median", "std", "min", "max"]
            })
            if result and 'content' in result:
                print("  📊 统计分析:")
                print("  " + result['content'][0]['text'][:200] + "...")
            
            # 单位转换
            result = await self.client.call_tool("calculator", "convert_units", {
                "value": 100,
                "from_unit": "cm",
                "to_unit": "m",
                "unit_type": "length"
            })
            if result and 'content' in result:
                print("  🔄 单位转换:")
                print("  " + result['content'][0]['text'])
                
        except Exception as e:
            print(f"  ❌ 计算器工具演示失败: {str(e)}")
        
        # 天气工具演示
        print("\n🌤️  天气工具演示:")
        try:
            # 当前天气
            result = await self.client.call_tool("weather", "get_current_weather", {"city": "北京"})
            if result and 'content' in result:
                print("  🌡️  当前天气:")
                print("  " + result['content'][0]['text'][:300] + "...")
            
            # 天气预报
            result = await self.client.call_tool("weather", "get_weather_forecast", {
                "city": "上海",
                "days": 3
            })
            if result and 'content' in result:
                print("  📅 天气预报:")
                print("  " + result['content'][0]['text'][:300] + "...")
            
            # 城市比较
            result = await self.client.call_tool("weather", "compare_weather", {
                "cities": ["北京", "上海", "广州"]
            })
            if result and 'content' in result:
                print("  🏙️  城市天气比较:")
                print("  " + result['content'][0]['text'][:300] + "...")
                
        except Exception as e:
            print(f"  ❌ 天气工具演示失败: {str(e)}")
    
    async def demo_prompts(self):
        """演示提示功能"""
        print("\n" + "="*50)
        print("💭 提示演示 (Prompts Demo)")
        print("="*50)
        
        # 文件分析提示
        print("📄 文件分析提示:")
        try:
            result = await self.client.get_prompt("filesystem", "analyze_file", {
                "file_path": "./test_data/demo.txt"
            })
            if result:
                print(f"  📝 提示描述: {result.get('description', 'N/A')}")
                if 'messages' in result and result['messages']:
                    message = result['messages'][0]
                    content = message.get('content', {})
                    text = content.get('text', 'N/A')
                    print(f"  💬 提示内容: {text[:200]}...")
                    
        except Exception as e:
            print(f"  ❌ 文件分析提示失败: {str(e)}")
        
        # 数学问题求解提示
        print("\n🧮 数学问题求解提示:")
        try:
            result = await self.client.get_prompt("calculator", "math_problem_solver", {
                "problem": "一个圆的半径是5米，求它的面积和周长"
            })
            if result:
                print(f"  📝 提示描述: {result.get('description', 'N/A')}")
                if 'messages' in result and result['messages']:
                    message = result['messages'][0]
                    content = message.get('content', {})
                    text = content.get('text', 'N/A')
                    print(f"  💬 提示内容: {text[:200]}...")
                    
        except Exception as e:
            print(f"  ❌ 数学问题求解提示失败: {str(e)}")
        
        # 天气报告提示
        print("\n🌤️  天气报告提示:")
        try:
            result = await self.client.get_prompt("weather", "weather_report", {
                "city": "深圳",
                "report_type": "detailed"
            })
            if result:
                print(f"  📝 提示描述: {result.get('description', 'N/A')}")
                if 'messages' in result and result['messages']:
                    message = result['messages'][0]
                    content = message.get('content', {})
                    text = content.get('text', 'N/A')
                    print(f"  💬 提示内容: {text[:200]}...")
                    
        except Exception as e:
            print(f"  ❌ 天气报告提示失败: {str(e)}")
    
    async def demo_integration(self):
        """演示集成场景"""
        print("\n" + "="*50)
        print("🔗 集成场景演示 (Integration Demo)")
        print("="*50)
        
        print("📊 综合数据分析场景:")
        print("  场景: 分析项目数据文件并生成统计报告")
        
        try:
            # 1. 创建示例数据文件
            data_content = """项目数据报告
销售数据: 100, 120, 95, 110, 130, 85, 140, 115, 125, 105
温度数据: 22.5, 23.1, 21.8, 24.2, 22.9, 23.5, 21.5, 24.0, 22.3, 23.8
"""
            data_file_path = Path("./test_data/project_data.txt")
            data_file_path.write_text(data_content, encoding='utf-8')
            
            # 2. 读取文件
            print("  1️⃣  读取数据文件...")
            result = await self.client.call_tool("filesystem", "read_file", {
                "path": str(data_file_path)
            })
            if result and 'content' in result:
                file_content = result['content'][0]['text']
                print("     ✓ 文件读取成功")
            
            # 3. 提取数据并进行统计分析
            print("  2️⃣  分析销售数据...")
            sales_data = [100, 120, 95, 110, 130, 85, 140, 115, 125, 105]
            result = await self.client.call_tool("calculator", "statistics", {
                "numbers": sales_data,
                "operations": ["mean", "median", "std", "min", "max", "sum"]
            })
            if result and 'content' in result:
                print("     ✓ 销售数据分析完成")
                stats_text = result['content'][0]['text']
            
            # 4. 分析温度数据
            print("  3️⃣  分析温度数据...")
            temp_data = [22.5, 23.1, 21.8, 24.2, 22.9, 23.5, 21.5, 24.0, 22.3, 23.8]
            result = await self.client.call_tool("calculator", "statistics", {
                "numbers": temp_data,
                "operations": ["mean", "min", "max"]
            })
            if result and 'content' in result:
                print("     ✓ 温度数据分析完成")
            
            # 5. 获取当前天气作为对比
            print("  4️⃣  获取当前天气信息...")
            result = await self.client.call_tool("weather", "get_current_weather", {
                "city": "北京"
            })
            if result and 'content' in result:
                print("     ✓ 天气信息获取完成")
            
            # 6. 生成综合报告
            print("  5️⃣  生成综合分析报告...")
            report_content = f"""
MCP集成演示 - 综合数据分析报告
==========================================
生成时间: {asyncio.get_event_loop().time()}

数据来源: {data_file_path}
分析工具: MCP文件系统服务器 + 计算器服务器 + 天气服务器

销售数据分析:
{stats_text}

结论: MCP协议成功实现了多个服务器的协同工作！
"""
            
            report_path = Path("./test_data/analysis_report.txt")
            result = await self.client.call_tool("filesystem", "write_file", {
                "path": str(report_path),
                "content": report_content
            })
            
            if result and 'content' in result:
                print("     ✓ 分析报告已生成")
                print(f"     📄 报告位置: {report_path}")
            
            print("\n🎉 集成场景演示完成！")
            print("   通过MCP协议，我们成功地:")
            print("   • 从文件系统读取数据")
            print("   • 使用计算器进行统计分析") 
            print("   • 获取天气信息作为补充")
            print("   • 生成综合分析报告")
            
        except Exception as e:
            print(f"  ❌ 集成场景演示失败: {str(e)}")
    
    async def run_demo(self):
        """运行完整演示"""
        print("🚀 MCP简单客户端演示")
        print("=" * 60)
        print("这个演示将展示MCP的三个核心功能:")
        print("1. Resources - 资源访问")
        print("2. Tools - 工具调用")
        print("3. Prompts - 提示获取")
        print("4. Integration - 集成场景")
        
        # 确保测试数据目录存在
        Path("./test_data").mkdir(exist_ok=True)
        
        await self.demo_resources()
        await self.demo_tools()
        await self.demo_prompts()
        await self.demo_integration()
        
        print("\n" + "="*60)
        print("✅ MCP演示完成！")
        print("MCP协议成功地标准化了AI系统与外部工具、数据源的集成方式。")
        print("通过这个演示，你看到了如何使用MCP构建强大的AI应用程序。")

async def main():
    """主函数"""
    client = SimpleClient()
    await client.run_demo()

if __name__ == "__main__":
    asyncio.run(main()) 
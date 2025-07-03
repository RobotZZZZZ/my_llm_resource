#!/usr/bin/env python3
"""
MCP基础使用示例

这个脚本展示了如何在实际项目中使用MCP客户端和服务器。
"""

import asyncio
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from clients.simple_client import SimpleClient

async def basic_file_operations():
    """基础文件操作示例"""
    print("🗂️  文件操作示例")
    print("-" * 30)
    
    client = SimpleClient()
    
    # 1. 创建测试文件
    test_content = """
这是一个测试文件
包含一些示例数据：
- 项目名称: MCP Demo
- 版本: 1.0.0
- 作者: 开发团队
"""
    
    try:
        result = await client.client.call_tool("filesystem", "write_file", {
            "path": "./test_data/basic_test.txt",
            "content": test_content
        })
        print("✓ 文件创建成功")
        
        # 2. 读取文件
        result = await client.client.call_tool("filesystem", "read_file", {
            "path": "./test_data/basic_test.txt"
        })
        print("✓ 文件读取成功")
        
        # 3. 获取文件信息
        result = await client.client.call_tool("filesystem", "get_file_info", {
            "path": "./test_data/basic_test.txt"
        })
        print("✓ 文件信息获取成功")
        
    except Exception as e:
        print(f"❌ 文件操作失败: {e}")

async def basic_calculations():
    """基础计算示例"""
    print("\n🧮 计算示例")
    print("-" * 30)
    
    client = SimpleClient()
    
    try:
        # 1. 简单计算
        result = await client.client.call_tool("calculator", "calculate", {
            "expression": "2 * 3.14159 * 5"  # 圆的周长
        })
        if result and 'content' in result:
            print("✓ 圆周长计算:", result['content'][0]['text'].split('\n')[-1])
        
        # 2. 数据统计
        data = [85, 92, 78, 96, 89, 84, 91, 88, 93, 87]  # 学生成绩
        result = await client.client.call_tool("calculator", "statistics", {
            "numbers": data,
            "operations": ["mean", "median", "std"]
        })
        if result and 'content' in result:
            print("✓ 成绩统计完成")
        
        # 3. 单位转换
        result = await client.client.call_tool("calculator", "convert_units", {
            "value": 25,
            "from_unit": "celsius",
            "to_unit": "fahrenheit",
            "unit_type": "temperature"
        })
        if result and 'content' in result:
            print("✓ 温度转换完成")
            
    except Exception as e:
        print(f"❌ 计算失败: {e}")

async def basic_weather_query():
    """基础天气查询示例"""
    print("\n🌤️  天气查询示例")
    print("-" * 30)
    
    client = SimpleClient()
    
    try:
        # 1. 获取当前天气
        result = await client.client.call_tool("weather", "get_current_weather", {
            "city": "北京"
        })
        if result and 'content' in result:
            print("✓ 北京天气查询成功")
        
        # 2. 城市搜索
        result = await client.client.call_tool("weather", "search_cities", {
            "query": "上海"
        })
        if result and 'content' in result:
            print("✓ 城市搜索完成")
        
        # 3. 天气比较
        result = await client.client.call_tool("weather", "compare_weather", {
            "cities": ["北京", "上海"]
        })
        if result and 'content' in result:
            print("✓ 城市天气比较完成")
            
    except Exception as e:
        print(f"❌ 天气查询失败: {e}")

async def integrated_workflow():
    """集成工作流示例"""
    print("\n🔗 集成工作流示例")
    print("-" * 30)
    
    client = SimpleClient()
    
    try:
        # 工作流：分析销售数据并生成报告
        
        # 1. 准备数据
        sales_data = [120, 135, 98, 156, 142, 108, 167, 134, 149, 128]
        
        # 2. 统计分析
        result = await client.client.call_tool("calculator", "statistics", {
            "numbers": sales_data,
            "operations": ["mean", "sum", "max", "min"]
        })
        
        if result and 'content' in result:
            stats_text = result['content'][0]['text']
        
        # 3. 获取天气信息（可能影响销售）
        result = await client.client.call_tool("weather", "get_current_weather", {
            "city": "上海"
        })
        
        if result and 'content' in result:
            weather_text = result['content'][0]['text']
        
        # 4. 生成综合报告
        report = f"""
销售数据分析报告
===============

{stats_text}

天气因素分析：
{weather_text[:200]}...

结论：通过MCP协议，我们成功整合了数据分析和天气信息，
为销售数据提供了更全面的分析视角。
"""
        
        result = await client.client.call_tool("filesystem", "write_file", {
            "path": "./test_data/sales_report.txt",
            "content": report
        })
        
        if result and 'content' in result:
            print("✓ 销售分析报告生成完成")
            print("  报告位置: ./test_data/sales_report.txt")
        
    except Exception as e:
        print(f"❌ 集成工作流失败: {e}")

async def main():
    """主函数"""
    print("🚀 MCP基础使用示例")
    print("=" * 50)
    
    # 确保测试目录存在
    Path("./test_data").mkdir(exist_ok=True)
    
    await basic_file_operations()
    await basic_calculations()
    await basic_weather_query()
    await integrated_workflow()
    
    print("\n✅ 所有示例执行完成！")
    print("💡 提示: 查看 ./test_data/ 目录中生成的文件")

if __name__ == "__main__":
    asyncio.run(main()) 
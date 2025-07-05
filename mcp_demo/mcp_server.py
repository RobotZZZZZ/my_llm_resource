from fastmcp import FastMCP

# 创建MCP Server
app = FastMCP("simple_server")

# 注册天气查询工具，用于查询城市天气
@app.tool(name="weather", description="城市天气查询")
def get_weather(city: str) -> dict:
    """
    查询指定城市的天气信息
    """
    # 用固定数据进行演示，实际应用中，需要调用外部的api
    weather_data = {
        "北京": {"temp": 25, "condition": "晴"},
        "上海": {"temp": 28, "condition": "多云"},
    }
    return weather_data.get(city, {"error": "未找到该城市"})

# 注册股票查询工具，用于获取指定股票代码的价格信息
@app.tool(name="stock", description="股票价格查询")
def get_stock(code: str) -> dict:
    """
    查询指定股票代码的价格信息
    """
    # 用固定数据进行演示，实际应用中，需要调用外部的api
    stock_data = {
        "600519": {"name": "贵州茅台", "price": 1825.0},
        "000858": {"name": "五粮液", "price": 158.3}
    }
    return stock_data.get(code, {"error": "未找到该股票代码"})

# 启动MCP Server
if __name__ == "__main__":
    app.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000,
        path="/mcp",
        log_level="debug",
    )




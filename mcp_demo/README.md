# MCP Demo - 模型上下文协议演示

这个demo展示了如何使用Model Context Protocol (MCP)来构建AI应用程序，包含多个服务器和客户端示例。

## 项目结构

```
mcp_demo/
├── README.md                 # 本文档
├── requirements.txt          # Python依赖
├── config/
│   └── mcp_servers.json     # MCP服务器配置
├── servers/
│   ├── __init__.py
│   ├── filesystem_server.py # 文件系统服务器
│   ├── calculator_server.py # 计算器服务器
│   └── weather_server.py    # 天气服务器
├── clients/
│   ├── __init__.py
│   ├── simple_client.py     # 简单客户端示例
│   └── chat_client.py       # 聊天客户端示例
├── examples/
│   ├── basic_usage.py       # 基本使用示例
│   └── advanced_usage.py    # 高级使用示例
└── test_data/
    ├── sample.txt           # 测试文件
    └── notes.md            # 测试笔记
```

## MCP核心概念演示

### 1. Resources (资源)
- **文件系统服务器**: 展示如何暴露本地文件作为资源
- **支持的操作**: 列出文件、读取文件内容、获取文件信息

### 2. Tools (工具)
- **计算器服务器**: 展示如何暴露计算功能
- **天气服务器**: 展示如何暴露API调用功能
- **支持的操作**: 数学计算、天气查询、数据处理

### 3. Prompts (提示)
- **预定义模板**: 文件分析、代码审查、数据总结等
- **参数化提示**: 支持动态参数的提示模板

## 快速开始

### 1. 安装依赖
```bash
cd mcp_demo
pip install -r requirements.txt
```

### 2. 启动服务器
```bash
# 启动文件系统服务器
python servers/filesystem_server.py

# 启动计算器服务器（新终端）
python servers/calculator_server.py

# 启动天气服务器（新终端）
python servers/weather_server.py
```

### 3. 运行客户端
```bash
# 基本客户端示例
python clients/simple_client.py

# 聊天客户端示例
python clients/chat_client.py
```

## 示例场景

### 场景1: 文件分析助手
1. 客户端连接到文件系统服务器
2. 列出指定目录的文件
3. 读取文件内容并分析
4. 生成分析报告

### 场景2: 数据处理助手
1. 客户端连接到计算器和文件系统服务器
2. 从文件中读取数据
3. 使用计算器进行数学运算
4. 生成处理结果

### 场景3: 智能助手
1. 客户端连接到所有服务器
2. 根据用户查询选择合适的工具
3. 组合多个服务器的功能
4. 提供综合性回答

## 架构说明

### 客户端-服务器通信
- 使用JSON-RPC 2.0协议
- 支持stdio和HTTP传输
- 实现能力协商机制

### 安全性特性
- 用户确认机制
- 资源访问控制
- 工具执行权限管理

### 扩展性设计
- 插件化架构
- 动态服务发现
- 标准化接口

## 下一步

1. 查看 `examples/` 目录中的详细示例
2. 尝试修改服务器配置
3. 创建自定义MCP服务器
4. 集成到你的AI应用中

## 注意事项

- 确保Python 3.8+环境
- 某些功能需要网络连接
- 文件操作需要适当的权限
- 建议在虚拟环境中运行 
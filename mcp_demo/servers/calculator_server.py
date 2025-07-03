#!/usr/bin/env python3
"""
计算器MCP服务器

这个服务器展示了如何使用MCP暴露计算工具。
主要演示Tools和Prompts功能。
"""

import asyncio
import json
import math
import statistics
from typing import Any, Dict, List, Union
import re

# 复用前面的MCPServer基础类
from filesystem_server import MCPServer

class CalculatorServer:
    def __init__(self, precision: int = 10, max_operations: int = 1000):
        """
        初始化计算器服务器
        
        Args:
            precision: 计算精度
            max_operations: 最大操作数量
        """
        self.server = MCPServer("calculator-server", "1.0.0")
        self.precision = precision
        self.max_operations = max_operations
        self.operation_count = 0
        
        self._setup_capabilities()
    
    def _setup_capabilities(self):
        """设置服务器能力"""
        # 基础数学运算工具
        self.server.add_tool(
            "calculate",
            "执行基础数学计算（支持 +, -, *, /, **, %, sqrt, sin, cos, tan, log 等）",
            {
                "expression": {"type": "string", "description": "数学表达式，如 '2 + 3 * 4' 或 'sqrt(16)'"}
            }
        )
        
        self.server.add_tool(
            "statistics",
            "计算数字列表的统计信息",
            {
                "numbers": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "数字列表"
                },
                "operations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "要计算的统计量：mean, median, mode, std, var, min, max, sum",
                    "default": ["mean", "median", "std"]
                }
            }
        )
        
        self.server.add_tool(
            "convert_units",
            "单位转换",
            {
                "value": {"type": "number", "description": "要转换的数值"},
                "from_unit": {"type": "string", "description": "源单位"},
                "to_unit": {"type": "string", "description": "目标单位"},
                "unit_type": {
                    "type": "string", 
                    "enum": ["length", "weight", "temperature", "area", "volume"],
                    "description": "单位类型"
                }
            }
        )
        
        self.server.add_tool(
            "solve_equation",
            "解简单的代数方程（一元一次方程）",
            {
                "equation": {"type": "string", "description": "方程式，如 '2x + 5 = 13'"}
            }
        )
        
        self.server.add_tool(
            "generate_sequence",
            "生成数学序列",
            {
                "sequence_type": {
                    "type": "string",
                    "enum": ["arithmetic", "geometric", "fibonacci", "prime"],
                    "description": "序列类型"
                },
                "start": {"type": "number", "description": "起始值", "default": 1},
                "length": {"type": "integer", "description": "序列长度", "default": 10},
                "step": {"type": "number", "description": "步长（仅用于等差/等比数列）", "default": 1}
            }
        )
        
        # 添加提示
        self.server.add_prompt(
            "math_problem_solver",
            "数学问题求解助手",
            [
                {"name": "problem", "description": "数学问题描述", "required": True}
            ]
        )
        
        self.server.add_prompt(
            "data_analysis",
            "数据分析助手",
            [
                {"name": "data", "description": "要分析的数据", "required": True},
                {"name": "analysis_type", "description": "分析类型", "required": False}
            ]
        )
    
    def _check_operation_limit(self):
        """检查操作次数限制"""
        if self.operation_count >= self.max_operations:
            raise RuntimeError(f"已达到最大操作次数限制: {self.max_operations}")
        self.operation_count += 1
    
    def _safe_eval(self, expression: str) -> float:
        """安全的数学表达式求值"""
        # 只允许安全的数学运算
        allowed_names = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sqrt": math.sqrt, "pow": pow,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "asin": math.asin, "acos": math.acos, "atan": math.atan,
            "log": math.log, "log10": math.log10, "exp": math.exp,
            "pi": math.pi, "e": math.e,
            "ceil": math.ceil, "floor": math.floor,
            "degrees": math.degrees, "radians": math.radians
        }
        
        # 移除空格并检查非法字符
        expression = expression.replace(" ", "")
        if re.search(r'[a-zA-Z_][a-zA-Z0-9_]*', expression):
            # 检查是否包含非允许的标识符
            for match in re.finditer(r'[a-zA-Z_][a-zA-Z0-9_]*', expression):
                if match.group() not in allowed_names:
                    raise ValueError(f"不允许的函数或变量: {match.group()}")
        
        # 替换常用函数
        expression = expression.replace("sqrt", "sqrt")
        
        try:
            # 使用受限的命名空间进行求值
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return round(result, self.precision)
        except Exception as e:
            raise ValueError(f"表达式求值错误: {str(e)}")
    
    async def call_tool(self, name: str, arguments: Dict) -> Dict:
        """调用工具"""
        self._check_operation_limit()
        
        if name == "calculate":
            return await self._calculate(arguments["expression"])
        elif name == "statistics":
            return await self._statistics(arguments["numbers"], arguments.get("operations", ["mean", "median", "std"]))
        elif name == "convert_units":
            return await self._convert_units(
                arguments["value"], 
                arguments["from_unit"], 
                arguments["to_unit"], 
                arguments["unit_type"]
            )
        elif name == "solve_equation":
            return await self._solve_equation(arguments["equation"])
        elif name == "generate_sequence":
            return await self._generate_sequence(
                arguments["sequence_type"],
                arguments.get("start", 1),
                arguments.get("length", 10),
                arguments.get("step", 1)
            )
        else:
            raise ValueError(f"未知工具: {name}")
    
    async def _calculate(self, expression: str) -> Dict:
        """执行计算"""
        try:
            result = self._safe_eval(expression)
            return {
                "content": [{
                    "type": "text",
                    "text": f"计算结果:\n表达式: {expression}\n结果: {result}"
                }]
            }
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"计算错误:\n表达式: {expression}\n错误: {str(e)}"
                }]
            }
    
    async def _statistics(self, numbers: List[float], operations: List[str]) -> Dict:
        """计算统计信息"""
        try:
            if not numbers:
                return {
                    "content": [{
                        "type": "text",
                        "text": "错误: 数字列表不能为空"
                    }]
                }
            
            results = []
            
            for op in operations:
                if op == "mean":
                    value = statistics.mean(numbers)
                    results.append(f"平均值: {round(value, self.precision)}")
                elif op == "median":
                    value = statistics.median(numbers)
                    results.append(f"中位数: {round(value, self.precision)}")
                elif op == "mode":
                    try:
                        value = statistics.mode(numbers)
                        results.append(f"众数: {round(value, self.precision)}")
                    except statistics.StatisticsError:
                        results.append("众数: 无唯一众数")
                elif op == "std":
                    if len(numbers) > 1:
                        value = statistics.stdev(numbers)
                        results.append(f"标准差: {round(value, self.precision)}")
                    else:
                        results.append("标准差: 需要至少2个数据点")
                elif op == "var":
                    if len(numbers) > 1:
                        value = statistics.variance(numbers)
                        results.append(f"方差: {round(value, self.precision)}")
                    else:
                        results.append("方差: 需要至少2个数据点")
                elif op == "min":
                    value = min(numbers)
                    results.append(f"最小值: {value}")
                elif op == "max":
                    value = max(numbers)
                    results.append(f"最大值: {value}")
                elif op == "sum":
                    value = sum(numbers)
                    results.append(f"总和: {round(value, self.precision)}")
                else:
                    results.append(f"未知操作: {op}")
            
            result_text = f"统计分析结果:\n数据: {numbers}\n数据量: {len(numbers)}\n\n" + "\n".join(results)
            
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
                    "text": f"统计计算错误: {str(e)}"
                }]
            }
    
    async def _convert_units(self, value: float, from_unit: str, to_unit: str, unit_type: str) -> Dict:
        """单位转换"""
        try:
            # 定义转换表（到基础单位的乘数）
            conversions = {
                "length": {
                    "mm": 0.001, "cm": 0.01, "m": 1, "km": 1000,
                    "inch": 0.0254, "ft": 0.3048, "yard": 0.9144, "mile": 1609.34
                },
                "weight": {
                    "mg": 0.001, "g": 1, "kg": 1000, "ton": 1000000,
                    "oz": 28.3495, "lb": 453.592
                },
                "temperature": {
                    "celsius": None, "fahrenheit": None, "kelvin": None
                },
                "area": {
                    "mm2": 0.000001, "cm2": 0.0001, "m2": 1, "km2": 1000000,
                    "inch2": 0.00064516, "ft2": 0.092903
                },
                "volume": {
                    "ml": 0.001, "l": 1, "m3": 1000,
                    "cup": 0.236588, "pint": 0.473176, "quart": 0.946353, "gallon": 3.78541
                }
            }
            
            if unit_type not in conversions:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"错误: 不支持的单位类型 '{unit_type}'"
                    }]
                }
            
            unit_map = conversions[unit_type]
            
            # 特殊处理温度转换
            if unit_type == "temperature":
                result = self._convert_temperature(value, from_unit, to_unit)
            else:
                if from_unit not in unit_map or to_unit not in unit_map:
                    available = ", ".join(unit_map.keys())
                    return {
                        "content": [{
                            "type": "text",
                            "text": f"错误: 不支持的单位。可用单位: {available}"
                        }]
                    }
                
                # 转换到基础单位，再转换到目标单位
                base_value = value * unit_map[from_unit]
                result = base_value / unit_map[to_unit]
                result = round(result, self.precision)
            
            return {
                "content": [{
                    "type": "text",
                    "text": f"单位转换结果:\n{value} {from_unit} = {result} {to_unit}"
                }]
            }
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"单位转换错误: {str(e)}"
                }]
            }
    
    def _convert_temperature(self, value: float, from_unit: str, to_unit: str) -> float:
        """温度转换"""
        # 先转换到摄氏度
        if from_unit == "celsius":
            celsius = value
        elif from_unit == "fahrenheit":
            celsius = (value - 32) * 5/9
        elif from_unit == "kelvin":
            celsius = value - 273.15
        else:
            raise ValueError(f"不支持的温度单位: {from_unit}")
        
        # 再从摄氏度转换到目标单位
        if to_unit == "celsius":
            result = celsius
        elif to_unit == "fahrenheit":
            result = celsius * 9/5 + 32
        elif to_unit == "kelvin":
            result = celsius + 273.15
        else:
            raise ValueError(f"不支持的温度单位: {to_unit}")
        
        return round(result, self.precision)
    
    async def _solve_equation(self, equation: str) -> Dict:
        """解一元一次方程"""
        try:
            # 简单的一元一次方程求解器
            # 格式: ax + b = c 或类似形式
            equation = equation.replace(" ", "")
            
            if "=" not in equation:
                return {
                    "content": [{
                        "type": "text",
                        "text": "错误: 方程必须包含等号 '='"
                    }]
                }
            
            left, right = equation.split("=")
            
            # 简化处理：假设方程是 ax + b = c 的形式
            # 这里只是一个简单的示例实现
            if "x" not in left and "x" not in right:
                return {
                    "content": [{
                        "type": "text",
                        "text": "错误: 方程必须包含变量 'x'"
                    }]
                }
            
            # 简单的线性方程求解（这里只是演示）
            # 实际实现会更复杂
            result_text = f"方程求解:\n原方程: {equation}\n注意: 这是一个简化的求解器，仅支持简单的一元一次方程"
            
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
                    "text": f"方程求解错误: {str(e)}"
                }]
            }
    
    async def _generate_sequence(self, sequence_type: str, start: float, length: int, step: float) -> Dict:
        """生成数学序列"""
        try:
            if length <= 0 or length > 100:
                return {
                    "content": [{
                        "type": "text",
                        "text": "错误: 序列长度必须在1-100之间"
                    }]
                }
            
            sequence = []
            
            if sequence_type == "arithmetic":
                # 等差数列
                for i in range(length):
                    sequence.append(start + i * step)
            elif sequence_type == "geometric":
                # 等比数列
                current = start
                for i in range(length):
                    sequence.append(current)
                    current *= step
            elif sequence_type == "fibonacci":
                # 斐波那契数列
                if length >= 1:
                    sequence.append(0)
                if length >= 2:
                    sequence.append(1)
                for i in range(2, length):
                    sequence.append(sequence[i-1] + sequence[i-2])
            elif sequence_type == "prime":
                # 质数序列
                def is_prime(n):
                    if n < 2:
                        return False
                    for i in range(2, int(n**0.5) + 1):
                        if n % i == 0:
                            return False
                    return True
                
                num = 2
                while len(sequence) < length:
                    if is_prime(num):
                        sequence.append(num)
                    num += 1
            else:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"错误: 不支持的序列类型 '{sequence_type}'"
                    }]
                }
            
            result_text = f"{sequence_type.upper()}序列:\n"
            result_text += f"参数: 起始值={start}, 长度={length}"
            if sequence_type in ["arithmetic", "geometric"]:
                result_text += f", 步长={step}"
            result_text += f"\n序列: {sequence}"
            
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
                    "text": f"序列生成错误: {str(e)}"
                }]
            }
    
    async def get_prompt(self, name: str, arguments: Dict) -> Dict:
        """获取提示"""
        if name == "math_problem_solver":
            problem = arguments["problem"]
            return {
                "description": f"解决数学问题: {problem}",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": f"请帮我解决以下数学问题，并提供详细的解答步骤:\n\n问题: {problem}\n\n请使用计算器工具进行必要的计算，并解释每个步骤。"
                        }
                    }
                ]
            }
        elif name == "data_analysis":
            data = arguments["data"]
            analysis_type = arguments.get("analysis_type", "基础统计分析")
            return {
                "description": f"数据分析: {analysis_type}",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": f"请对以下数据进行{analysis_type}:\n\n数据: {data}\n\n请使用统计工具计算相关指标，并提供分析结论。"
                        }
                    }
                ]
            }
        else:
            raise ValueError(f"未知提示: {name}")

# 服务器运行逻辑
async def run_server():
    """运行计算器服务器"""
    server = CalculatorServer(precision=10, max_operations=1000)
    
    print("计算器MCP服务器已启动")
    print("支持的功能:")
    print("- Tools: calculate, statistics, convert_units, solve_equation, generate_sequence")
    print("- Prompts: math_problem_solver, data_analysis")
    print(f"- 操作限制: {server.max_operations} 次")
    print(f"- 计算精度: {server.precision} 位小数")
    print("\n等待客户端连接...")
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n服务器停止")

if __name__ == "__main__":
    asyncio.run(run_server()) 
"""
错误上下文重试机制 - 核心系统模块

实现智能的错误信息收集、上下文管理和Prompt增强功能，
在模型重试时将上一次的错误信息集成到新的prompt中。
"""

import json
import traceback
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import re


class ErrorSeverity(Enum):
    """错误严重程度枚举"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """错误类别枚举"""
    SYNTAX = "syntax"           # 语法错误
    RUNTIME = "runtime"         # 运行时错误
    LOGIC = "logic"            # 逻辑错误
    TIMEOUT = "timeout"        # 超时错误
    DEPENDENCY = "dependency"   # 依赖错误
    DATABASE = "database"      # 数据库错误
    NETWORK = "network"        # 网络错误
    UNKNOWN = "unknown"        # 未知错误


@dataclass
class ErrorInfo:
    """标准化的错误信息结构"""
    error_type: str                    # 错误类型名称
    error_message: str                 # 错误消息
    stack_trace: Optional[str] = None  # 堆栈跟踪
    timestamp: Optional[datetime] = None  # 时间戳
    context: Optional[Dict[str, Any]] = None  # 上下文信息
    severity: ErrorSeverity = ErrorSeverity.MEDIUM  # 严重程度
    category: ErrorCategory = ErrorCategory.UNKNOWN  # 错误类别
    retry_count: int = 0               # 重试次数
    
    def __post_init__(self):
        """初始化后处理"""
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.context is None:
            self.context = {}
    
    def to_context_string(self) -> str:
        """转换为上下文字符串，用于添加到prompt中"""
        context_str = f"错误类型: {self.error_type}\n"
        context_str += f"错误消息: {self.error_message}\n"
        context_str += f"严重程度: {self.severity.value}\n"
        context_str += f"错误类别: {self.category.value}\n"
        
        if self.stack_trace:
            # 简化堆栈跟踪，只保留关键信息
            simplified_trace = self._simplify_stack_trace(self.stack_trace)
            context_str += f"关键堆栈信息: {simplified_trace}\n"
        
        if self.context:
            # 添加重要的上下文信息
            important_context = self._extract_important_context()
            if important_context:
                context_str += f"相关上下文: {important_context}\n"
        
        return context_str
    
    def _simplify_stack_trace(self, stack_trace: str) -> str:
        """简化堆栈跟踪，提取关键信息"""
        try:
            lines = stack_trace.split('\n')
            # 保留最后几行关键错误信息
            key_lines = []
            for line in lines[-5:]:
                line = line.strip()
                if line and not line.startswith('  '):
                    key_lines.append(line)
            return ' | '.join(key_lines[-2:]) if key_lines else stack_trace[:100]
        except:
            return stack_trace[:100]
    
    def _extract_important_context(self) -> str:
        """提取重要的上下文信息"""
        try:
            important_keys = ['sql', 'query', 'operation', 'file', 'line', 'function']
            important_info = []
            
            for key in important_keys:
                if key in self.context and self.context[key]:
                    value = str(self.context[key])[:50]  # 限制长度
                    important_info.append(f"{key}={value}")
            
            return ', '.join(important_info)
        except:
            return ""


@dataclass 
class ErrorPattern:
    """错误模式识别结果"""
    pattern_type: str      # 模式类型
    frequency: int         # 出现频率
    description: str       # 模式描述
    suggested_fix: str     # 建议修复方法


@dataclass
class RetryContext:
    """重试上下文信息"""
    errors: List[ErrorInfo]              # 错误历史
    retry_count: int                     # 当前重试次数
    error_patterns: List[ErrorPattern]   # 识别的错误模式
    suggestions: List[str]               # 修复建议
    
    def format_for_prompt(self) -> str:
        """格式化为prompt文本"""
        if not self.errors:
            return ""
        
        prompt_text = f"\n🚨 **错误上下文信息** (重试第 {self.retry_count} 次):\n\n"
        
        # 添加最近的错误信息
        recent_errors = self.errors[-3:]  # 最近3个错误
        for i, error in enumerate(recent_errors, 1):
            prompt_text += f"**错误 {i}** ({error.timestamp.strftime('%H:%M:%S')}):\n"
            prompt_text += error.to_context_string()
            prompt_text += "\n"
        
        # 添加错误模式分析
        if self.error_patterns:
            prompt_text += "**识别的错误模式**:\n"
            for pattern in self.error_patterns[:2]:  # 最多显示2个模式
                prompt_text += f"- {pattern.description} (出现{pattern.frequency}次)\n"
                prompt_text += f"  建议: {pattern.suggested_fix}\n"
            prompt_text += "\n"
        
        # 添加修复建议
        if self.suggestions:
            prompt_text += "**修复建议**:\n"
            for suggestion in self.suggestions[:3]:  # 最多显示3个建议
                prompt_text += f"- {suggestion}\n"
            prompt_text += "\n"
        
        prompt_text += "请根据上述错误信息进行针对性修正，避免重复相同的错误。\n\n"
        
        return prompt_text


class ErrorCollector:
    """错误信息收集和标准化"""
    
    def __init__(self):
        self.error_patterns = {
            # SQL相关错误模式
            r"no such table": (ErrorCategory.DATABASE, ErrorSeverity.HIGH, "表不存在"),
            r"no such column": (ErrorCategory.DATABASE, ErrorSeverity.HIGH, "列不存在"),
            r"syntax error": (ErrorCategory.SYNTAX, ErrorSeverity.HIGH, "SQL语法错误"),
            r"near.*unexpected": (ErrorCategory.SYNTAX, ErrorSeverity.HIGH, "SQL语法错误"),
            
            # 网络相关错误
            r"connection.*timeout": (ErrorCategory.NETWORK, ErrorSeverity.MEDIUM, "连接超时"),
            r"connection.*refused": (ErrorCategory.NETWORK, ErrorSeverity.HIGH, "连接被拒绝"),
            
            # 依赖相关错误
            r"module.*not found": (ErrorCategory.DEPENDENCY, ErrorSeverity.HIGH, "模块未找到"),
            r"import.*error": (ErrorCategory.DEPENDENCY, ErrorSeverity.HIGH, "导入错误"),
            
            # 运行时错误
            r"division by zero": (ErrorCategory.RUNTIME, ErrorSeverity.MEDIUM, "除零错误"),
            r"index.*out of range": (ErrorCategory.RUNTIME, ErrorSeverity.MEDIUM, "索引越界"),
            r"key.*error": (ErrorCategory.RUNTIME, ErrorSeverity.MEDIUM, "键错误"),
        }
    
    def capture_exception(self, exception: Exception, context: Dict[str, Any] = None) -> ErrorInfo:
        """捕获异常信息"""
        error_type = type(exception).__name__
        error_message = str(exception)
        stack_trace = traceback.format_exc()
        
        # 分析错误类别和严重程度
        category, severity = self._analyze_error(error_message, error_type)
        
        return ErrorInfo(
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack_trace,
            context=context or {},
            severity=severity,
            category=category
        )
    
    def capture_execution_error(self, command: str, output: str, exit_code: int, 
                              context: Dict[str, Any] = None) -> ErrorInfo:
        """捕获执行错误"""
        error_type = f"ExecutionError (exit_code: {exit_code})"
        error_message = output.strip() if output else f"命令执行失败，退出码: {exit_code}"
        
        # 分析错误类别和严重程度
        category, severity = self._analyze_error(error_message, error_type)
        
        # 添加命令信息到上下文
        exec_context = context or {}
        exec_context.update({
            'command': command,
            'exit_code': exit_code,
            'output': output[:200] if output else ""  # 限制输出长度
        })
        
        return ErrorInfo(
            error_type=error_type,
            error_message=error_message,
            context=exec_context,
            severity=severity,
            category=category
        )
    
    def capture_timeout_error(self, operation: str, timeout: float, 
                            context: Dict[str, Any] = None) -> ErrorInfo:
        """捕获超时错误"""
        error_type = "TimeoutError"
        error_message = f"操作 '{operation}' 超时 ({timeout}秒)"
        
        timeout_context = context or {}
        timeout_context.update({
            'operation': operation,
            'timeout': timeout,
            'timestamp': datetime.now().isoformat()
        })
        
        return ErrorInfo(
            error_type=error_type,
            error_message=error_message,
            context=timeout_context,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.TIMEOUT
        )
    
    def capture_sql_error(self, sql: str, error_message: str, 
                         context: Dict[str, Any] = None) -> ErrorInfo:
        """捕获SQL执行错误"""
        error_type = "SQLError"
        
        # 分析SQL错误类别和严重程度
        category, severity = self._analyze_error(error_message, error_type)
        
        sql_context = context or {}
        sql_context.update({
            'sql': sql[:200] if sql else "",  # 限制SQL长度
            'error_source': 'database'
        })
        
        return ErrorInfo(
            error_type=error_type,
            error_message=error_message,
            context=sql_context,
            severity=severity,
            category=category
        )
    
    def _analyze_error(self, error_message: str, error_type: str) -> Tuple[ErrorCategory, ErrorSeverity]:
        """分析错误类别和严重程度"""
        error_text = error_message.lower()
        
        # 使用正则表达式匹配错误模式
        for pattern, (category, severity, _) in self.error_patterns.items():
            if re.search(pattern, error_text):
                return category, severity
        
        # 基于错误类型的默认分类
        if "sql" in error_type.lower() or "database" in error_type.lower():
            return ErrorCategory.DATABASE, ErrorSeverity.HIGH
        elif "timeout" in error_type.lower():
            return ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM
        elif "syntax" in error_type.lower():
            return ErrorCategory.SYNTAX, ErrorSeverity.HIGH
        elif "import" in error_type.lower() or "module" in error_type.lower():
            return ErrorCategory.DEPENDENCY, ErrorSeverity.HIGH
        elif "connection" in error_type.lower() or "network" in error_type.lower():
            return ErrorCategory.NETWORK, ErrorSeverity.MEDIUM
        else:
            return ErrorCategory.RUNTIME, ErrorSeverity.MEDIUM


class ErrorContextManager:
    """管理错误历史和上下文"""
    
    def __init__(self, max_history: int = 10):
        self.error_history: List[ErrorInfo] = []
        self.max_history = max_history
        self.error_collector = ErrorCollector()
    
    def add_error(self, error_info: ErrorInfo) -> None:
        """添加错误信息到历史"""
        # 设置重试次数
        error_info.retry_count = len(self.error_history) + 1
        
        # 添加到历史记录
        self.error_history.append(error_info)
        
        # 保持历史记录在限制范围内
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history:]
    
    def get_retry_context(self, max_errors: int = 3) -> RetryContext:
        """获取重试上下文"""
        recent_errors = self.error_history[-max_errors:] if self.error_history else []
        retry_count = len(self.error_history)  # 总错误数量作为重试次数
        
        # 分析错误模式
        error_patterns = self.analyze_error_patterns()
        
        # 生成修复建议
        suggestions = self._generate_suggestions(recent_errors, error_patterns)
        
        return RetryContext(
            errors=recent_errors,
            retry_count=retry_count,
            error_patterns=error_patterns,
            suggestions=suggestions
        )
    
    def analyze_error_patterns(self) -> List[ErrorPattern]:
        """分析错误模式"""
        if not self.error_history:
            return []
        
        patterns = []
        
        # 统计错误类型频率
        error_type_count = {}
        category_count = {}
        
        for error in self.error_history:
            # 统计错误类型
            error_type_count[error.error_type] = error_type_count.get(error.error_type, 0) + 1
            # 统计错误类别
            category_count[error.category] = category_count.get(error.category, 0) + 1
        
        # 识别重复的错误类型
        for error_type, count in error_type_count.items():
            if count >= 2:  # 出现2次以上认为是模式
                pattern = ErrorPattern(
                    pattern_type="repeated_error_type",
                    frequency=count,
                    description=f"重复出现的 {error_type} 错误",
                    suggested_fix=self._get_fix_suggestion_for_error_type(error_type)
                )
                patterns.append(pattern)
        
        # 识别错误类别模式
        for category, count in category_count.items():
            if count >= 2:
                pattern = ErrorPattern(
                    pattern_type="category_pattern",
                    frequency=count,
                    description=f"频繁的 {category.value} 类错误",
                    suggested_fix=self._get_fix_suggestion_for_category(category)
                )
                patterns.append(pattern)
        
        return patterns[:3]  # 最多返回3个模式
    
    def _generate_suggestions(self, errors: List[ErrorInfo], patterns: List[ErrorPattern]) -> List[str]:
        """生成修复建议"""
        suggestions = []
        
        if not errors:
            return suggestions
        
        # 基于最近的错误生成建议
        latest_error = errors[-1]
        
        if latest_error.category == ErrorCategory.SYNTAX:
            suggestions.append("检查SQL语法，特别注意括号、引号和关键字的正确使用")
        elif latest_error.category == ErrorCategory.DATABASE:
            suggestions.append("验证表名和列名是否存在，检查数据库连接状态")
        elif latest_error.category == ErrorCategory.TIMEOUT:
            suggestions.append("优化查询性能，考虑添加索引或简化查询条件")
        elif latest_error.category == ErrorCategory.NETWORK:
            suggestions.append("检查网络连接和API服务状态")
        
        # 基于错误模式生成建议
        for pattern in patterns:
            if pattern.suggested_fix not in suggestions:
                suggestions.append(pattern.suggested_fix)
        
        # 基于重试次数的建议
        if len(errors) >= 3:
            suggestions.append("考虑简化查询需求或寻求人工协助")
        
        return suggestions[:5]  # 最多返回5个建议
    
    def _get_fix_suggestion_for_error_type(self, error_type: str) -> str:
        """根据错误类型获取修复建议"""
        suggestions = {
            "SQLError": "检查SQL语法和表结构",
            "TimeoutError": "优化查询性能或增加超时时间",
            "ConnectionError": "检查网络连接和服务状态",
            "ImportError": "检查依赖包是否正确安装",
            "KeyError": "验证数据结构和键名",
            "IndexError": "检查数组边界和索引范围"
        }
        return suggestions.get(error_type, "检查错误详情并进行相应修正")
    
    def _get_fix_suggestion_for_category(self, category: ErrorCategory) -> str:
        """根据错误类别获取修复建议"""
        suggestions = {
            ErrorCategory.SYNTAX: "仔细检查代码语法",
            ErrorCategory.DATABASE: "验证数据库结构和连接",
            ErrorCategory.TIMEOUT: "优化性能或调整超时设置",
            ErrorCategory.NETWORK: "检查网络和服务状态",
            ErrorCategory.DEPENDENCY: "确认依赖包安装正确",
            ErrorCategory.RUNTIME: "检查运行时环境和数据"
        }
        return suggestions.get(category, "进行全面的错误排查")
    
    def clear_history(self) -> None:
        """清空错误历史"""
        self.error_history.clear()
    
    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误统计摘要"""
        if not self.error_history:
            return {"total_errors": 0}
        
        summary = {
            "total_errors": len(self.error_history),
            "error_types": {},
            "categories": {},
            "severities": {},
            "latest_error": self.error_history[-1].timestamp.isoformat() if self.error_history else None
        }
        
        for error in self.error_history:
            # 统计错误类型
            summary["error_types"][error.error_type] = summary["error_types"].get(error.error_type, 0) + 1
            # 统计错误类别
            summary["categories"][error.category.value] = summary["categories"].get(error.category.value, 0) + 1
            # 统计严重程度
            summary["severities"][error.severity.value] = summary["severities"].get(error.severity.value, 0) + 1
        
        return summary


class PromptEnhancer:
    """将错误信息集成到prompt中"""
    
    def __init__(self, max_context_length: int = 1000):
        self.max_context_length = max_context_length
    
    def enhance_retry_prompt(self, original_prompt: str, retry_context: RetryContext) -> str:
        """增强重试prompt"""
        if not retry_context.errors:
            return original_prompt
        
        # 格式化错误上下文
        error_context = retry_context.format_for_prompt()
        
        # 如果错误上下文太长，进行摘要
        if len(error_context) > self.max_context_length:
            error_context = self.summarize_long_errors(error_context, self.max_context_length)
        
        # 将错误上下文插入到原始prompt中
        enhanced_prompt = f"{original_prompt}\n\n{error_context}"
        
        return enhanced_prompt
    
    def format_error_context(self, errors: List[ErrorInfo]) -> str:
        """格式化错误上下文"""
        if not errors:
            return ""
        
        context_parts = []
        
        for i, error in enumerate(errors, 1):
            error_section = f"错误 {i}:\n"
            error_section += error.to_context_string()
            context_parts.append(error_section)
        
        return "\n".join(context_parts)
    
    def summarize_long_errors(self, error_text: str, max_length: int) -> str:
        """摘要过长的错误信息"""
        if len(error_text) <= max_length:
            return error_text
        
        # 简单的摘要策略：保留开头和结尾，中间用省略号
        keep_length = max_length // 2 - 50
        
        if keep_length > 0:
            start_part = error_text[:keep_length]
            end_part = error_text[-keep_length:]
            return f"{start_part}\n\n... [错误信息过长，已省略中间部分] ...\n\n{end_part}"
        else:
            # 如果太短，只保留开头
            return error_text[:max_length] + "\n... [已截断]"
    
    def sanitize_sensitive_data(self, text: str) -> str:
        """脱敏处理敏感信息"""
        # 移除可能的敏感信息
        patterns = [
            (r'password["\s]*[:=]["\s]*[^"\s,}]+', 'password="***"'),
            (r'token["\s]*[:=]["\s]*[^"\s,}]+', 'token="***"'),
            (r'key["\s]*[:=]["\s]*[^"\s,}]+', 'key="***"'),
            (r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '****-****-****-****'),  # 信用卡号
            (r'\b\d{3}-\d{2}-\d{4}\b', '***-**-****'),  # SSN格式
        ]
        
        sanitized_text = text
        for pattern, replacement in patterns:
            sanitized_text = re.sub(pattern, replacement, sanitized_text, flags=re.IGNORECASE)
        
        return sanitized_text
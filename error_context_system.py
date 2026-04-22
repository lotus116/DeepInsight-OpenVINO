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


class SQLErrorSubType(Enum):
    """
    SQL 错误子类型枚举
    
    用于 Self-Healing 机制中的针对性修复建议生成
    """
    UNKNOWN_COLUMN = "unknown_column"      # 列名错误
    UNKNOWN_TABLE = "unknown_table"        # 表名错误
    SYNTAX_ERROR = "syntax_error"          # 语法错误
    AMBIGUOUS_COLUMN = "ambiguous_column"  # 歧义列
    JOIN_ERROR = "join_error"              # JOIN 错误
    AGGREGATION_ERROR = "aggregation_error" # 聚合函数错误
    TYPE_ERROR = "type_error"              # 类型不匹配
    OTHER = "other"


# SQL 错误修复建议映射
SQL_FIX_SUGGESTIONS = {
    SQLErrorSubType.UNKNOWN_COLUMN: "请检查列名拼写，参考 Schema 中的正确列名，注意大小写",
    SQLErrorSubType.UNKNOWN_TABLE: "请检查表名拼写，参考 Schema 中的正确表名",
    SQLErrorSubType.SYNTAX_ERROR: "请检查 SQL 语法，确保关键字、括号、逗号正确",
    SQLErrorSubType.AMBIGUOUS_COLUMN: "请为歧义列添加表名前缀，如 table.column",
    SQLErrorSubType.JOIN_ERROR: "请检查 JOIN 条件，确保关联字段类型和名称正确匹配",
    SQLErrorSubType.AGGREGATION_ERROR: "请检查 GROUP BY 是否包含所有非聚合列",
    SQLErrorSubType.TYPE_ERROR: "请检查数据类型，字符串需加引号，日期格式需符合数据库要求",
    SQLErrorSubType.OTHER: "请仔细检查 SQL 语法和表结构",
}


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
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，用于序列化"""
        return {
            'error_type': self.error_type,
            'message': self.error_message,
            'category': self.category.value if isinstance(self.category, ErrorCategory) else str(self.category),
            'severity': self.severity.value if isinstance(self.severity, ErrorSeverity) else str(self.severity),
            'retry_count': self.retry_count,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


@dataclass 
class ErrorPattern:
    """错误模式识别结果"""
    pattern_type: str      # 模式类型
    frequency: int         # 出现频率
    description: str       # 模式描述
    suggested_fix: str     # 建议修复方法
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'pattern_type': self.pattern_type,
            'frequency': self.frequency,
            'description': self.description,
            'suggested_fix': self.suggested_fix
        }


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
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，用于序列化传递给 prompt_template_system"""
        return {
            'errors': [e.to_dict() for e in self.errors] if self.errors else [],
            'retry_count': self.retry_count,
            'error_patterns': [p.to_dict() for p in self.error_patterns] if self.error_patterns else [],
            'suggestions': self.suggestions if self.suggestions else []
        }


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
    
    def clear(self) -> None:
        """清空错误历史，用于评测时确保每个 case 独立运行"""
        self.error_history.clear()
    
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
        """增强重试prompt，添加针对性修复建议"""
        if not retry_context.errors:
            return original_prompt
        
        # 构建增强的错误上下文（包含针对性建议）
        error_section = "\n\n⚠️ 之前的尝试失败，请避免以下错误：\n"
        
        for error in retry_context.errors[:3]:  # 最多显示3个错误
            # 分类 SQL 错误
            sub_type = self._classify_sql_error(error.error_message)
            fix_suggestion = SQL_FIX_SUGGESTIONS.get(sub_type, SQL_FIX_SUGGESTIONS[SQLErrorSubType.OTHER])
            
            # 截取错误消息前100字符
            short_msg = error.error_message[:100] + "..." if len(error.error_message) > 100 else error.error_message
            
            error_section += f"- 错误: {short_msg}\n"
            error_section += f"  💡 建议: {fix_suggestion}\n"
        
        # 如果有通用建议，也添加进来
        if retry_context.suggestions:
            error_section += "\n🔧 通用建议:\n"
            for suggestion in retry_context.suggestions[:2]:
                error_section += f"  - {suggestion}\n"
        
        # 检查长度限制
        if len(error_section) > self.max_context_length:
            error_section = self.summarize_long_errors(error_section, self.max_context_length)
        
        return original_prompt + error_section
    
    def _classify_sql_error(self, error_msg: str) -> SQLErrorSubType:
        """
        根据错误信息分类 SQL 错误
        
        支持 MySQL, SQLite, PostgreSQL 等常见错误模式
        """
        error_lower = error_msg.lower()
        
        # 列名错误
        if any(kw in error_lower for kw in ["unknown column", "no such column", "column not found", "invalid column"]):
            return SQLErrorSubType.UNKNOWN_COLUMN
        
        # 表名错误
        if any(kw in error_lower for kw in ["unknown table", "no such table", "table not found", "doesn't exist"]):
            return SQLErrorSubType.UNKNOWN_TABLE
        
        # 语法错误
        if any(kw in error_lower for kw in ["syntax error", "syntax", "parse error", "unexpected"]):
            return SQLErrorSubType.SYNTAX_ERROR
        
        # 歧义列
        if "ambiguous" in error_lower:
            return SQLErrorSubType.AMBIGUOUS_COLUMN
        
        # JOIN 错误
        if "join" in error_lower:
            return SQLErrorSubType.JOIN_ERROR
        
        # 聚合错误
        if any(kw in error_lower for kw in ["group by", "aggregate", "not in group by", "full-group-by"]):
            return SQLErrorSubType.AGGREGATION_ERROR
        
        # 类型错误
        if any(kw in error_lower for kw in ["type", "convert", "cast", "truncated"]):
            return SQLErrorSubType.TYPE_ERROR
        
        return SQLErrorSubType.OTHER
    
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


class SQLPreValidator:
    """
    SQL 预验证器 - 在执行前检测语法问题
    
    集成到 Self-Healing 机制中，在 SQL 执行前进行语法检查，
    如果发现可修复的问题（如 CTE 不完整），则尝试自动修复。
    
    功能:
    1. CTE 结构完整性检测和修复
    2. 括号匹配检查
    3. 基本语法验证 (使用 sqlparse)
    """
    
    def __init__(self):
        self._sqlparse_available = False
        try:
            import sqlparse
            self._sqlparse_available = True
        except ImportError:
            pass
    
    def validate(self, sql: str) -> tuple:
        """
        验证 SQL 语句并尝试修复常见问题
        
        Args:
            sql: 原始 SQL 语句
            
        Returns:
            tuple: (is_valid, error_message, fixed_sql_or_none)
                - is_valid: 是否有效
                - error_message: 错误描述（如果无效）
                - fixed_sql_or_none: 修复后的 SQL（如果能修复）
        """
        if not sql or not sql.strip():
            return (False, "SQL 为空", None)
        
        sql = sql.strip()
        
        # Step 1: CTE 结构检测和修复
        cte_result = self._check_and_fix_cte(sql)
        if cte_result[0]:  # 如果进行了修复
            sql = cte_result[1]
        
        # Step 2: 括号匹配检查
        bracket_ok, bracket_error = self._check_brackets(sql)
        if not bracket_ok:
            return (False, bracket_error, None)
        
        # Step 3: sqlparse 语法验证
        if self._sqlparse_available:
            syntax_ok, syntax_error = self._validate_with_sqlparse(sql)
            if not syntax_ok:
                return (False, syntax_error, None)
        
        # Step 4: 基本关键字检查
        keyword_ok, keyword_error = self._check_basic_keywords(sql)
        if not keyword_ok:
            return (False, keyword_error, None)
        
        return (True, "", sql if cte_result[0] else None)
    
    def _check_and_fix_cte(self, sql: str) -> tuple:
        """
        检测并修复不完整的 CTE (Common Table Expression)
        
        常见问题: LLM 生成的 CTE 缺少 WITH 关键字前缀，或者第一个 CTE 名称丢失
        例如: "SELECT ... ), CTE2 AS (..." 应该是 "WITH CTE1 AS (SELECT ...), CTE2 AS (..."
        
        Returns:
            tuple: (was_fixed, fixed_sql)
        """
        sql_upper = sql.upper().strip()
        
        # 如果已经以 WITH 开头，无需修复
        if sql_upper.startswith('WITH'):
            return (False, sql)
        
        # 模式 1: 直接以 "CTE_name AS (" 开头（缺少 WITH）
        cte_pattern = r'^(\w+)\s+AS\s*\('
        match = re.match(cte_pattern, sql.strip(), re.IGNORECASE)
        if match:
            fixed_sql = f"WITH {sql}"
            return (True, fixed_sql)
        
        # 模式 2: 以 SELECT 开头但中间有 "), CTE_name AS (" 模式
        # 这表明 LLM 生成了 CTE 但:
        # - 丢失了 "WITH FirstCTE AS (" 前缀
        # - 第一个 SELECT 实际上是第一个 CTE 的内容
        if sql_upper.startswith('SELECT'):
            # 检测中间的 CTE 模式: "), xxx AS (" 或 "), xxx AS ("
            dangling_cte = r'\)\s*,\s*(\w+)\s+AS\s*\('
            match = re.search(dangling_cte, sql, re.IGNORECASE)
            if match:
                # 找到了第二个 CTE，说明第一个 CTE 名称丢失了
                # 尝试从后面的引用中推断第一个 CTE 的名称
                # 通常第一个 CTE 会被后面的 CTE 或 SELECT 引用
                
                # 查找所有 CTE 定义及其名称
                cte_names = re.findall(r'\)\s*,\s*(\w+)\s+AS\s*\(', sql, re.IGNORECASE)
                
                # 查找 FROM 子句中引用的表名（可能是 CTE 名）
                from_refs = re.findall(r'FROM\s+(\w+)', sql, re.IGNORECASE)
                
                # 尝试推断第一个 CTE 的名称
                first_cte_name = None
                for ref in from_refs:
                    if ref.upper() not in [c.upper() for c in cte_names]:
                        # 这可能是第一个 CTE 的名称
                        # 排除常见的真实表名前缀
                        if not any(prefix in ref.lower() for prefix in ['product', 'sales', 'customer', 'order', 'vendor']):
                            first_cte_name = ref
                            break
                
                if not first_cte_name:
                    # 使用通用名称
                    first_cte_name = "CTE_Main"
                
                fixed_sql = f"WITH {first_cte_name} AS ({sql}"
                return (True, fixed_sql)
        
        return (False, sql)
    
    def _check_brackets(self, sql: str) -> tuple:
        """
        检查括号匹配
        
        Returns:
            tuple: (is_ok, error_message)
        """
        # 移除字符串内容以避免误判
        cleaned = re.sub(r"'[^']*'", '', sql)
        cleaned = re.sub(r'"[^"]*"', '', cleaned)
        
        count_paren = 0
        count_square = 0
        
        for char in cleaned:
            if char == '(':
                count_paren += 1
            elif char == ')':
                count_paren -= 1
            elif char == '[':
                count_square += 1
            elif char == ']':
                count_square -= 1
            
            # 检测到右括号多于左括号
            if count_paren < 0:
                return (False, "括号不匹配: 右括号多于左括号")
            if count_square < 0:
                return (False, "方括号不匹配: 右括号多于左括号")
        
        if count_paren != 0:
            return (False, f"括号不匹配: {'缺少右括号' if count_paren > 0 else '多余右括号'}")
        if count_square != 0:
            return (False, f"方括号不匹配: {'缺少右方括号' if count_square > 0 else '多余右方括号'}")
        
        return (True, "")
    
    def _validate_with_sqlparse(self, sql: str) -> tuple:
        """
        使用 sqlparse 进行语法验证
        
        Returns:
            tuple: (is_ok, error_message)
        """
        try:
            import sqlparse
            
            parsed = sqlparse.parse(sql)
            if not parsed or not parsed[0].tokens:
                return (False, "sqlparse 无法解析 SQL")
            
            # 检查第一个 token 是否是有效的 SQL 关键字
            first_token = parsed[0].tokens[0]
            first_word = str(first_token).strip().upper()
            
            valid_starts = ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER']
            
            # 跳过空白和注释
            for token in parsed[0].tokens:
                if token.is_whitespace or str(token).strip().startswith('--'):
                    continue
                first_word = str(token).strip().upper().split()[0] if str(token).strip() else ""
                break
            
            if first_word and not any(first_word.startswith(kw) for kw in valid_starts):
                return (False, f"SQL 不以有效关键字开头: {first_word[:20]}")
            
            return (True, "")
            
        except Exception as e:
            # sqlparse 解析失败可能表示语法问题
            return (False, f"SQL 解析异常: {str(e)[:100]}")
    
    def _check_basic_keywords(self, sql: str) -> tuple:
        """
        检查基本 SQL 关键字
        
        Returns:
            tuple: (is_ok, error_message)
        """
        sql_upper = sql.upper()
        
        # 如果以 WITH 开头，这是 CTE，跳过 FROM 检查（CTE 的 SELECT 在内部）
        if sql_upper.strip().startswith('WITH'):
            return (True, "")
        
        # SELECT 语句必须包含 FROM（除非是 SELECT 常量）
        if 'SELECT' in sql_upper:
            # 简单 SELECT 常量检测 (如 SELECT 1, SELECT 'hello', SELECT COUNT(*) 等)
            simple_select = re.match(r"^\s*SELECT\s+[\d'\"\w\(\)\*\s,]+\s*$", sql, re.IGNORECASE)
            if not simple_select and 'FROM' not in sql_upper:
                return (False, "SELECT 语句缺少 FROM 子句")
        
        return (True, "")
    
    def fix_and_format(self, sql: str) -> str:
        """
        尝试修复并格式化 SQL
        
        Args:
            sql: 原始 SQL
            
        Returns:
            修复和格式化后的 SQL（如果无法修复或格式化，返回原始 SQL）
        """
        if not sql or not sql.strip():
            return sql
        
        original_sql = sql
        
        # 尝试验证和修复
        is_valid, error, fixed = self.validate(sql)
        
        if fixed:
            sql = fixed
        elif not is_valid:
            # 如果验证失败且无法自动修复，返回原始 SQL
            # 让后续的数据库执行来报告具体错误
            return original_sql
        
        # 格式化
        if self._sqlparse_available:
            try:
                import sqlparse
                formatted = sqlparse.format(
                    sql,
                    strip_comments=True,
                    reindent=False,
                    keyword_case='upper'
                ).strip()
                # 只有格式化成功且非空才使用
                if formatted:
                    sql = formatted
            except Exception:
                pass
        
        return sql
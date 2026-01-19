"""
核心数据模型定义

包含Agent上下文记忆系统的所有数据结构和类型定义。
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import uuid


class ContextType(Enum):
    """上下文项类型枚举"""
    USER_INPUT = "user_input"
    AGENT_RESPONSE = "agent_response"
    CODE_SNIPPET = "code_snippet"
    ERROR_INFO = "error_info"
    FILE_REFERENCE = "file_reference"
    SYSTEM_INFO = "system_info"


class SectionType(Enum):
    """提示部分类型枚举"""
    USER_INPUT = "user_input"
    HISTORICAL_CONTEXT = "historical_context"
    SYSTEM_INSTRUCTION = "system_instruction"
    CODE_CONTEXT = "code_context"
    ERROR_CONTEXT = "error_context"


@dataclass
class ContextItem:
    """上下文项数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    type: ContextType = ContextType.USER_INPUT
    timestamp: datetime = field(default_factory=datetime.now)
    relevance_score: float = 0.0
    source_interaction_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """确保时间戳是datetime对象"""
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)


@dataclass
class Interaction:
    """交互记录数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    user_input: str = ""
    agent_response: str = ""
    context_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """确保时间戳是datetime对象"""
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)


@dataclass
class ContextConfig:
    """上下文配置数据结构"""
    max_context_length: int = 4000
    relevance_threshold: float = 0.3
    max_history_items: int = 20
    enable_topic_detection: bool = True
    context_retention_days: int = 30
    token_limit: int = 8000
    enable_semantic_search: bool = False
    debug_mode: bool = False
    
    def validate(self) -> bool:
        """验证配置参数的有效性"""
        if self.max_context_length <= 0:
            return False
        if not 0.0 <= self.relevance_threshold <= 1.0:
            return False
        if self.max_history_items <= 0:
            return False
        if self.context_retention_days <= 0:
            return False
        if self.token_limit <= 0:
            return False
        return True


@dataclass
class SessionContext:
    """会话上下文数据结构"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    interaction_count: int = 0
    current_topic: Optional[str] = None
    active_files: List[str] = field(default_factory=list)
    configuration: ContextConfig = field(default_factory=ContextConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """确保时间戳是datetime对象"""
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.last_updated, str):
            self.last_updated = datetime.fromisoformat(self.last_updated)


@dataclass
class StorageStats:
    """存储统计信息数据结构"""
    total_sessions: int = 0
    total_interactions: int = 0
    total_context_items: int = 0
    storage_size_mb: float = 0.0
    avg_retrieval_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    last_cleanup: Optional[datetime] = None
    
    def __post_init__(self):
        """确保时间戳是datetime对象"""
        if isinstance(self.last_cleanup, str):
            self.last_cleanup = datetime.fromisoformat(self.last_cleanup)


@dataclass
class FilterStrategy:
    """过滤策略数据结构"""
    strategy_name: str = "default"
    use_semantic_similarity: bool = False
    use_temporal_decay: bool = True
    use_topic_relevance: bool = True
    temporal_decay_factor: float = 0.9
    topic_weight: float = 0.4
    recency_weight: float = 0.3
    similarity_weight: float = 0.3
    
    def validate(self) -> bool:
        """验证策略参数的有效性"""
        total_weight = self.topic_weight + self.recency_weight + self.similarity_weight
        return abs(total_weight - 1.0) < 0.01  # 允许小的浮点误差


# 异常类定义
class ContextMemoryError(Exception):
    """上下文记忆系统基础异常"""
    pass


class StorageError(ContextMemoryError):
    """存储相关异常"""
    pass


class ContextError(ContextMemoryError):
    """上下文处理相关异常"""
    pass


class PerformanceIssue(ContextMemoryError):
    """性能问题异常"""
    pass


class ConfigurationError(ContextMemoryError):
    """配置错误异常"""
    pass
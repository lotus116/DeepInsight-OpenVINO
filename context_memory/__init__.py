"""
Agent上下文记忆系统

提供智能的对话上下文管理，包括：
- 会话级别的对话历史记忆
- 智能上下文选择和过滤
- 结构化的提示构建
- 高效的存储和检索
"""

__version__ = "1.0.0"

from .models import (
    Interaction,
    ContextItem,
    SessionContext,
    ContextConfig,
    ContextType,
    SectionType
)

from .models import (
    Interaction,
    ContextItem,
    SessionContext,
    ContextConfig,
    ContextType,
    SectionType
)

from .context_manager import ContextManager
from .memory_store import MemoryStore
from .context_filter import ContextFilter
from .prompt_builder import PromptBuilder

__all__ = [
    "Interaction",
    "ContextItem", 
    "SessionContext",
    "ContextConfig",
    "ContextType",
    "SectionType",
    "ContextManager",
    "MemoryStore",
    "ContextFilter",
    "PromptBuilder"
]
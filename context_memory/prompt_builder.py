"""
提示构建器实现

构建包含上下文的完整、结构化提示。
"""

import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .models import (
    ContextItem, SectionType, ContextType,
    ContextError
)


@dataclass
class PromptSection:
    """提示部分数据结构"""
    type: SectionType
    title: str
    content: str
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PromptBuilder:
    """提示构建器类 - 负责构建结构化的提示"""
    
    def __init__(self, max_tokens: int = 8000):
        """
        初始化提示构建器
        
        Args:
            max_tokens: 最大token限制
        """
        self.max_tokens = max_tokens
        self.token_per_char_ratio = 0.25  # 估算：平均4个字符约等于1个token
    
    def build_contextual_prompt(
        self, 
        user_input: str, 
        context: List[ContextItem],
        system_instruction: Optional[str] = None
    ) -> str:
        """
        构建包含上下文的完整提示
        
        Args:
            user_input: 用户输入
            context: 上下文项列表
            system_instruction: 系统指令
            
        Returns:
            结构化的完整提示
        """
        try:
            sections = []
            
            # 1. 添加系统指令部分
            if system_instruction:
                sections.append(PromptSection(
                    type=SectionType.SYSTEM_INSTRUCTION,
                    title="系统指令",
                    content=system_instruction
                ))
            
            # 2. 添加历史上下文部分
            if context:
                context_content = self.format_context_section(context)
                sections.append(PromptSection(
                    type=SectionType.HISTORICAL_CONTEXT,
                    title="历史上下文",
                    content=context_content
                ))
            
            # 3. 添加用户输入部分
            sections.append(PromptSection(
                type=SectionType.USER_INPUT,
                title="当前问题",
                content=user_input
            ))
            
            # 4. 构建最终提示
            prompt = self._build_prompt_from_sections(sections)
            
            # 5. 确保不超过token限制
            prompt = self.ensure_token_limits(prompt, self.max_tokens)
            
            return prompt
            
        except Exception as e:
            raise ContextError(f"构建上下文提示失败: {e}")
    
    def format_context_section(self, context: List[ContextItem]) -> str:
        """
        格式化上下文部分
        
        Args:
            context: 上下文项列表
            
        Returns:
            格式化的上下文字符串
        """
        if not context:
            return ""
        
        formatted_items = []
        
        # 按类型分组上下文项
        grouped_context = self._group_context_by_type(context)
        
        for context_type, items in grouped_context.items():
            type_title = self._get_context_type_title(context_type)
            
            if items:
                formatted_items.append(f"\n### {type_title}\n")
                
                for item in items:
                    formatted_item = self._format_single_context_item(item)
                    formatted_items.append(formatted_item)
        
        return "".join(formatted_items)
    
    def ensure_token_limits(self, prompt: str, max_tokens: int) -> str:
        """
        确保提示不超过token限制
        
        Args:
            prompt: 原始提示
            max_tokens: 最大token数
            
        Returns:
            截断后的提示
        """
        estimated_tokens = len(prompt) * self.token_per_char_ratio
        
        if estimated_tokens <= max_tokens:
            return prompt
        
        # 需要截断，优先保留用户输入和系统指令
        sections = self._parse_prompt_sections(prompt)
        
        # 计算必需部分的token数
        essential_tokens = 0
        essential_sections = []
        context_sections = []
        
        for section in sections:
            section_tokens = len(section.content) * self.token_per_char_ratio
            
            if section.type in [SectionType.USER_INPUT, SectionType.SYSTEM_INSTRUCTION]:
                essential_tokens += section_tokens
                essential_sections.append(section)
            else:
                context_sections.append((section, section_tokens))
        
        # 计算可用于上下文的token数
        available_tokens = max_tokens - essential_tokens - 100  # 保留100个token的缓冲
        
        if available_tokens <= 0:
            # 如果必需部分已经超出限制，只保留用户输入
            user_input_section = next(
                (s for s in essential_sections if s.type == SectionType.USER_INPUT), 
                None
            )
            if user_input_section:
                return self._build_prompt_from_sections([user_input_section])
            return prompt[:int(max_tokens / self.token_per_char_ratio)]
        
        # 选择最重要的上下文项
        selected_context_sections = self._select_context_within_limit(
            context_sections, available_tokens
        )
        
        # 重新构建提示
        final_sections = essential_sections + selected_context_sections
        return self._build_prompt_from_sections(final_sections)
    
    def add_section_markers(self, content: str, section_type: SectionType) -> str:
        """
        为内容添加部分标记
        
        Args:
            content: 内容
            section_type: 部分类型
            
        Returns:
            带标记的内容
        """
        # 使用不可见的XML风格标记，避免在UI中显示
        markers = {
            SectionType.USER_INPUT: ("<!-- USER_INPUT -->", "<!-- /USER_INPUT -->"),
            SectionType.HISTORICAL_CONTEXT: ("<!-- HISTORICAL_CONTEXT -->", "<!-- /HISTORICAL_CONTEXT -->"),
            SectionType.SYSTEM_INSTRUCTION: ("<!-- SYSTEM_INSTRUCTION -->", "<!-- /SYSTEM_INSTRUCTION -->"),
            SectionType.CODE_CONTEXT: ("<!-- CODE_CONTEXT -->", "<!-- /CODE_CONTEXT -->"),
            SectionType.ERROR_CONTEXT: ("<!-- ERROR_CONTEXT -->", "<!-- /ERROR_CONTEXT -->")
        }
        
        start_marker, end_marker = markers.get(section_type, ("<!-- CONTENT -->", "<!-- /CONTENT -->"))
        
        return f"{start_marker}\n{content}\n{end_marker}"
    
    def _build_prompt_from_sections(self, sections: List[PromptSection]) -> str:
        """从部分列表构建提示"""
        prompt_parts = []
        
        for section in sections:
            marked_content = self.add_section_markers(section.content, section.type)
            prompt_parts.append(marked_content)
        
        return "\n".join(prompt_parts)
    
    def _group_context_by_type(self, context: List[ContextItem]) -> Dict[ContextType, List[ContextItem]]:
        """按类型分组上下文项"""
        grouped = {}
        
        for item in context:
            if item.type not in grouped:
                grouped[item.type] = []
            grouped[item.type].append(item)
        
        # 按时间排序每个组
        for context_type in grouped:
            grouped[context_type].sort(key=lambda x: x.timestamp, reverse=True)
        
        return grouped
    
    def _get_context_type_title(self, context_type: ContextType) -> str:
        """获取上下文类型的标题"""
        titles = {
            ContextType.USER_INPUT: "用户输入历史",
            ContextType.AGENT_RESPONSE: "助手响应历史", 
            ContextType.CODE_SNIPPET: "相关代码",
            ContextType.ERROR_INFO: "错误信息",
            ContextType.FILE_REFERENCE: "文件引用",
            ContextType.SYSTEM_INFO: "系统信息"
        }
        
        return titles.get(context_type, "其他信息")
    
    def _format_single_context_item(self, item: ContextItem) -> str:
        """格式化单个上下文项"""
        timestamp_str = item.timestamp.strftime("%H:%M:%S")
        
        # 根据类型选择不同的格式
        if item.type == ContextType.CODE_SNIPPET:
            return f"\n```\n{item.content}\n```\n*时间: {timestamp_str}*\n"
        elif item.type == ContextType.ERROR_INFO:
            return f"\n❌ **错误**: {item.content}\n*时间: {timestamp_str}*\n"
        else:
            return f"\n- {item.content}\n  *时间: {timestamp_str}*\n"
    
    def _parse_prompt_sections(self, prompt: str) -> List[PromptSection]:
        """解析提示中的部分"""
        sections = []
        
        # 简单的部分解析（基于标记）
        section_patterns = {
            r'🔵 用户输入\n(.*?)\n(?=🔵|📚|⚙️|💻|❌|$)': SectionType.USER_INPUT,
            r'📚 历史上下文\n(.*?)\n(?=🔵|📚|⚙️|💻|❌|$)': SectionType.HISTORICAL_CONTEXT,
            r'⚙️ 系统指令\n(.*?)\n(?=🔵|📚|⚙️|💻|❌|$)': SectionType.SYSTEM_INSTRUCTION,
            r'💻 代码上下文\n(.*?)\n(?=🔵|📚|⚙️|💻|❌|$)': SectionType.CODE_CONTEXT,
            r'❌ 错误上下文\n(.*?)\n(?=🔵|📚|⚙️|💻|❌|$)': SectionType.ERROR_CONTEXT
        }
        
        for pattern, section_type in section_patterns.items():
            matches = re.findall(pattern, prompt, re.DOTALL)
            for match in matches:
                sections.append(PromptSection(
                    type=section_type,
                    title=section_type.value,
                    content=match.strip()
                ))
        
        return sections
    
    def _select_context_within_limit(
        self, 
        context_sections: List[Tuple[PromptSection, float]], 
        available_tokens: float
    ) -> List[PromptSection]:
        """在token限制内选择上下文部分"""
        selected = []
        used_tokens = 0.0
        
        # 按重要性排序（这里简单按类型优先级）
        type_priority = {
            SectionType.CODE_CONTEXT: 3,
            SectionType.ERROR_CONTEXT: 2,
            SectionType.HISTORICAL_CONTEXT: 1
        }
        
        context_sections.sort(
            key=lambda x: type_priority.get(x[0].type, 0), 
            reverse=True
        )
        
        for section, tokens in context_sections:
            if used_tokens + tokens <= available_tokens:
                selected.append(section)
                used_tokens += tokens
            else:
                # 尝试截断这个部分
                remaining_tokens = available_tokens - used_tokens
                if remaining_tokens > 50:  # 至少保留50个token的内容
                    max_chars = int(remaining_tokens / self.token_per_char_ratio)
                    truncated_content = section.content[:max_chars] + "..."
                    truncated_section = PromptSection(
                        type=section.type,
                        title=section.title,
                        content=truncated_content
                    )
                    selected.append(truncated_section)
                break
        
        return selected
    
    def get_prompt_stats(self, prompt: str) -> Dict[str, any]:
        """
        获取提示统计信息
        
        Args:
            prompt: 提示字符串
            
        Returns:
            统计信息字典
        """
        return {
            "character_count": len(prompt),
            "estimated_tokens": int(len(prompt) * self.token_per_char_ratio),
            "line_count": len(prompt.split('\n')),
            "section_count": len(self._parse_prompt_sections(prompt)),
            "within_limit": len(prompt) * self.token_per_char_ratio <= self.max_tokens
        }
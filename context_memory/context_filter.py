"""
上下文过滤器实现

智能选择与当前问题最相关的历史上下文。
"""

import re
import math
from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional, Tuple
from collections import Counter

from .models import (
    Interaction, ContextItem, FilterStrategy,
    ContextType, ContextError
)


class ContextFilter:
    """上下文过滤器类 - 负责智能选择相关上下文"""
    
    def __init__(self, strategy: Optional[FilterStrategy] = None):
        """
        初始化上下文过滤器
        
        Args:
            strategy: 过滤策略，如果为None则使用默认策略
        """
        self.strategy = strategy or FilterStrategy()
        self._topic_keywords_cache: Dict[str, Set[str]] = {}
    
    def select_relevant_context(
        self, 
        current_input: str, 
        history: List[Interaction],
        max_items: int = 10
    ) -> List[ContextItem]:
        """
        选择与当前输入最相关的上下文
        
        Args:
            current_input: 当前用户输入
            history: 历史交互记录
            max_items: 最大返回项目数
            
        Returns:
            按相关性排序的上下文项列表
        """
        if not history:
            return []
        
        try:
            # 从历史记录中提取上下文项
            context_items = self._extract_context_items_from_history(history)
            
            if not context_items:
                return []
            
            # 计算每个上下文项的相关性分数
            scored_items = []
            current_keywords = self._extract_keywords(current_input)
            
            for item in context_items:
                score = self.calculate_relevance_score(item, current_input, current_keywords)
                if score >= self.strategy.topic_weight * 0.1:  # 最低阈值
                    scored_items.append((item, score))
            
            # 按分数排序并返回前N项
            scored_items.sort(key=lambda x: x[1], reverse=True)
            return [item for item, score in scored_items[:max_items]]
            
        except Exception as e:
            raise ContextError(f"选择相关上下文失败: {e}")
    
    def calculate_relevance_score(
        self, 
        context: ContextItem, 
        query: str,
        query_keywords: Optional[Set[str]] = None
    ) -> float:
        """
        计算上下文项与查询的相关性分数
        
        Args:
            context: 上下文项
            query: 查询字符串
            query_keywords: 预计算的查询关键词
            
        Returns:
            相关性分数 (0.0 - 1.0)
        """
        try:
            if query_keywords is None:
                query_keywords = self._extract_keywords(query)
            
            # 1. 计算话题相关性
            topic_score = self._calculate_topic_relevance(context, query_keywords)
            
            # 2. 计算时间衰减分数
            recency_score = self._calculate_recency_score(context.timestamp)
            
            # 3. 计算内容相似度
            similarity_score = self._calculate_content_similarity(context.content, query)
            
            # 4. 根据上下文类型调整权重
            type_weight = self._get_type_weight(context.type)
            
            # 5. 综合计算最终分数
            final_score = (
                self.strategy.topic_weight * topic_score +
                self.strategy.recency_weight * recency_score +
                self.strategy.similarity_weight * similarity_score
            ) * type_weight
            
            return min(1.0, max(0.0, final_score))
            
        except Exception as e:
            # 如果计算失败，返回基础分数
            return 0.1
    
    def detect_topic_change(
        self, 
        current_input: str, 
        recent_context: List[ContextItem],
        threshold: float = 0.3
    ) -> bool:
        """
        检测话题是否发生变化
        
        Args:
            current_input: 当前输入
            recent_context: 最近的上下文项
            threshold: 话题变化阈值
            
        Returns:
            是否发生话题变化
        """
        if not recent_context:
            return True
        
        try:
            current_keywords = self._extract_keywords(current_input)
            
            # 计算与最近上下文的平均相关性
            total_relevance = 0.0
            valid_items = 0
            
            for context_item in recent_context[-5:]:  # 只考虑最近5个项目
                relevance = self._calculate_topic_relevance(context_item, current_keywords)
                if relevance > 0:
                    total_relevance += relevance
                    valid_items += 1
            
            if valid_items == 0:
                return True
            
            avg_relevance = total_relevance / valid_items
            return avg_relevance < threshold
            
        except Exception:
            return False
    
    def apply_filtering_strategy(self, strategy: FilterStrategy) -> None:
        """
        应用新的过滤策略
        
        Args:
            strategy: 新的过滤策略
        """
        if not strategy.validate():
            raise ContextError("无效的过滤策略：权重总和必须为1.0")
        
        self.strategy = strategy
        # 清空缓存以应用新策略
        self._topic_keywords_cache.clear()
    
    def _extract_context_items_from_history(self, history: List[Interaction]) -> List[ContextItem]:
        """从历史记录中提取上下文项"""
        context_items = []
        
        for interaction in history:
            # 从用户输入创建上下文项
            if interaction.user_input.strip():
                user_context = ContextItem(
                    content=interaction.user_input,
                    type=ContextType.USER_INPUT,
                    timestamp=interaction.timestamp,
                    source_interaction_id=interaction.id,
                    metadata=interaction.metadata.copy()
                )
                context_items.append(user_context)
            
            # 从Agent响应创建上下文项
            if interaction.agent_response.strip():
                agent_context = ContextItem(
                    content=interaction.agent_response,
                    type=ContextType.AGENT_RESPONSE,
                    timestamp=interaction.timestamp,
                    source_interaction_id=interaction.id,
                    metadata=interaction.metadata.copy()
                )
                context_items.append(agent_context)
            
            # 检查是否包含代码片段
            code_snippets = self._extract_code_snippets(interaction.user_input)
            for snippet in code_snippets:
                code_context = ContextItem(
                    content=snippet,
                    type=ContextType.CODE_SNIPPET,
                    timestamp=interaction.timestamp,
                    source_interaction_id=interaction.id,
                    metadata={"extracted_from": "user_input"}
                )
                context_items.append(code_context)
        
        return context_items
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """提取文本关键词"""
        if not text:
            return set()
        
        # 移除标点符号并转为小写，支持中英文
        import re
        
        # 先处理英文单词
        english_words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # 处理中文，简单按字符分割（实际应用中可能需要更复杂的分词）
        # 移除标点符号，保留中文字符
        chinese_text = re.sub(r'[^\u4e00-\u9fff]', ' ', text)
        # 简单的中文分词：按常见词汇模式分割
        chinese_words = []
        for word in ['代码', '调试', '问题', '函数', '定义', '变量', '赋值', 'python', 'Python']:
            if word.lower() in text.lower():
                chinese_words.append(word.lower())
        
        # 合并所有词汇
        all_words = english_words + chinese_words
        
        # 停用词列表
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'can', 'may', 'might', 'must', 'shall', 'this', 'that', 'these', 'those',
            '的', '了', '在', '是', '我', '你', '他', '她', '它', '我们', '你们', '他们',
            '这', '那', '这个', '那个', '什么', '怎么', '为什么', '哪里', '什么时候',
            '如何', '哪个', '多少', '怎样', '为了', '因为', '所以', '但是', '然后',
            '现在', '刚才', '之前', '以后', '已经', '正在', '将要', '一个', '一些'
        }
        
        # 过滤停用词和短词
        keywords = {word for word in all_words if len(word) > 1 and word not in stop_words}
        
        return keywords
    
    def _calculate_topic_relevance(self, context: ContextItem, query_keywords: Set[str]) -> float:
        """计算话题相关性"""
        if not query_keywords:
            return 0.0
        
        context_keywords = self._extract_keywords(context.content)
        if not context_keywords:
            return 0.0
        
        # 计算关键词重叠度
        intersection = query_keywords.intersection(context_keywords)
        union = query_keywords.union(context_keywords)
        
        if not union:
            return 0.0
        
        # Jaccard相似度
        jaccard_similarity = len(intersection) / len(union)
        
        # 考虑关键词频率
        context_word_freq = Counter(self._extract_keywords(context.content))
        query_word_freq = Counter(query_keywords)
        
        # 计算加权相似度
        weighted_score = 0.0
        total_weight = 0.0
        
        for word in intersection:
            weight = min(context_word_freq[word], query_word_freq[word])
            weighted_score += weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_similarity = weighted_score / max(len(query_keywords), len(context_keywords))
        else:
            weighted_similarity = 0.0
        
        # 综合Jaccard相似度和加权相似度
        return (jaccard_similarity + weighted_similarity) / 2.0
    
    def _calculate_recency_score(self, timestamp: datetime) -> float:
        """计算时间新近性分数"""
        if not self.strategy.use_temporal_decay:
            return 1.0
        
        now = datetime.now()
        time_diff = now - timestamp
        
        # 使用指数衰减函数
        # 1小时内的内容得分为1.0，之后按指数衰减
        hours_passed = time_diff.total_seconds() / 3600
        decay_factor = self.strategy.temporal_decay_factor
        
        score = math.exp(-hours_passed * (1 - decay_factor))
        return min(1.0, max(0.0, score))
    
    def _calculate_content_similarity(self, content: str, query: str) -> float:
        """计算内容相似度（简单实现）"""
        if not content or not query:
            return 0.0
        
        # 简单的字符串相似度计算
        content_lower = content.lower()
        query_lower = query.lower()
        
        # 检查直接包含关系
        if query_lower in content_lower:
            return 0.8
        
        # 计算共同子字符串
        common_chars = set(content_lower) & set(query_lower)
        total_chars = set(content_lower) | set(query_lower)
        
        if not total_chars:
            return 0.0
        
        char_similarity = len(common_chars) / len(total_chars)
        
        # 计算词汇重叠
        content_words = set(re.findall(r'\b\w+\b', content_lower))
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        if content_words and query_words:
            word_overlap = len(content_words & query_words) / len(content_words | query_words)
        else:
            word_overlap = 0.0
        
        return (char_similarity + word_overlap) / 2.0
    
    def _get_type_weight(self, context_type: ContextType) -> float:
        """根据上下文类型获取权重"""
        type_weights = {
            ContextType.CODE_SNIPPET: 1.2,      # 代码片段更重要
            ContextType.ERROR_INFO: 1.1,        # 错误信息较重要
            ContextType.USER_INPUT: 1.0,        # 用户输入标准权重
            ContextType.AGENT_RESPONSE: 0.9,    # Agent响应稍低
            ContextType.FILE_REFERENCE: 0.8,    # 文件引用较低
            ContextType.SYSTEM_INFO: 0.7        # 系统信息最低
        }
        
        return type_weights.get(context_type, 1.0)
    
    def _extract_code_snippets(self, text: str) -> List[str]:
        """从文本中提取代码片段"""
        code_snippets = []
        
        # 匹配代码块（三个反引号包围）
        code_block_pattern = r'```[\s\S]*?```'
        code_blocks = re.findall(code_block_pattern, text)
        code_snippets.extend([block.strip('`').strip() for block in code_blocks])
        
        # 匹配行内代码（单个反引号包围）
        inline_code_pattern = r'`([^`]+)`'
        inline_codes = re.findall(inline_code_pattern, text)
        code_snippets.extend(inline_codes)
        
        # 匹配可能的函数定义
        function_patterns = [
            r'def\s+\w+\s*\([^)]*\)\s*:',  # Python函数
            r'function\s+\w+\s*\([^)]*\)\s*{',  # JavaScript函数
            r'\w+\s+\w+\s*\([^)]*\)\s*{',  # C/Java风格函数
        ]
        
        for pattern in function_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            code_snippets.extend(matches)
        
        # 过滤掉太短的片段
        return [snippet for snippet in code_snippets if len(snippet.strip()) > 10]
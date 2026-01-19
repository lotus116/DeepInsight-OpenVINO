"""
上下文管理器实现

协调整个上下文记忆系统的运作。
"""

import time
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging

from .models import (
    Interaction, ContextItem, SessionContext, ContextConfig,
    FilterStrategy, StorageStats, ContextMemoryError, PerformanceIssue,
    ContextType
)
from .memory_store import MemoryStore
from .context_filter import ContextFilter
from .prompt_builder import PromptBuilder


class ContextManager:
    """上下文管理器类 - 协调整个上下文记忆系统"""
    
    def __init__(
        self,
        db_path: str = "context_memory.db",
        config: Optional[ContextConfig] = None,
        filter_strategy: Optional[FilterStrategy] = None
    ):
        """
        初始化上下文管理器
        
        Args:
            db_path: 数据库文件路径
            config: 上下文配置
            filter_strategy: 过滤策略
        """
        self.config = config or ContextConfig()
        self.memory_store = MemoryStore(db_path)
        self.context_filter = ContextFilter(filter_strategy)
        self.prompt_builder = PromptBuilder(max_tokens=self.config.token_limit)
        
        # 性能监控
        self._performance_stats = {
            "total_requests": 0,
            "avg_response_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # 简单的缓存机制
        self._session_cache: Dict[str, SessionContext] = {}
        self._cache_max_size = 100
        
        # 日志配置
        self.logger = logging.getLogger(__name__)
        if self.config.debug_mode:
            logging.basicConfig(level=logging.DEBUG)
    
    def process_user_input(
        self, 
        user_input: str, 
        session_id: str,
        system_instruction: Optional[str] = None
    ) -> str:
        """
        处理用户输入并生成上下文感知的提示
        
        Args:
            user_input: 用户输入
            session_id: 会话ID
            system_instruction: 系统指令
            
        Returns:
            包含上下文的完整提示
        """
        start_time = time.time()
        
        try:
            self._performance_stats["total_requests"] += 1
            
            if self.config.debug_mode:
                self.logger.debug(f"处理用户输入: {user_input[:50]}...")
            
            # 1. 获取或创建会话上下文
            session_context = self.get_session_context(session_id)
            if not session_context:
                session_context = self._create_new_session(session_id)
            
            # 2. 获取历史交互记录
            history = self.memory_store.get_session_history(
                session_id, 
                limit=self.config.max_history_items
            )
            
            # 3. 检测话题变化
            if self.config.enable_topic_detection:
                recent_context = self._get_recent_context_items(history)
                topic_changed = self.context_filter.detect_topic_change(
                    user_input, recent_context
                )
                
                if topic_changed:
                    self._handle_topic_change(session_context, user_input)
            
            # 4. 选择相关上下文
            relevant_context = self.context_filter.select_relevant_context(
                user_input, 
                history,
                max_items=min(10, self.config.max_history_items // 2)
            )
            
            # 5. 构建上下文感知的提示
            contextual_prompt = self.prompt_builder.build_contextual_prompt(
                user_input, relevant_context, system_instruction
            )
            
            # 6. 记录性能统计
            response_time = time.time() - start_time
            self._update_performance_stats(response_time)
            
            if self.config.debug_mode:
                self.logger.debug(f"处理完成，耗时: {response_time:.3f}秒")
                self.logger.debug(f"选择了 {len(relevant_context)} 个相关上下文项")
            
            return contextual_prompt
            
        except Exception as e:
            self.logger.error(f"处理用户输入失败: {e}")
            # 降级处理：返回基本提示
            return self._create_fallback_prompt(user_input, system_instruction)
    
    def update_context(self, session_id: str, interaction: Interaction) -> None:
        """
        更新会话上下文
        
        Args:
            session_id: 会话ID
            interaction: 交互记录
        """
        try:
            # 1. 保存交互记录
            self.memory_store.save_interaction(session_id, interaction)
            
            # 2. 从交互中提取上下文项并保存
            context_items = self._extract_context_items_from_interaction(interaction)
            for item in context_items:
                self.memory_store.save_context_item(session_id, item)
            
            # 3. 更新会话上下文
            session_context = self.get_session_context(session_id)
            if session_context:
                session_context.last_updated = datetime.now()
                session_context.interaction_count += 1
                
                # 更新活跃文件列表
                self._update_active_files(session_context, interaction)
                
                # 保存更新的会话上下文
                self.memory_store.save_session_context(session_context)
                
                # 更新缓存
                self._session_cache[session_id] = session_context
            
            if self.config.debug_mode:
                self.logger.debug(f"更新会话上下文: {session_id}")
                
        except Exception as e:
            self.logger.error(f"更新上下文失败: {e}")
            raise ContextMemoryError(f"更新上下文失败: {e}")
    
    def get_session_context(self, session_id: str) -> Optional[SessionContext]:
        """
        获取会话上下文
        
        Args:
            session_id: 会话ID
            
        Returns:
            会话上下文或None
        """
        try:
            # 先检查缓存
            if session_id in self._session_cache:
                self._performance_stats["cache_hits"] += 1
                return self._session_cache[session_id]
            
            # 从存储中获取
            self._performance_stats["cache_misses"] += 1
            session_context = self.memory_store.get_session_context(session_id)
            
            if session_context:
                # 添加到缓存
                self._add_to_cache(session_id, session_context)
            
            return session_context
            
        except Exception as e:
            self.logger.error(f"获取会话上下文失败: {e}")
            return None
    
    def configure_strategy(self, config: ContextConfig) -> None:
        """
        配置上下文策略
        
        Args:
            config: 新的配置
        """
        try:
            if not config.validate():
                raise ContextMemoryError("无效的配置参数")
            
            old_config = self.config
            self.config = config
            
            # 更新组件配置
            self.prompt_builder.max_tokens = config.token_limit
            
            # 如果调试模式发生变化，更新日志级别
            if config.debug_mode != old_config.debug_mode:
                if config.debug_mode:
                    logging.getLogger(__name__).setLevel(logging.DEBUG)
                else:
                    logging.getLogger(__name__).setLevel(logging.INFO)
            
            # 清空缓存以应用新配置
            self._session_cache.clear()
            
            if self.config.debug_mode:
                self.logger.debug("配置已更新")
                
        except Exception as e:
            self.logger.error(f"配置策略失败: {e}")
            raise ContextMemoryError(f"配置策略失败: {e}")
    
    def cleanup_expired_data(self) -> int:
        """
        清理过期数据
        
        Returns:
            清理的记录数量
        """
        try:
            deleted_count = self.memory_store.cleanup_expired_data(
                self.config.context_retention_days
            )
            
            # 清空缓存以确保一致性
            self._session_cache.clear()
            
            if self.config.debug_mode:
                self.logger.debug(f"清理了 {deleted_count} 条过期记录")
            
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"清理过期数据失败: {e}")
            raise ContextMemoryError(f"清理过期数据失败: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计信息
        
        Returns:
            性能统计字典
        """
        storage_stats = self.memory_store.get_storage_stats()
        
        return {
            "context_manager": self._performance_stats.copy(),
            "storage": storage_stats.__dict__,
            "cache": {
                "size": len(self._session_cache),
                "max_size": self._cache_max_size,
                "hit_rate": self._calculate_cache_hit_rate()
            }
        }
    
    def handle_high_load(self) -> None:
        """处理高负载情况"""
        try:
            # 1. 减少缓存大小
            if len(self._session_cache) > self._cache_max_size // 2:
                self._trim_cache()
            
            # 2. 降低上下文项数量
            original_max_items = self.config.max_history_items
            self.config.max_history_items = max(5, original_max_items // 2)
            
            # 3. 提高相关性阈值
            if hasattr(self.context_filter.strategy, 'relevance_threshold'):
                self.context_filter.strategy.relevance_threshold = min(
                    0.8, self.context_filter.strategy.relevance_threshold * 1.5
                )
            
            if self.config.debug_mode:
                self.logger.debug("启用高负载处理模式")
                
        except Exception as e:
            self.logger.error(f"处理高负载失败: {e}")
            raise PerformanceIssue(f"处理高负载失败: {e}")
    
    def _create_new_session(self, session_id: str) -> SessionContext:
        """创建新会话"""
        session_context = SessionContext(
            session_id=session_id,
            configuration=self.config
        )
        
        self.memory_store.save_session_context(session_context)
        self._add_to_cache(session_id, session_context)
        
        return session_context
    
    def _get_recent_context_items(self, history: List[Interaction]) -> List[ContextItem]:
        """从历史记录中获取最近的上下文项"""
        context_items = []
        
        # 只取最近的几个交互
        recent_history = history[:5]
        
        for interaction in recent_history:
            if interaction.user_input:
                context_items.append(ContextItem(
                    content=interaction.user_input,
                    type=ContextType.USER_INPUT,
                    timestamp=interaction.timestamp,
                    source_interaction_id=interaction.id
                ))
        
        return context_items
    
    def _handle_topic_change(self, session_context: SessionContext, new_input: str) -> None:
        """处理话题变化"""
        # 简单的话题提取（实际应用中可能需要更复杂的NLP）
        keywords = self.context_filter._extract_keywords(new_input)
        if keywords:
            # 取最重要的关键词作为新话题
            session_context.current_topic = list(keywords)[0]
            
            if self.config.debug_mode:
                self.logger.debug(f"检测到话题变化: {session_context.current_topic}")
    
    def _extract_context_items_from_interaction(self, interaction: Interaction) -> List[ContextItem]:
        """从交互中提取上下文项"""
        context_items = []
        
        # 从用户输入中提取代码片段
        if interaction.user_input:
            code_snippets = self.context_filter._extract_code_snippets(interaction.user_input)
            for snippet in code_snippets:
                context_items.append(ContextItem(
                    content=snippet,
                    type=ContextType.CODE_SNIPPET,
                    timestamp=interaction.timestamp,
                    source_interaction_id=interaction.id
                ))
        
        # 检查是否包含错误信息
        error_keywords = ["error", "错误", "exception", "异常", "failed", "失败"]
        if any(keyword in interaction.user_input.lower() for keyword in error_keywords):
            context_items.append(ContextItem(
                content=interaction.user_input,
                type=ContextType.ERROR_INFO,
                timestamp=interaction.timestamp,
                source_interaction_id=interaction.id
            ))
        
        # 检查文件引用
        file_references = self._extract_file_references(interaction.user_input)
        for file_ref in file_references:
            context_items.append(ContextItem(
                content=file_ref,
                type=ContextType.FILE_REFERENCE,
                timestamp=interaction.timestamp,
                source_interaction_id=interaction.id
            ))
        
        return context_items
    
    def _update_active_files(self, session_context: SessionContext, interaction: Interaction) -> None:
        """更新活跃文件列表"""
        # 简单的文件名检测
        import re
        file_pattern = r'\b\w+\.\w+\b'
        
        text = interaction.user_input + " " + interaction.agent_response
        potential_files = re.findall(file_pattern, text)
        
        for file_name in potential_files:
            if file_name not in session_context.active_files:
                session_context.active_files.append(file_name)
        
        # 限制活跃文件数量
        if len(session_context.active_files) > 10:
            session_context.active_files = session_context.active_files[-10:]
    
    def _create_fallback_prompt(self, user_input: str, system_instruction: Optional[str]) -> str:
        """创建降级提示"""
        if system_instruction:
            return f"{system_instruction}\n\n用户问题: {user_input}"
        return f"用户问题: {user_input}"
    
    def _add_to_cache(self, session_id: str, session_context: SessionContext) -> None:
        """添加到缓存"""
        if len(self._session_cache) >= self._cache_max_size:
            self._trim_cache()
        
        self._session_cache[session_id] = session_context
    
    def _trim_cache(self) -> None:
        """修剪缓存"""
        # 移除最旧的一半缓存项
        items = list(self._session_cache.items())
        items.sort(key=lambda x: x[1].last_updated)
        
        keep_count = len(items) // 2
        self._session_cache = dict(items[-keep_count:])
    
    def _update_performance_stats(self, response_time: float) -> None:
        """更新性能统计"""
        total_requests = self._performance_stats["total_requests"]
        current_avg = self._performance_stats["avg_response_time"]
        
        # 计算新的平均响应时间
        new_avg = ((current_avg * (total_requests - 1)) + response_time) / total_requests
        self._performance_stats["avg_response_time"] = new_avg
    
    def _calculate_cache_hit_rate(self) -> float:
        """计算缓存命中率"""
        total_requests = self._performance_stats["cache_hits"] + self._performance_stats["cache_misses"]
        if total_requests == 0:
            return 0.0
        
        return self._performance_stats["cache_hits"] / total_requests
    
    def _extract_file_references(self, text: str) -> List[str]:
        """从文本中提取文件引用"""
        import re
        
        # 匹配常见的文件模式
        file_patterns = [
            r'\b\w+\.\w+\b',  # 基本文件名模式
            r'["\']([^"\']+\.[^"\']+)["\']',  # 引号中的文件路径
            r'(?:src/|lib/|test/|tests/)[\w/]+\.\w+',  # 常见目录结构
        ]
        
        file_references = []
        for pattern in file_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]  # 从捕获组中提取
                
                # 过滤掉明显不是文件的匹配
                if self._is_likely_file_reference(match):
                    file_references.append(match)
        
        return list(set(file_references))  # 去重
    
    def _is_likely_file_reference(self, text: str) -> bool:
        """判断文本是否可能是文件引用"""
        # 常见的文件扩展名
        file_extensions = {
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp',
            '.html', '.css', '.json', '.xml', '.yaml', '.yml', '.md',
            '.txt', '.log', '.sql', '.sh', '.bat', '.ps1'
        }
        
        # 检查是否有文件扩展名
        for ext in file_extensions:
            if text.lower().endswith(ext):
                return True
        
        # 排除明显不是文件的文本
        if len(text) > 100 or ' ' in text:
            return False
        
        return False
    
    def track_code_modification(self, session_id: str, file_path: str, old_code: str, new_code: str) -> None:
        """
        跟踪代码修改历史
        
        Args:
            session_id: 会话ID
            file_path: 文件路径
            old_code: 修改前的代码
            new_code: 修改后的代码
        """
        try:
            # 创建代码修改记录
            modification_record = {
                "file_path": file_path,
                "old_code": old_code,
                "new_code": new_code,
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id
            }
            
            # 保存为上下文项
            context_item = ContextItem(
                content=f"代码修改记录: {file_path}\n旧代码:\n{old_code}\n新代码:\n{new_code}",
                type=ContextType.CODE_SNIPPET,
                timestamp=datetime.now(),
                metadata=modification_record
            )
            
            self.memory_store.save_context_item(session_id, context_item)
            
            if self.config.debug_mode:
                self.logger.debug(f"记录代码修改: {file_path}")
                
        except Exception as e:
            self.logger.error(f"跟踪代码修改失败: {e}")
    
    def get_code_modification_history(self, session_id: str, file_path: Optional[str] = None) -> List[ContextItem]:
        """
        获取代码修改历史
        
        Args:
            session_id: 会话ID
            file_path: 可选的文件路径过滤
            
        Returns:
            代码修改历史列表
        """
        try:
            # 获取所有代码片段类型的上下文项
            all_items = self.memory_store.get_context_items_by_type(session_id, ContextType.CODE_SNIPPET)
            
            # 过滤出修改记录
            modification_history = []
            for item in all_items:
                if "代码修改记录" in item.content:
                    if file_path is None or (item.metadata and item.metadata.get("file_path") == file_path):
                        modification_history.append(item)
            
            # 按时间排序
            modification_history.sort(key=lambda x: x.timestamp, reverse=True)
            
            return modification_history
            
        except Exception as e:
            self.logger.error(f"获取代码修改历史失败: {e}")
            return []
    
    def track_error_resolution(self, session_id: str, error_message: str, solution: str, success: bool = True) -> None:
        """
        跟踪错误解决尝试
        
        Args:
            session_id: 会话ID
            error_message: 错误信息
            solution: 解决方案
            success: 是否成功解决
        """
        try:
            # 创建错误解决记录
            resolution_record = {
                "error_message": error_message,
                "solution": solution,
                "success": success,
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id
            }
            
            # 保存为上下文项
            context_item = ContextItem(
                content=f"错误解决记录: {'成功' if success else '失败'}\n错误: {error_message}\n解决方案: {solution}",
                type=ContextType.ERROR_INFO,
                timestamp=datetime.now(),
                metadata=resolution_record
            )
            
            self.memory_store.save_context_item(session_id, context_item)
            
            if self.config.debug_mode:
                self.logger.debug(f"记录错误解决尝试: {success}")
                
        except Exception as e:
            self.logger.error(f"跟踪错误解决失败: {e}")
    
    def get_error_resolution_history(self, session_id: str, error_pattern: Optional[str] = None) -> List[ContextItem]:
        """
        获取错误解决历史
        
        Args:
            session_id: 会话ID
            error_pattern: 可选的错误模式过滤
            
        Returns:
            错误解决历史列表
        """
        try:
            # 获取所有错误信息类型的上下文项
            all_items = self.memory_store.get_context_items_by_type(session_id, ContextType.ERROR_INFO)
            
            # 过滤出解决记录
            resolution_history = []
            for item in all_items:
                if "错误解决记录" in item.content:
                    if error_pattern is None or error_pattern.lower() in item.content.lower():
                        resolution_history.append(item)
            
            # 按时间排序，成功的解决方案优先
            resolution_history.sort(key=lambda x: (x.metadata.get("success", False), x.timestamp), reverse=True)
            
            return resolution_history
            
        except Exception as e:
            self.logger.error(f"获取错误解决历史失败: {e}")
            return []
    
    def manage_file_associations(self, session_id: str, primary_file: str, related_files: List[str]) -> None:
        """
        管理文件关联关系
        
        Args:
            session_id: 会话ID
            primary_file: 主文件
            related_files: 相关文件列表
        """
        try:
            # 创建文件关联记录
            association_record = {
                "primary_file": primary_file,
                "related_files": related_files,
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id
            }
            
            # 保存为上下文项
            context_item = ContextItem(
                content=f"文件关联: {primary_file} 关联文件: {', '.join(related_files)}",
                type=ContextType.FILE_REFERENCE,
                timestamp=datetime.now(),
                metadata=association_record
            )
            
            self.memory_store.save_context_item(session_id, context_item)
            
            # 更新会话上下文中的活跃文件
            session_context = self.get_session_context(session_id)
            if session_context:
                all_files = [primary_file] + related_files
                for file_name in all_files:
                    if file_name not in session_context.active_files:
                        session_context.active_files.append(file_name)
                
                # 限制活跃文件数量
                if len(session_context.active_files) > 20:
                    session_context.active_files = session_context.active_files[-20:]
                
                self.memory_store.save_session_context(session_context)
                self._session_cache[session_id] = session_context
            
            if self.config.debug_mode:
                self.logger.debug(f"管理文件关联: {primary_file} -> {related_files}")
                
        except Exception as e:
            self.logger.error(f"管理文件关联失败: {e}")
    
    def get_related_files(self, session_id: str, file_path: str) -> List[str]:
        """
        获取与指定文件相关的文件列表
        
        Args:
            session_id: 会话ID
            file_path: 文件路径
            
        Returns:
            相关文件列表
        """
        try:
            # 获取所有文件引用类型的上下文项
            all_items = self.memory_store.get_context_items_by_type(session_id, ContextType.FILE_REFERENCE)
            
            related_files = set()
            
            for item in all_items:
                if item.metadata and "文件关联" in item.content:
                    primary_file = item.metadata.get("primary_file")
                    related_file_list = item.metadata.get("related_files", [])
                    
                    # 如果当前文件是主文件，添加所有相关文件
                    if primary_file == file_path:
                        related_files.update(related_file_list)
                    
                    # 如果当前文件在相关文件中，添加主文件和其他相关文件
                    elif file_path in related_file_list:
                        related_files.add(primary_file)
                        related_files.update(f for f in related_file_list if f != file_path)
            
            return list(related_files)
            
        except Exception as e:
            self.logger.error(f"获取相关文件失败: {e}")
            return []
    
    def enable_debug_mode(self, enable: bool = True) -> None:
        """
        启用或禁用调试模式
        
        Args:
            enable: 是否启用调试模式
        """
        self.config.debug_mode = enable
        
        if enable:
            logging.getLogger(__name__).setLevel(logging.DEBUG)
            self.logger.debug("调试模式已启用")
        else:
            logging.getLogger(__name__).setLevel(logging.INFO)
            self.logger.info("调试模式已禁用")
    
    def get_decision_trace(self, session_id: str, user_input: str) -> Dict[str, Any]:
        """
        获取决策过程跟踪信息
        
        Args:
            session_id: 会话ID
            user_input: 用户输入
            
        Returns:
            决策过程跟踪信息
        """
        trace = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "user_input": user_input[:100] + "..." if len(user_input) > 100 else user_input,
            "steps": []
        }
        
        try:
            # 1. 会话上下文获取
            session_context = self.get_session_context(session_id)
            trace["steps"].append({
                "step": "获取会话上下文",
                "result": "成功" if session_context else "创建新会话",
                "details": {
                    "session_exists": session_context is not None,
                    "interaction_count": session_context.interaction_count if session_context else 0
                }
            })
            
            # 2. 历史记录检索
            history = self.memory_store.get_session_history(session_id, limit=self.config.max_history_items)
            trace["steps"].append({
                "step": "检索历史记录",
                "result": f"找到 {len(history)} 条记录",
                "details": {
                    "history_count": len(history),
                    "limit": self.config.max_history_items
                }
            })
            
            # 3. 话题检测
            if self.config.enable_topic_detection and history:
                recent_context = self._get_recent_context_items(history)
                topic_changed = self.context_filter.detect_topic_change(user_input, recent_context)
                trace["steps"].append({
                    "step": "话题变化检测",
                    "result": "话题已变化" if topic_changed else "话题未变化",
                    "details": {
                        "topic_detection_enabled": True,
                        "topic_changed": topic_changed,
                        "recent_context_count": len(recent_context)
                    }
                })
            
            # 4. 上下文选择
            relevant_context = self.context_filter.select_relevant_context(
                user_input, history, max_items=min(10, self.config.max_history_items // 2)
            )
            trace["steps"].append({
                "step": "相关上下文选择",
                "result": f"选择了 {len(relevant_context)} 个上下文项",
                "details": {
                    "total_available": len(history),
                    "selected_count": len(relevant_context),
                    "max_items": min(10, self.config.max_history_items // 2)
                }
            })
            
            # 5. 提示构建
            contextual_prompt = self.prompt_builder.build_contextual_prompt(
                user_input, relevant_context
            )
            trace["steps"].append({
                "step": "提示构建",
                "result": f"生成 {len(contextual_prompt)} 字符的提示",
                "details": {
                    "prompt_length": len(contextual_prompt),
                    "token_limit": self.config.token_limit
                }
            })
            
            trace["success"] = True
            
        except Exception as e:
            trace["steps"].append({
                "step": "错误处理",
                "result": f"发生错误: {str(e)}",
                "details": {
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            })
            trace["success"] = False
        
        return trace
    
    def recover_from_error(self, session_id: str, error: Exception) -> str:
        """
        从错误中恢复
        
        Args:
            session_id: 会话ID
            error: 发生的错误
            
        Returns:
            恢复后的响应
        """
        try:
            self.logger.error(f"尝试从错误中恢复: {error}")
            
            # 记录错误信息
            error_context = ContextItem(
                content=f"系统错误: {type(error).__name__}: {str(error)}",
                type=ContextType.ERROR_INFO,
                timestamp=datetime.now(),
                metadata={
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "recovery_attempted": True
                }
            )
            
            self.memory_store.save_context_item(session_id, error_context)
            
            # 根据错误类型选择恢复策略
            if isinstance(error, StorageError):
                return self._recover_from_storage_error(session_id)
            elif isinstance(error, ContextMemoryError):
                return self._recover_from_context_error(session_id)
            elif isinstance(error, PerformanceIssue):
                return self._recover_from_performance_issue(session_id)
            else:
                return self._generic_error_recovery(session_id)
                
        except Exception as recovery_error:
            self.logger.error(f"错误恢复失败: {recovery_error}")
            return "系统遇到了无法恢复的错误，请重新开始对话。"
    
    def _recover_from_storage_error(self, session_id: str) -> str:
        """从存储错误中恢复"""
        try:
            # 尝试重新初始化数据库连接
            self.memory_store._init_database()
            return "存储系统已重新初始化，请重试您的请求。"
        except Exception:
            return "存储系统暂时不可用，当前对话将使用基本模式。"
    
    def _recover_from_context_error(self, session_id: str) -> str:
        """从上下文错误中恢复"""
        try:
            # 清空会话缓存
            if session_id in self._session_cache:
                del self._session_cache[session_id]
            return "上下文已重置，请重新描述您的问题。"
        except Exception:
            return "上下文系统遇到问题，建议开始新的对话会话。"
    
    def _recover_from_performance_issue(self, session_id: str) -> str:
        """从性能问题中恢复"""
        try:
            # 启用高负载处理模式
            self.handle_high_load()
            return "系统已切换到高效模式，请重试您的请求。"
        except Exception:
            return "系统负载过高，请稍后重试。"
    
    def _generic_error_recovery(self, session_id: str) -> str:
        """通用错误恢复"""
        return "系统遇到了意外错误，已记录相关信息。请重新描述您的问题，我会尽力帮助您。"
    
    def validate_system_health(self) -> Dict[str, Any]:
        """
        验证系统健康状态
        
        Returns:
            系统健康状态报告
        """
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "components": {},
            "issues": []
        }
        
        try:
            # 检查数据库连接
            try:
                stats = self.memory_store.get_storage_stats()
                health_report["components"]["database"] = {
                    "status": "healthy",
                    "total_sessions": stats.total_sessions,
                    "total_interactions": stats.total_interactions,
                    "storage_size_mb": stats.storage_size_mb
                }
            except Exception as e:
                health_report["components"]["database"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_report["issues"].append(f"数据库连接问题: {e}")
            
            # 检查缓存状态
            cache_size = len(self._session_cache)
            cache_hit_rate = self._calculate_cache_hit_rate()
            
            if cache_size > self._cache_max_size * 0.9:
                health_report["issues"].append("缓存使用率过高")
            
            health_report["components"]["cache"] = {
                "status": "healthy" if cache_size <= self._cache_max_size else "warning",
                "size": cache_size,
                "max_size": self._cache_max_size,
                "hit_rate": cache_hit_rate
            }
            
            # 检查性能指标
            avg_response_time = self._performance_stats["avg_response_time"]
            if avg_response_time > 1.0:  # 超过1秒认为性能有问题
                health_report["issues"].append(f"平均响应时间过长: {avg_response_time:.3f}秒")
            
            health_report["components"]["performance"] = {
                "status": "healthy" if avg_response_time <= 1.0 else "warning",
                "avg_response_time": avg_response_time,
                "total_requests": self._performance_stats["total_requests"]
            }
            
            # 检查配置有效性
            if not self.config.validate():
                health_report["issues"].append("配置参数无效")
                health_report["components"]["configuration"] = {"status": "unhealthy"}
            else:
                health_report["components"]["configuration"] = {"status": "healthy"}
            
            # 确定整体状态
            if health_report["issues"]:
                if any("unhealthy" in str(comp) for comp in health_report["components"].values()):
                    health_report["overall_status"] = "unhealthy"
                else:
                    health_report["overall_status"] = "warning"
            
        except Exception as e:
            health_report["overall_status"] = "unhealthy"
            health_report["issues"].append(f"健康检查失败: {e}")
        
        return health_report
#!/usr/bin/env python3
"""
上下文记忆系统集成模块

将上下文记忆系统集成到主应用程序中，提供无缝的对话上下文管理。
"""

import streamlit as st
from typing import List, Dict, Any, Optional
from datetime import datetime

from context_memory.context_manager import ContextManager
from context_memory.models import Interaction, ContextConfig, ContextType


class StreamlitContextIntegration:
    """Streamlit应用的上下文记忆集成"""
    
    def __init__(self, db_path: str = "streamlit_context_memory.db"):
        """
        初始化上下文记忆集成
        
        Args:
            db_path: 数据库文件路径
        """
        # 配置上下文记忆系统
        config = ContextConfig(
            debug_mode=False,  # 生产环境关闭调试
            max_history_items=15,  # 适中的历史记录数量
            enable_topic_detection=True,  # 启用话题检测
            token_limit=8000,  # 适合大多数LLM的token限制
            context_retention_days=7  # 保留7天的对话历史
        )
        
        self.context_manager = ContextManager(db_path, config)
        
        # 初始化session state - 从配置文件加载设置
        self._load_memory_settings()
        
        if 'current_session_id' not in st.session_state:
            st.session_state.current_session_id = self._generate_session_id()
    
    def _load_memory_settings(self):
        """从配置文件加载记忆设置"""
        try:
            import json
            import os
            
            config_file = "data/memory_config.json"
            
            # 默认设置
            default_settings = {
                'context_memory_enabled': True,
                'context_memory_depth': 5,
                'context_memory_strength': 0.7,
                'context_auto_clean': True,
                'context_persist_memory': False,
                'context_privacy_mode': False
            }
            
            # 首先设置默认值
            for key, value in default_settings.items():
                if key not in st.session_state:
                    st.session_state[key] = value
            
            # 如果配置文件存在，加载并覆盖默认设置
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    saved_settings = json.load(f)
                    
                # 更新session_state中的设置
                for key, value in saved_settings.items():
                    if key in default_settings:  # 只加载已知的设置项
                        st.session_state[key] = value
                        
                print(f"✅ 已加载记忆设置: context_memory_enabled = {st.session_state.get('context_memory_enabled')}")
            else:
                print("📝 使用默认记忆设置")
                    
        except Exception as e:
            print(f"加载记忆设置失败: {e}")
            # 确保至少有默认设置
            if 'context_memory_enabled' not in st.session_state:
                st.session_state.context_memory_enabled = True
    
    def _save_memory_settings(self):
        """保存记忆设置到配置文件"""
        try:
            import json
            import os
            
            os.makedirs("data", exist_ok=True)
            config_file = "data/memory_config.json"
            
            settings = {
                'context_memory_enabled': st.session_state.get('context_memory_enabled', True),
                'context_memory_depth': st.session_state.get('context_memory_depth', 5),
                'context_memory_strength': st.session_state.get('context_memory_strength', 0.7),
                'context_auto_clean': st.session_state.get('context_auto_clean', True),
                'context_persist_memory': st.session_state.get('context_persist_memory', False),
                'context_privacy_mode': st.session_state.get('context_privacy_mode', False)
            }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"保存记忆设置失败: {e}")
    
    def _generate_session_id(self) -> str:
        """生成会话ID"""
        # 使用streamlit的session_state作为基础，加上时间戳
        import hashlib
        base_id = f"streamlit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return hashlib.md5(base_id.encode()).hexdigest()[:16]
    
    def get_contextual_prompt(self, user_input: str, system_instruction: Optional[str] = None) -> str:
        """
        获取包含上下文的提示
        
        Args:
            user_input: 用户输入
            system_instruction: 系统指令
            
        Returns:
            包含上下文的完整提示
        """
        if not st.session_state.get('context_memory_enabled', True):
            # 如果禁用了上下文记忆，返回原始输入
            if system_instruction:
                return f"{system_instruction}\n\n用户问题: {user_input}"
            return user_input
        
        try:
            session_id = st.session_state.current_session_id
            return self.context_manager.process_user_input(
                user_input, 
                session_id, 
                system_instruction
            )
        except Exception as e:
            st.error(f"上下文处理失败: {e}")
            # 降级处理
            if system_instruction:
                return f"{system_instruction}\n\n用户问题: {user_input}"
            return user_input
    
    def update_conversation_context(self, user_input: str, agent_response: str) -> None:
        """
        更新对话上下文
        
        Args:
            user_input: 用户输入
            agent_response: Agent回复
        """
        if not st.session_state.get('context_memory_enabled', True):
            return
        
        try:
            # 应用隐私模式过滤
            filtered_user_input = self.apply_privacy_mode(user_input)
            filtered_agent_response = self.apply_privacy_mode(agent_response)
            
            session_id = st.session_state.current_session_id
            
            # 创建交互记录
            interaction = Interaction(
                session_id=session_id,
                user_input=filtered_user_input,
                agent_response=filtered_agent_response,
                timestamp=datetime.now()
            )
            
            # 更新上下文
            self.context_manager.update_context(session_id, interaction)
            
            # 自动清理过期记忆
            self.auto_cleanup_expired_memory()
            
        except Exception as e:
            st.error(f"上下文更新失败: {e}")
    
    def track_code_discussion(self, file_path: str, old_code: str, new_code: str) -> None:
        """
        跟踪代码讨论
        
        Args:
            file_path: 文件路径
            old_code: 旧代码
            new_code: 新代码
        """
        if not st.session_state.get('context_memory_enabled', True):
            return
        
        try:
            session_id = st.session_state.current_session_id
            
            # 应用隐私模式过滤
            filtered_old_code = self.apply_privacy_mode(old_code)
            filtered_new_code = self.apply_privacy_mode(new_code)
            
            self.context_manager.track_code_modification(
                session_id, file_path, filtered_old_code, filtered_new_code
            )
        except Exception as e:
            st.error(f"代码跟踪失败: {e}")
    
    def track_error_resolution(self, error_message: str, solution: str, success: bool = True) -> None:
        """
        跟踪错误解决
        
        Args:
            error_message: 错误信息
            solution: 解决方案
            success: 是否成功
        """
        if not st.session_state.get('context_memory_enabled', True):
            return
        
        try:
            session_id = st.session_state.current_session_id
            
            # 应用隐私模式过滤
            filtered_error = self.apply_privacy_mode(error_message)
            filtered_solution = self.apply_privacy_mode(solution)
            
            self.context_manager.track_error_resolution(
                session_id, filtered_error, filtered_solution, success
            )
        except Exception as e:
            st.error(f"错误跟踪失败: {e}")
    
    def clear_all_memory(self) -> bool:
        """清理所有记忆数据"""
        try:
            # 直接调用 memory_store 的 cleanup_expired_data 方法清理所有数据
            deleted_count = self.context_manager.memory_store.cleanup_expired_data(0)  # 0天表示清理所有数据
            
            # 清理缓存
            self.context_manager._session_cache.clear()
            
            # 重新创建当前会话
            session_id = st.session_state.current_session_id
            self.context_manager._create_new_session(session_id)
            
            return True
        except Exception as e:
            st.error(f"清理记忆失败: {e}")
            return False
    
    def auto_cleanup_expired_memory(self):
        """自动清理过期记忆"""
        try:
            if st.session_state.get('context_auto_clean', True):
                deleted_count = self.context_manager.cleanup_expired_data()  # 使用配置的保留天数
                if deleted_count > 0:
                    print(f"自动清理了 {deleted_count} 条过期记忆")
        except Exception as e:
            print(f"自动清理失败: {e}")
    
    def apply_privacy_mode(self, content: str) -> str:
        """应用隐私模式过滤"""
        if not st.session_state.get('context_privacy_mode', False):
            return content
        
        try:
            import re
            
            # 简单的敏感信息过滤
            # 过滤邮箱
            content = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[邮箱]', content)
            # 过滤电话号码
            content = re.sub(r'\b\d{3}-?\d{3,4}-?\d{4}\b', '[电话]', content)
            # 过滤身份证号（简单模式）
            content = re.sub(r'\b\d{15}|\d{18}\b', '[身份证]', content)
            # 过滤IP地址
            content = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP地址]', content)
            
            return content
        except Exception as e:
            print(f"隐私模式过滤失败: {e}")
            return content
    
    def get_context_stats(self) -> Dict[str, Any]:
        """获取上下文统计信息"""
        try:
            # 获取实际的统计数据
            performance_stats = self.context_manager.get_performance_stats()
            
            # 获取当前会话的统计
            session_id = st.session_state.current_session_id
            session_context = self.context_manager.get_session_context(session_id)
            
            # 计算记忆容量使用率
            max_items = self.context_manager.config.max_history_items
            current_items = session_context.interaction_count if session_context else 0
            memory_usage = min(100, (current_items / max_items) * 100) if max_items > 0 else 0
            
            # 计算关联精度（基于最近的上下文匹配成功率）
            cache_stats = performance_stats.get('cache', {})
            hit_rate = cache_stats.get('hit_rate', 0.0)
            association_accuracy = min(100, hit_rate * 100 + 70)  # 基础70% + 缓存命中率加成
            
            return {
                'saved_conversations': current_items,
                'memory_capacity_percent': int(memory_usage),
                'association_accuracy_percent': int(association_accuracy),
                'total_requests': performance_stats.get('context_manager', {}).get('total_requests', 0),
                'avg_response_time': performance_stats.get('context_manager', {}).get('avg_response_time', 0.0),
                'cache_hit_rate': hit_rate
            }
        except Exception as e:
            st.error(f"获取统计信息失败: {e}")
            return {
                'saved_conversations': 0,
                'memory_capacity_percent': 0,
                'association_accuracy_percent': 0,
                'total_requests': 0,
                'avg_response_time': 0.0,
                'cache_hit_rate': 0.0
            }
    
    def render_context_sidebar(self) -> None:
        """渲染上下文记忆侧边栏"""
        with st.sidebar:
            st.markdown("### 🧠 上下文记忆")
            
            # 启用/禁用开关
            enabled = st.checkbox(
                "启用上下文记忆", 
                value=st.session_state.get('context_memory_enabled', True),
                help="启用后，AI将记住对话历史并提供更智能的回复"
            )
            st.session_state.context_memory_enabled = enabled
            
            if enabled:
                # 显示统计信息
                stats = self.get_context_stats()
                if stats:
                    context_stats = stats.get('context_manager', {})
                    cache_stats = stats.get('cache', {})
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("总请求", context_stats.get('total_requests', 0))
                        st.metric("缓存命中率", f"{cache_stats.get('hit_rate', 0):.1%}")
                    
                    with col2:
                        st.metric("响应时间", f"{context_stats.get('avg_response_time', 0):.3f}s")
                        st.metric("缓存大小", cache_stats.get('size', 0))
                
                # 会话管理 - 移除重复的新建会话按钮，使用下方统一的会话管理
                
                # 显示当前会话信息
                session_id = st.session_state.current_session_id
                try:
                    session_context = self.context_manager.get_session_context(session_id)
                    if session_context:
                        st.caption(f"会话ID: {session_id[:8]}...")
                        st.caption(f"交互次数: {session_context.interaction_count}")
                        if session_context.current_topic:
                            st.caption(f"当前话题: {session_context.current_topic}")
                except Exception:
                    pass
                
                # 数据清理
                if st.button("🧹 清理过期数据", help="清理7天前的对话数据"):
                    try:
                        deleted_count = self.context_manager.cleanup_expired_data()
                        st.success(f"已清理 {deleted_count} 条过期记录")
                    except Exception as e:
                        st.error(f"清理失败: {e}")
            
            else:
                st.info("上下文记忆已禁用，AI将不会记住对话历史")
    
    def render_context_debug_info(self) -> None:
        """渲染调试信息（仅在调试模式下）"""
        if not st.session_state.get('context_memory_enabled', True):
            return
        
        # 只在开发模式下显示
        if st.sidebar.checkbox("显示上下文调试信息", value=False):
            with st.expander("🔍 上下文调试信息", expanded=False):
                session_id = st.session_state.current_session_id
                
                try:
                    # 获取决策跟踪
                    trace = self.context_manager.get_decision_trace(
                        session_id, "调试信息查询"
                    )
                    
                    st.json(trace)
                    
                    # 健康检查
                    health = self.context_manager.validate_system_health()
                    st.subheader("系统健康状态")
                    
                    status_color = {
                        "healthy": "🟢",
                        "warning": "🟡", 
                        "unhealthy": "🔴"
                    }
                    
                    st.write(f"{status_color.get(health['overall_status'], '⚪')} 总体状态: {health['overall_status']}")
                    
                    if health['issues']:
                        st.warning("发现问题:")
                        for issue in health['issues']:
                            st.write(f"- {issue}")
                    
                except Exception as e:
                    st.error(f"调试信息获取失败: {e}")


# 全局实例
_context_integration = None

def get_context_integration() -> StreamlitContextIntegration:
    """获取上下文集成实例（单例模式）"""
    global _context_integration
    if _context_integration is None:
        _context_integration = StreamlitContextIntegration()
    return _context_integration


def integrate_with_messages(messages: List[Dict[str, Any]], user_input: str, system_instruction: Optional[str] = None) -> str:
    """
    与现有的messages系统集成
    
    Args:
        messages: 现有的消息列表
        user_input: 用户输入
        system_instruction: 系统指令
        
    Returns:
        包含上下文的提示
    """
    integration = get_context_integration()
    
    # 如果启用了上下文记忆，使用智能上下文选择
    if st.session_state.get('context_memory_enabled', True):
        return integration.get_contextual_prompt(user_input, system_instruction)
    else:
        # 否则使用传统的messages方式
        if system_instruction:
            return f"{system_instruction}\n\n用户问题: {user_input}"
        return user_input


def update_context_after_response(user_input: str, agent_response: str) -> None:
    """
    在Agent回复后更新上下文
    
    Args:
        user_input: 用户输入
        agent_response: Agent回复
    """
    integration = get_context_integration()
    integration.update_conversation_context(user_input, agent_response)


def render_context_ui() -> None:
    """渲染上下文记忆UI"""
    integration = get_context_integration()
    integration.render_context_sidebar()
    integration.render_context_debug_info()
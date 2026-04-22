"""
Intel® DeepInsight - 上下文记忆面板

从 app.py 提取的侧边栏上下文记忆系统 UI 组件。
"""
import time
import streamlit as st

# 延迟导入标志
_CONTEXT_MEMORY_AVAILABLE = None
_context_integration = None


def _get_context_memory_available():
    """延迟检查上下文记忆系统可用性"""
    global _CONTEXT_MEMORY_AVAILABLE
    if _CONTEXT_MEMORY_AVAILABLE is None:
        try:
            from context_memory_integration import get_context_integration
            _CONTEXT_MEMORY_AVAILABLE = True
        except ImportError:
            _CONTEXT_MEMORY_AVAILABLE = False
    return _CONTEXT_MEMORY_AVAILABLE


def _get_context_integration():
    """获取上下文集成实例"""
    global _context_integration
    if _context_integration is None:
        from context_memory_integration import get_context_integration
        _context_integration = get_context_integration()
    return _context_integration


def render_context_memory_panel():
    """渲染上下文记忆系统面板"""
    if not _get_context_memory_available():
        _render_unavailable_panel()
        return
    
    with st.expander("🧠 上下文记忆系统", expanded=False):
        _inject_panel_styles()
        _render_status_section()
        _render_config_section()
        _render_advanced_settings()
        _render_statistics_section()
        _render_usage_tips()


def _render_unavailable_panel():
    """渲染不可用状态的面板"""
    with st.expander("🧠 上下文记忆系统", expanded=False):
        st.warning("上下文记忆系统当前不可用")
        st.info("要启用上下文记忆功能，请确保已正确安装并配置相关模块。")


def _inject_panel_styles():
    """注入面板专用样式"""
    st.markdown("""
    <style>
        .context-status-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 600;
            margin-right: 8px;
            margin-bottom: 8px;
        }
        .context-status-enabled {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            color: #155724;
            border: 1px solid #b1dfbb;
        }
        .context-status-disabled {
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            color: #721c24;
            border: 1px solid #f1b0b7;
        }
        .context-setting-label {
            font-weight: 600;
            font-size: 0.9em;
            color: #495057;
            margin-bottom: 5px;
            display: flex;
            align-items: center;
        }
        .context-setting-label i {
            margin-right: 8px;
            color: #0068B5;
        }
        .context-tooltip {
            font-size: 0.85em;
            color: #6c757d;
            margin-top: 4px;
            line-height: 1.4;
            font-style: italic;
        }
    </style>
    """, unsafe_allow_html=True)


def _render_status_section():
    """渲染状态显示区域"""
    col_status, col_actions = st.columns([3, 2])
    
    with col_status:
        st.markdown("**📊 当前状态**")
        memory_enabled = st.session_state.get('context_memory_enabled', True)
        
        if memory_enabled:
            st.markdown('<span class="context-status-badge context-status-enabled">✅ 已启用</span>', unsafe_allow_html=True)
            st.caption("AI将记住对话历史，提供更智能的回复")
        else:
            st.markdown('<span class="context-status-badge context-status-disabled">⏸️ 已禁用</span>', unsafe_allow_html=True)
            st.caption("AI将不会记住对话历史")
    
    with col_actions:
        st.markdown("**⚙️ 操作**")
        current_memory_enabled = st.session_state.get('context_memory_enabled', True)
        toggle_label = "禁用记忆" if current_memory_enabled else "启用记忆"
        toggle_icon = "⏸️" if current_memory_enabled else "▶️"
        
        if st.button(f"{toggle_icon} {toggle_label}", 
                    use_container_width=True,
                    key="toggle_memory_btn"):
            new_state = not current_memory_enabled
            st.session_state.context_memory_enabled = new_state
            
            try:
                context_integration = _get_context_integration()
                context_integration._save_memory_settings()
                
                if new_state:
                    st.success("✅ 上下文记忆已启用")
                else:
                    st.info("⏸️ 上下文记忆已禁用")
            except Exception as e:
                st.error(f"保存设置失败: {e}")
            
            time.sleep(0.5)
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


def _render_config_section():
    """渲染配置设置区域"""
    st.markdown("**🔧 记忆配置**")
    
    # 记忆深度设置
    st.markdown('<div class="context-setting-label"><i>📏</i> 记忆深度</div>', unsafe_allow_html=True)
    
    if "context_memory_depth" not in st.session_state:
        st.session_state.context_memory_depth = 5
    
    memory_depth = st.slider(
        "保留的对话轮数",
        min_value=1,
        max_value=20,
        value=st.session_state.context_memory_depth,
        key="memory_depth_slider",
        label_visibility="collapsed",
        help="设置AI能够记住的最近对话轮数。推荐值：3-8轮"
    )
    
    if memory_depth != st.session_state.context_memory_depth:
        st.session_state.context_memory_depth = memory_depth
        try:
            context_integration = _get_context_integration()
            context_integration._save_memory_settings()
        except Exception:
            pass
    
    st.markdown('<div class="context-tooltip">💡 <strong>算法说明</strong>: 系统使用滑动窗口算法保留最近N轮对话。</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 记忆强度设置
    st.markdown('<div class="context-setting-label"><i>💪</i> 记忆强度</div>', unsafe_allow_html=True)
    
    if "context_memory_strength" not in st.session_state:
        st.session_state.context_memory_strength = 0.7
    
    memory_strength = st.slider(
        "记忆影响力权重",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.context_memory_strength,
        step=0.1,
        key="memory_strength_slider",
        label_visibility="collapsed",
        help="设置历史对话对当前回答的影响程度。推荐值：0.5-0.8"
    )
    
    if memory_strength != st.session_state.context_memory_strength:
        st.session_state.context_memory_strength = memory_strength
        try:
            context_integration = _get_context_integration()
            context_integration._save_memory_settings()
        except Exception:
            pass
    
    st.markdown('<div class="context-tooltip">💡 <strong>算法说明</strong>: 使用加权融合算法混合历史上下文与当前输入。</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def _render_advanced_settings():
    """渲染高级设置区域"""
    with st.expander("⚡ 高级设置", expanded=False):
        # 自动清理选项
        auto_clean = st.checkbox(
            "自动清理过期记忆",
            value=st.session_state.get('context_auto_clean', True),
            help="自动清理超过24小时的旧记忆，保持系统性能",
            key="auto_clean_checkbox"
        )
        if auto_clean != st.session_state.get('context_auto_clean', True):
            st.session_state.context_auto_clean = auto_clean
            try:
                context_integration = _get_context_integration()
                context_integration._save_memory_settings()
                if auto_clean:
                    context_integration.auto_cleanup_expired_memory()
                    st.success("✅ 已启用自动清理并执行了一次清理")
            except Exception as e:
                st.error(f"设置自动清理失败: {e}")
        
        # 记忆持久化选项
        persist_memory = st.checkbox(
            "持久化记忆到磁盘",
            value=st.session_state.get('context_persist_memory', False),
            help="将对话记忆保存到本地文件",
            key="persist_memory_checkbox"
        )
        if persist_memory != st.session_state.get('context_persist_memory', False):
            st.session_state.context_persist_memory = persist_memory
            try:
                context_integration = _get_context_integration()
                context_integration._save_memory_settings()
                if persist_memory:
                    st.info("💾 记忆持久化已启用")
                else:
                    st.info("⚠️ 注意：禁用持久化不会删除已保存的数据")
            except Exception as e:
                st.error(f"设置持久化失败: {e}")
        
        # 隐私模式
        privacy_mode = st.checkbox(
            "隐私模式（不保存敏感信息）",
            value=st.session_state.get('context_privacy_mode', False),
            help="启用后，系统会自动过滤敏感信息",
            key="privacy_mode_checkbox"
        )
        if privacy_mode != st.session_state.get('context_privacy_mode', False):
            st.session_state.context_privacy_mode = privacy_mode
            try:
                context_integration = _get_context_integration()
                context_integration._save_memory_settings()
                if privacy_mode:
                    st.success("🔒 隐私模式已启用")
                else:
                    st.info("🔓 隐私模式已禁用")
            except Exception as e:
                st.error(f"设置隐私模式失败: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)


def _render_statistics_section():
    """渲染统计信息区域"""
    current_memory_enabled = st.session_state.get('context_memory_enabled', True)
    if not current_memory_enabled:
        return
    
    st.markdown("**📈 记忆统计**")
    
    try:
        context_integration = _get_context_integration()
        stats = context_integration.get_context_stats()
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            saved_conversations = stats.get('saved_conversations', 0)
            st.metric("已保存对话", f"{saved_conversations}轮", 
                    delta=f"+{min(2, saved_conversations)}" if saved_conversations > 0 else None)
        with col_stat2:
            memory_capacity = stats.get('memory_capacity_percent', 0)
            st.metric("记忆容量", f"{memory_capacity}%", 
                    delta=f"+{min(5, memory_capacity//10)}%" if memory_capacity > 0 else None)
        with col_stat3:
            association_accuracy = stats.get('association_accuracy_percent', 0)
            st.metric("关联精度", f"{association_accuracy}%", 
                    delta=f"+{min(3, association_accuracy//20)}%" if association_accuracy > 0 else None)
    except Exception as e:
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("已保存对话", "0轮")
        with col_stat2:
            st.metric("记忆容量", "0%")
        with col_stat3:
            st.metric("关联精度", "0%")
        st.caption(f"⚠️ 统计数据获取失败: {e}")
    
    # 清理记忆按钮
    if st.button("🗑️ 清理所有记忆", use_container_width=True, type="secondary"):
        if 'confirm_clear_memory' not in st.session_state:
            st.session_state.confirm_clear_memory = False
        
        if not st.session_state.confirm_clear_memory:
            st.session_state.confirm_clear_memory = True
            st.rerun()
    
    # 显示确认对话框
    if st.session_state.get('confirm_clear_memory', False):
        st.warning("⚠️ 确定要清理所有对话记忆吗？此操作不可撤销。")
        col_confirm1, col_confirm2 = st.columns(2)
        with col_confirm1:
            if st.button("✅ 确认清理", use_container_width=True, key="confirm_clear_btn"):
                try:
                    context_integration = _get_context_integration()
                    success = context_integration.clear_all_memory()
                    if success:
                        st.success("✅ 所有记忆已清理")
                        st.session_state.confirm_clear_memory = False
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ 清理失败")
                except Exception as e:
                    st.error(f"❌ 清理失败: {e}")
                st.session_state.confirm_clear_memory = False
        with col_confirm2:
            if st.button("❌ 取消", use_container_width=True, key="cancel_clear_btn"):
                st.session_state.confirm_clear_memory = False
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


def _render_usage_tips():
    """渲染使用提示"""
    st.info("💡 **使用提示**: 启用上下文记忆可以让AI更好地理解多轮对话的上下文，提供更连贯、更准确的回答。")

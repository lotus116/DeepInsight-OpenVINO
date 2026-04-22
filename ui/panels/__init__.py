"""
Intel® DeepInsight - UI 面板模块

从 app.py 提取的侧边栏面板组件。
"""

from ui.panels.context_memory import render_context_memory_panel

# 延迟导入其他面板（待实现）
def render_prompt_config_panel():
    """渲染 Prompt 配置面板（占位符）"""
    pass

def render_hardware_status_panel():
    """渲染硬件状态面板（占位符）"""
    pass

def render_model_settings_panel():
    """渲染模型设置面板（占位符）"""
    pass

def render_database_config_panel():
    """渲染数据库配置面板（占位符）"""
    pass

def render_session_manager_panel():
    """渲染会话管理面板（占位符）"""
    pass

__all__ = [
    'render_context_memory_panel',
    'render_prompt_config_panel',
    'render_hardware_status_panel',
    'render_model_settings_panel',
    'render_database_config_panel',
    'render_session_manager_panel',
]

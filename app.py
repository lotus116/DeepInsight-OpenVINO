# PS D:\比赛\intel\intel> wmic path win32_videocontroller get name
# Name
# Intel(R) Iris(R) Xe Graphics

# PS D:\比赛\intel\intel> D:\anaconda_download\anaconda3\python.exe -m streamlit run app.py


def create_openai_client_safe(api_key, base_url, timeout=60.0):
    """安全创建 OpenAI 客户端，兼容不同版本的 OpenAI 库"""
    try:
        from openai import OpenAI
        
        # 检查是否支持 http_client 参数
        import inspect
        sig = inspect.signature(OpenAI.__init__)
        supports_http_client = 'http_client' in sig.parameters
        
        if supports_http_client:
            try:
                import httpx
                # 尝试使用 http_client 参数（新版本）
                return OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    http_client=httpx.Client(proxies={}),
                    timeout=timeout
                )
            except Exception:
                # 如果 httpx 有问题，回退到基础版本
                pass
        
        # 使用基础版本（兼容旧版本）
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout
        )
        
    except ImportError:
        raise ImportError("请安装 OpenAI 库: pip install openai>=1.0.0")


import streamlit as st
import pandas as pd
import time
import psutil
import os
import logging
from rag_engine import IntelRAG
from agent_core import Text2SQLAgent
from utils import load_config, save_config, load_history, create_new_session, delete_session, update_session_messages

# 配置日志记录
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
from visualization_engine import viz_engine
from recommendation_engine import recommendation_engine
from export_manager import export_manager
from performance_monitor import performance_monitor
from data_filter import data_filter
from anomaly_detector import anomaly_detector
from chart_key_utils import generate_sidebar_chart_key, generate_history_chart_key, generate_query_chart_key, create_chart_with_key

# 通用硬件优化系统集成
try:
    from universal_hardware_optimizer import (
        get_optimization_status, 
        optimize_query_performance, 
        universal_optimizer,
        HardwareVendor
    )
    HARDWARE_OPTIMIZATION_AVAILABLE = True
    hw_status = get_optimization_status()
    if hw_status['enabled']:
        vendor = hw_status.get('vendor', 'Unknown')
        print(f"✅ {vendor}硬件优化系统已加载")
    else:
        print("⚠️ 硬件优化系统不可用")
except ImportError as e:
    HARDWARE_OPTIMIZATION_AVAILABLE = False
    print(f"⚠️ 硬件优化系统不可用: {e}")

# 🧠 Prompt模板系统集成
try:
    from prompt_template_system import PromptTemplateManager, PromptMode, LLMProvider
    from prompt_config_ui import PromptConfigUI
    from prompt_integration import EnhancedPromptBuilder
    PROMPT_TEMPLATE_AVAILABLE = True
    print("✅ Prompt模板系统已加载")
except ImportError as e:
    PROMPT_TEMPLATE_AVAILABLE = False
    print(f"⚠️ Prompt模板系统不可用: {e}")

# 🧠 上下文记忆系统集成
try:
    from context_memory_integration import (
        get_context_integration, 
        integrate_with_messages, 
        update_context_after_response,
        render_context_ui
    )
    CONTEXT_MEMORY_AVAILABLE = True
    print("✅ 上下文记忆系统已加载")
except ImportError as e:
    CONTEXT_MEMORY_AVAILABLE = False
    print(f"⚠️ 上下文记忆系统不可用: {e}")

# 技术卓越性集成系统 - 后端功能启用，前端UI禁用
try:
    from technical_excellence_integration import (
        get_technical_excellence_manager,
        optimize_operation,
        render_technical_excellence_ui,
        get_technical_recommendations
    )
    TECHNICAL_EXCELLENCE_AVAILABLE = True
    tech_manager = get_technical_excellence_manager()
    tech_status = tech_manager.get_technical_status()
    # 技术卓越性后端系统已加载，评分: {tech_status.overall_score:.1f}% ({tech_status.maturity_level})
except ImportError as e:
    TECHNICAL_EXCELLENCE_AVAILABLE = False
    print(f"⚠️ 技术卓越性系统不可用: {e}")

# 独立控制前端UI显示
TECHNICAL_EXCELLENCE_UI_ENABLED = False  # 前端UI面板禁用

# 性能优化配置
st.set_page_config(
    page_title="Intel® DeepInsight", 
    layout="wide", 
    page_icon="assets/团队Logo.png",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Intel® DeepInsight - 基于OpenVINO™的智能零售决策系统"
    }
)

# 缓存优化 - 增加TTL和更大的缓存
@st.cache_data(ttl=3600, max_entries=50)
def load_cached_data(file_path):
    """缓存数据加载"""
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

@st.cache_data(ttl=1800, max_entries=20)
def get_chart_recommendations(df_shape, columns):
    """缓存图表推荐"""
    return viz_engine.get_chart_options_cached(df_shape, columns)

# --- CSS 美化与样式定义 (已修复按钮高度问题 + 移动端优化) ---
st.markdown("""
<style>
    /* 全局背景 */
    .stApp { background-color: #f8f9fa; }
    
    /* 主内容区域优化 */
    .main .block-container {
        padding-top: 1rem !important;
        max-width: 1200px !important;
    }
    
    /* 聊天气泡优化 */
    .stChatMessage { 
        padding: 1.2rem; 
        border-radius: 15px; 
        border: 1px solid #eef0f3; 
        background: white; 
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
        margin-bottom: 10px;
    }
    
    /* 标题样式增强 */
    h5 {
        color: #0068B5; 
        font-weight: 600; 
        margin-top: 20px !important; 
        margin-bottom: 10px !important;
        display: flex;
        align-items: center;
    }
    
    /* 上下文记忆状态指示器 */
    .context-status {
        background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%);
        padding: 8px 12px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 10px 0;
        font-size: 0.85em;
        color: #155724;
    }
    
    .context-disabled {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left-color: #ffc107;
        color: #856404;
    }
    
    /* 思维链持久化样式 - 增强版 */
    .thought-persist {
        background: linear-gradient(135deg, #f0f7ff 0%, #e8f4fd 100%); 
        padding: 16px 20px; 
        border-radius: 12px; 
        font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace; 
        font-size: 0.88em; 
        border-left: 5px solid #0068B5; 
        margin-bottom: 18px;
        color: #2c3e50;
        white-space: pre-wrap;
        line-height: 1.6;
        box-shadow: 0 3px 10px rgba(0,104,181,0.1);
        position: relative;
    }
    
    .thought-persist::before {
        content: "🧠 AI思考过程";
        position: absolute;
        top: -8px;
        left: 15px;
        background: #0068B5;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.7em;
        font-weight: 600;
    }
    
    /* 实时思考流样式 - 增强版 */
    .thought-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
        padding: 14px 16px; 
        border-radius: 10px; 
        font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace; 
        font-size: 0.86em; 
        border-left: 4px solid #6c757d; 
        margin: 12px 0;
        white-space: pre-wrap; 
        color: #495057;
        line-height: 1.5;
        box-shadow: 0 2px 6px rgba(108,117,125,0.1);
        animation: fadeInUp 0.3s ease-out;
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* 加载状态优化 */
    .stStatus > div {
        border-radius: 10px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
    }
    
    /* 成功反馈动画 */
    .success-feedback {
        animation: successPulse 0.6s ease-out;
    }
    
    @keyframes successPulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); background-color: #d4edda; }
        100% { transform: scale(1); }
    }
    
    /* 操作确认样式 */
    .confirm-action {
        background: #fff3cd !important;
        border: 2px solid #ffc107 !important;
        border-radius: 8px !important;
        animation: confirmShake 0.5s ease-out;
    }
    
    @keyframes confirmShake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-2px); }
        75% { transform: translateX(2px); }
    }

    /* 侧边栏监控卡片 */
    .monitor-box {
        background: white; padding: 15px; border-radius: 10px; border: 1px solid #eee;
        font-size: 0.85rem; margin-top: 20px; line-height: 1.8;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .metric-row { display: flex; justify-content: space-between; border-bottom: 1px dashed #eee; padding-bottom: 6px; margin-bottom: 6px; }
    .metric-val { font-weight: bold; font-family: monospace; }
    
    #MainMenu {visibility: hidden;}

    /* ==================================================================== */
    /* 🔥【核心修复】强制统一示例问题按钮的高度与换行 🔥 */
    /* ==================================================================== */
    section.main div[data-testid="column"] button {
        height: 100px !important;        /* 强制固定高度，确保所有卡片一样高 */
        min-height: 100px !important;    /* 最小高度保护 */
        white-space: normal !important;  /* 强制允许文字换行 */
        word-wrap: break-word !important; /*防止长单词溢出 */
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        line-height: 1.4 !important;     /* 调整行高，让多行文字不拥挤 */
        padding: 5px 10px !important;    /* 内部边距 */
    }
    
    /* 鼠标悬停时的微效 */
    section.main div[data-testid="column"] button:hover {
        border-color: #0068B5 !important;
        color: #0068B5 !important;
        background-color: #f0f7ff !important;
    }
    
    /* ==================================================================== */
    /* 📱 移动端响应式优化 - 增强版 */
    /* ==================================================================== */
    
    /* 大屏设备 (1920px+) */
    @media screen and (min-width: 1920px) {
        .main .block-container {
            max-width: 1400px !important;
        }
    }
    
    /* 平板设备 (768px - 1024px) */
    @media screen and (max-width: 1024px) {
        .stSidebar {
            width: 280px !important;
        }
        
        .main .block-container {
            max-width: 100% !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
        }
        
        .monitor-box {
            font-size: 0.8rem;
            padding: 12px;
        }
        
        section.main div[data-testid="column"] button {
            height: 90px !important;
            min-height: 90px !important;
            font-size: 0.9em !important;
        }
        
        /* 图表容器适配 */
        .js-plotly-plot {
            max-width: 100% !important;
        }
        
        /* 输入框优化 */
        .stTextInput input, .stTextArea textarea {
            font-size: 16px !important; /* 防止iOS自动缩放 */
        }
    }
    
    /* 移动设备 (最大宽度 768px) */
    @media screen and (max-width: 768px) {
        /* 侧边栏移动端优化 */
        .stSidebar {
            width: 85% !important;
            max-width: 320px !important;
            position: fixed !important;
            left: 0 !important;
            top: 0 !important;
            height: 100vh !important;
            z-index: 999999 !important;
            box-shadow: 2px 0 10px rgba(0,0,0,0.3) !important;
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        
        /* 侧边栏收起状态 */
        .stSidebar[aria-expanded="false"] {
            transform: translateX(-100%) !important;
        }
        
        /* 侧边栏展开状态 */
        .stSidebar[aria-expanded="true"] {
            transform: translateX(0) !important;
        }
        
        /* 侧边栏遮罩层 */
        .stSidebar::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(0, 0, 0, 0.5);
            z-index: -1;
            opacity: 0;
            transition: opacity 0.3s ease;
            pointer-events: none;
        }
        
        .stSidebar[aria-expanded="true"]::before {
            opacity: 1;
            pointer-events: auto;
        }
        
        /* 展开按钮优化 */
        [data-testid="collapsedControl"] {
            position: fixed !important;
            left: 10px !important;
            top: 10px !important;
            z-index: 999997 !important;
            background: white !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
            border-radius: 8px !important;
            padding: 10px !important;
            width: 44px !important;
            height: 44px !important;
        }
        
        [data-testid="collapsedControl"]:active {
            transform: scale(0.95) !important;
        }
        
        /* 主内容区域 */
        .main .block-container {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            padding-top: 0.5rem !important;
            max-width: 100% !important;
        }
        
        /* 移动端按钮优化 */
        section.main div[data-testid="column"] button {
            height: 75px !important;
            min-height: 75px !important;
            font-size: 0.85em !important;
            padding: 8px !important;
            margin-bottom: 8px !important;
        }
        
        /* 聊天消息优化 */
        .stChatMessage {
            padding: 0.9rem !important;
            margin-bottom: 8px !important;
            border-radius: 12px !important;
        }
        
        /* 聊天输入框优化 */
        .stChatInput {
            position: sticky !important;
            bottom: 0 !important;
            background: white !important;
            padding: 10px 0 !important;
            z-index: 100 !important;
        }
        
        .stChatInput input {
            font-size: 16px !important; /* 防止iOS自动缩放 */
        }
        
        /* 监控面板移动端优化 */
        .monitor-box {
            font-size: 0.75rem;
            padding: 10px;
            margin-top: 15px;
        }
        
        .metric-row {
            padding-bottom: 4px;
            margin-bottom: 4px;
            font-size: 0.85em;
        }
        
        /* 表格响应式 */
        .dataframe {
            font-size: 0.75em !important;
            overflow-x: auto !important;
            display: block !important;
        }
        
        .dataframe table {
            min-width: 100% !important;
        }
        
        /* 图表容器优化 */
        .js-plotly-plot {
            width: 100% !important;
            height: auto !important;
            min-height: 300px !important;
        }
        
        /* 标题优化 */
        h1 {
            font-size: 1.5rem !important;
            line-height: 1.3 !important;
        }
        
        h2 {
            font-size: 1.3rem !important;
        }
        
        h3 {
            font-size: 1.1rem !important;
        }
        
        /* 展开器优化 */
        .streamlit-expanderHeader {
            font-size: 0.9em !important;
            padding: 10px !important;
        }
        
        /* 选择框优化 */
        .stSelectbox, .stMultiSelect {
            font-size: 0.9em !important;
        }
        
        /* 滑块优化 */
        .stSlider {
            padding: 10px 0 !important;
        }
    }
    
    /* 小屏手机 (最大宽度 480px) */
    @media screen and (max-width: 480px) {
        /* 主内容区域 */
        .main .block-container {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
            padding-top: 0.25rem !important;
        }
        
        /* 超小屏按钮 */
        section.main div[data-testid="column"] button {
            height: 65px !important;
            min-height: 65px !important;
            font-size: 0.8em !important;
            padding: 6px !important;
            margin-bottom: 6px !important;
        }
        
        /* 列布局优化 - 强制单列 */
        .row-widget.stColumns {
            flex-direction: column !important;
            gap: 0 !important;
        }
        
        .row-widget.stColumns > div {
            width: 100% !important;
            margin-bottom: 8px !important;
            padding: 0 !important;
        }
        
        /* 标题字体缩小 */
        h1 {
            font-size: 1.3rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        h2 {
            font-size: 1.15rem !important;
        }
        
        h3, h4, h5 {
            font-size: 1rem !important;
        }
        
        /* 思维链样式移动端优化 */
        .thought-persist, .thought-box {
            font-size: 0.7em !important;
            padding: 8px 10px !important;
            margin: 8px 0 !important;
            line-height: 1.4 !important;
        }
        
        .thought-persist::before {
            font-size: 0.65em !important;
            padding: 1px 6px !important;
        }
        
        /* 聊天消息紧凑化 */
        .stChatMessage {
            padding: 0.75rem !important;
            margin-bottom: 6px !important;
        }
        
        /* 监控卡片紧凑化 */
        .monitor-box {
            font-size: 0.7rem;
            padding: 8px;
            margin-top: 10px;
        }
        
        /* 图表高度调整 */
        .js-plotly-plot {
            min-height: 250px !important;
        }
        
        /* 输入框字体大小 */
        input, textarea, select {
            font-size: 16px !important; /* 防止iOS自动缩放 */
        }
        
        /* 按钮文字大小 */
        button {
            font-size: 0.85em !important;
        }
        
        /* 侧边栏宽度 */
        .stSidebar {
            width: 90% !important;
            max-width: 280px !important;
        }
    }
    
    /* 超小屏手机 (最大宽度 360px) */
    @media screen and (max-width: 360px) {
        .main .block-container {
            padding-left: 0.25rem !important;
            padding-right: 0.25rem !important;
        }
        
        section.main div[data-testid="column"] button {
            height: 60px !important;
            min-height: 60px !important;
            font-size: 0.75em !important;
        }
        
        h1 {
            font-size: 1.2rem !important;
        }
        
        .stChatMessage {
            padding: 0.6rem !important;
        }
        
        .thought-persist, .thought-box {
            font-size: 0.65em !important;
            padding: 6px 8px !important;
        }
    }
    
    /* 触摸设备优化 */
    @media (hover: none) and (pointer: coarse) {
        /* 增大触摸目标 */
        button, .stSelectbox, .stTextInput, a {
            min-height: 44px !important;
            min-width: 44px !important;
        }
        
        /* 触摸反馈 */
        button:active {
            transform: scale(0.96);
            transition: transform 0.1s ease;
            background-color: rgba(0, 104, 181, 0.1) !important;
        }
        
        /* 滚动优化 */
        .main, .stSidebar {
            -webkit-overflow-scrolling: touch;
            overflow-y: auto;
        }
        
        /* 禁用悬停效果 */
        button:hover {
            transform: none !important;
        }
        
        /* 链接触摸优化 */
        a {
            padding: 8px !important;
            display: inline-block !important;
        }
    }
    
    /* 横屏模式优化 */
    @media screen and (max-height: 500px) and (orientation: landscape) {
        .stSidebar {
            width: 250px !important;
        }
        
        .main .block-container {
            padding-top: 0.5rem !important;
        }
        
        section.main div[data-testid="column"] button {
            height: 55px !important;
            min-height: 55px !important;
        }
        
        .monitor-box {
            padding: 6px;
            font-size: 0.7rem;
            margin-top: 8px;
        }
        
        .stChatMessage {
            padding: 0.6rem !important;
        }
        
        /* 紧凑化间距 */
        h1, h2, h3, h4, h5 {
            margin-top: 0.5rem !important;
            margin-bottom: 0.5rem !important;
        }
    }
    
    /* 暗色模式支持 (可选) */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #1a1a1a !important;
        }
        
        .stChatMessage {
            background: #2d2d2d !important;
            border-color: #404040 !important;
            color: #e0e0e0 !important;
        }
        
        .monitor-box {
            background: #2d2d2d !important;
            border-color: #404040 !important;
            color: #e0e0e0 !important;
        }
    }
    
    /* 打印样式优化 */
    @media print {
        .stSidebar {
            display: none !important;
        }
        
        .main .block-container {
            max-width: 100% !important;
            padding: 0 !important;
        }
        
        button {
            display: none !important;
        }
        
        .stChatMessage {
            page-break-inside: avoid !important;
        }
    }
    }

    /* ==================================================================== */
    /* ⌨️ 键盘快捷键支持 */
    /* ==================================================================== */
    
    /* 快捷键提示 */
    .keyboard-hint {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: rgba(0,0,0,0.8);
        color: white;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.75em;
        z-index: 1000;
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .keyboard-hint.show {
        opacity: 1;
    }
    
    /* 聚焦输入框样式 */
    .stChatInput input:focus {
        border-color: #0068B5 !important;
        box-shadow: 0 0 0 2px rgba(0,104,181,0.2) !important;
    }
    
    /* 按钮聚焦样式 */
    button:focus {
        outline: 2px solid #0068B5 !important;
        outline-offset: 2px !important;
    }
    
    /* ==================================================================== */
    /* 🎯 交互反馈增强 */
    /* ==================================================================== */
    
    /* 按钮点击反馈 */
    button:active {
        transform: scale(0.98);
        transition: transform 0.1s ease;
    }
    
    /* 悬停效果增强 */
    button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
        transition: all 0.2s ease;
    }
    
    /* 数据表格交互增强 */
    .dataframe tbody tr:hover {
        background-color: #f8f9fa !important;
        transform: scale(1.01);
        transition: all 0.2s ease;
    }
    
    /* 展开器动画 */
    .streamlit-expanderHeader {
        transition: all 0.3s ease !important;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #f0f7ff !important;
    }
    
    /* 进度条美化 */
    .stProgress > div > div {
        background: linear-gradient(90deg, #0068B5, #00a8ff) !important;
        border-radius: 10px !important;
    }

</style>

<!-- 移动端侧边栏控制和键盘快捷键JavaScript -->
<script>
(function() {
    'use strict';
    
    let hintTimeout;
    
    // ========================================
    // 📱 移动端侧边栏控制 - 简化版
    // ========================================
    function setupMobileSidebar() {
        const isMobile = window.innerWidth <= 768;
        
        if (!isMobile) return;
        
        // 等待Streamlit完全加载
        setTimeout(function() {
            const sidebar = document.querySelector('[data-testid="stSidebar"]');
            const collapseButton = document.querySelector('[data-testid="collapsedControl"]');
            
            if (!sidebar) return;
            
            // 创建遮罩层（如果不存在）
            let overlay = document.getElementById('mobile-sidebar-overlay');
            if (!overlay) {
                overlay = document.createElement('div');
                overlay.id = 'mobile-sidebar-overlay';
                overlay.style.cssText = `
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100vw;
                    height: 100vh;
                    background: rgba(0, 0, 0, 0.5);
                    z-index: 999998;
                    display: none;
                    opacity: 0;
                    transition: opacity 0.3s ease;
                `;
                document.body.appendChild(overlay);
            }
            
            // 检查侧边栏状态
            function isSidebarOpen() {
                const sidebarContent = sidebar.querySelector('[data-testid="stSidebarContent"]');
                return sidebarContent && window.getComputedStyle(sidebarContent).display !== 'none';
            }
            
            // 关闭侧边栏
            function closeSidebar() {
                // 查找并点击侧边栏内的收起按钮
                const closeBtn = sidebar.querySelector('button[kind="header"]');
                if (closeBtn) {
                    closeBtn.click();
                }
                
                // 隐藏遮罩层
                overlay.style.display = 'none';
                overlay.style.opacity = '0';
                document.body.style.overflow = '';
            }
            
            // 打开侧边栏
            function openSidebar() {
                // 显示遮罩层
                overlay.style.display = 'block';
                setTimeout(() => {
                    overlay.style.opacity = '1';
                }, 10);
                document.body.style.overflow = 'hidden';
            }
            
            // 监听遮罩层点击
            overlay.onclick = closeSidebar;
            
            // 监听展开按钮点击
            if (collapseButton) {
                collapseButton.addEventListener('click', function() {
                    setTimeout(openSidebar, 100);
                });
            }
            
            // 监听侧边栏内的收起按钮
            const sidebarCloseBtn = sidebar.querySelector('button[kind="header"]');
            if (sidebarCloseBtn) {
                sidebarCloseBtn.addEventListener('click', function() {
                    setTimeout(closeSidebar, 100);
                });
            }
            
            // 监听侧边栏内的链接和选项点击（点击后自动关闭）
            sidebar.addEventListener('click', function(e) {
                const target = e.target;
                if (target.tagName === 'A' || 
                    target.closest('[role="option"]') ||
                    target.closest('button[kind="secondary"]')) {
                    setTimeout(closeSidebar, 300);
                }
            });
            
            // 初始状态检查
            if (isSidebarOpen()) {
                openSidebar();
            } else {
                closeSidebar();
            }
            
            // 监听侧边栏状态变化
            const sidebarObserver = new MutationObserver(function() {
                if (isSidebarOpen()) {
                    openSidebar();
                } else {
                    overlay.style.display = 'none';
                    overlay.style.opacity = '0';
                }
            });
            
            const sidebarContent = sidebar.querySelector('[data-testid="stSidebarContent"]');
            if (sidebarContent) {
                sidebarObserver.observe(sidebarContent, {
                    attributes: true,
                    attributeFilter: ['style']
                });
            }
        }, 500);
    }
    
    // 初始化
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setupMobileSidebar);
    } else {
        setupMobileSidebar();
    }
    
    // 监听窗口大小变化
    let resizeTimer;
    window.addEventListener('resize', function() {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(setupMobileSidebar, 250);
    });
    
    // 监听Streamlit重新渲染
    window.addEventListener('load', function() {
        setTimeout(setupMobileSidebar, 1000);
    });
    
    // ========================================
    // ⌨️ 键盘快捷键
    // ========================================
    
    // 显示快捷键提示
    function showKeyboardHint(text) {
        let hint = document.querySelector('.keyboard-hint');
        if (!hint) {
            hint = document.createElement('div');
            hint.className = 'keyboard-hint';
            document.body.appendChild(hint);
        }
        hint.textContent = text;
        hint.classList.add('show');
        
        clearTimeout(hintTimeout);
        hintTimeout = setTimeout(() => {
            hint.classList.remove('show');
        }, 2000);
    }
    
    // 键盘事件监听
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + Enter: 发送消息
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            const chatInput = document.querySelector('.stChatInput input');
            if (chatInput && chatInput.value.trim()) {
                const submitBtn = document.querySelector('.stChatInput button');
                if (submitBtn) {
                    submitBtn.click();
                    showKeyboardHint('消息已发送 (Ctrl+Enter)');
                }
            }
        }
        
        // Ctrl/Cmd + N: 新建会话
        if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
            e.preventDefault();
            const newSessionBtn = document.querySelector('button[title*="新建"]');
            if (newSessionBtn) {
                newSessionBtn.click();
                showKeyboardHint('新建会话 (Ctrl+N)');
            }
        }
        
        // Ctrl/Cmd + /: 聚焦搜索
        if ((e.ctrlKey || e.metaKey) && e.key === '/') {
            e.preventDefault();
            const chatInput = document.querySelector('.stChatInput input');
            if (chatInput) {
                chatInput.focus();
                showKeyboardHint('聚焦输入框 (Ctrl+/)');
            }
        }
        
        // Esc: 清除输入
        if (e.key === 'Escape') {
            const chatInput = document.querySelector('.stChatInput input');
            if (chatInput && chatInput.value) {
                chatInput.value = '';
                showKeyboardHint('输入已清除 (Esc)');
            }
        }
        
        // F1: 显示快捷键帮助
        if (e.key === 'F1') {
            e.preventDefault();
            showKeyboardHint('快捷键: Ctrl+Enter发送, Ctrl+N新建, Ctrl+/聚焦, Esc清除');
        }
    });
    
    // 添加成功反馈动画
    function addSuccessFeedback(element) {
        element.classList.add('success-feedback');
        setTimeout(() => {
            element.classList.remove('success-feedback');
        }, 600);
    }
    
    // 监听按钮点击，添加反馈
    document.addEventListener('click', function(e) {
        if (e.target.tagName === 'BUTTON') {
            addSuccessFeedback(e.target);
        }
    });
});
</script>
""", unsafe_allow_html=True)

# --- 状态管理 ---
if "config" not in st.session_state: st.session_state.config = load_config()
if "history" not in st.session_state: st.session_state.history = load_history()
if "last_total_latency" not in st.session_state: st.session_state.last_total_latency = 0.0
if "last_rag_latency" not in st.session_state: st.session_state.last_rag_latency = 0.0
if "prompt_trigger" not in st.session_state: st.session_state.prompt_trigger = None
if "agent_loaded" not in st.session_state: st.session_state.agent_loaded = False

# 🧠 Prompt模板系统状态初始化
if PROMPT_TEMPLATE_AVAILABLE:
    if "prompt_mode" not in st.session_state:
        st.session_state.prompt_mode = "flexible"
    if "show_advanced_prompt_config" not in st.session_state:
        st.session_state.show_advanced_prompt_config = False

# 🧠 初始化上下文记忆设置 - 从配置文件加载
if CONTEXT_MEMORY_AVAILABLE:
    try:
        # 首先确保基本的 session_state 属性存在
        if 'context_memory_enabled' not in st.session_state:
            st.session_state.context_memory_enabled = True
        if 'context_memory_depth' not in st.session_state:
            st.session_state.context_memory_depth = 5
        if 'context_memory_strength' not in st.session_state:
            st.session_state.context_memory_strength = 0.7
        if 'context_auto_clean' not in st.session_state:
            st.session_state.context_auto_clean = True
        if 'context_persist_memory' not in st.session_state:
            st.session_state.context_persist_memory = False
        if 'context_privacy_mode' not in st.session_state:
            st.session_state.context_privacy_mode = False
            
        # 然后初始化上下文集成（这会从配置文件加载并覆盖默认值）
        context_integration = get_context_integration()
        # 这会自动加载保存的设置到 session_state
    except Exception as e:
        print(f"上下文记忆设置加载失败: {e}")
        # 使用默认设置
        if 'context_memory_enabled' not in st.session_state:
            st.session_state.context_memory_enabled = True

# 确保有当前会话ID
# 修改优化：系统启动或刷新时，强制创建一个新会话，确保显示“欢迎页”
if "current_session_id" not in st.session_state or st.session_state.current_session_id not in st.session_state.history:
    # 直接调用新建会话逻辑
    sid, hist = create_new_session(st.session_state.history)
    st.session_state.history = hist
    st.session_state.current_session_id = sid

# 🧠 处理高级Prompt配置页面
if PROMPT_TEMPLATE_AVAILABLE and st.session_state.get('show_advanced_prompt_config', False):
    st.markdown("## 🔧 Prompt模板高级配置")
    
    # 返回按钮
    if st.button("🔙 返回主界面", key="back_to_main"):
        st.session_state.show_advanced_prompt_config = False
        st.rerun()
    
    # 渲染高级配置界面
    try:
        config_ui = st.session_state.prompt_config_ui
        
        # 标签页
        tab1, tab2, tab3, tab4 = st.tabs([
            "📝 业务上下文", "📚 术语词典", "💡 示例查询", "👁️ Prompt预览"
        ])
        
        with tab1:
            config_ui.render_business_context_config()
        
        with tab2:
            config_ui.render_term_dictionary_config()
        
        with tab3:
            config_ui.render_example_queries_config()
        
        with tab4:
            config_ui.render_prompt_preview()
    
    except Exception as e:
        st.error(f"高级配置界面错误: {e}")
        if st.button("🔙 返回主界面", key="back_to_main_error"):
            st.session_state.show_advanced_prompt_config = False
            st.rerun()
    
    # 停止执行，不显示正常的主界面
    st.stop()

# --- 侧边栏配置 ---
with st.sidebar:
    # Intel Logo 和品牌标识
    if os.path.exists("assets/intel.svg"):
        st.image("assets/intel.svg", width=120)
    else:
        st.markdown("### Intel® DeepInsight")
    
    # 🧠 渲染上下文记忆UI
    # 🧠 美化后的上下文记忆系统UI
    if CONTEXT_MEMORY_AVAILABLE:
        with st.expander("🧠 上下文记忆系统", expanded=False):
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
            
            # 内存状态卡片
            
            
            # 显示当前状态
            col_status, col_actions = st.columns([3, 2])
            with col_status:
                st.markdown("**📊 当前状态**")
                
                # 获取当前状态 - 确保从session_state获取最新值
                memory_enabled = st.session_state.get('context_memory_enabled', True)
                
                # 状态标签 - 使用动态更新
                if memory_enabled:
                    st.markdown('<span class="context-status-badge context-status-enabled">✅ 已启用</span>', unsafe_allow_html=True)
                    st.caption("AI将记住对话历史，提供更智能的回复")
                else:
                    st.markdown('<span class="context-status-badge context-status-disabled">⏸️ 已禁用</span>', unsafe_allow_html=True)
                    st.caption("AI将不会记住对话历史")
            
            with col_actions:
                st.markdown("**⚙️ 操作**")
                
                # 切换开关 - 使用当前状态
                current_memory_enabled = st.session_state.get('context_memory_enabled', True)
                toggle_label = "禁用记忆" if current_memory_enabled else "启用记忆"
                toggle_icon = "⏸️" if current_memory_enabled else "▶️"
                
                if st.button(f"{toggle_icon} {toggle_label}", 
                            use_container_width=True,
                            key="toggle_memory_btn"):
                    # 切换状态
                    new_state = not current_memory_enabled
                    st.session_state.context_memory_enabled = new_state
                    
                    # 立即保存设置到配置文件
                    if CONTEXT_MEMORY_AVAILABLE:
                        try:
                            context_integration = get_context_integration()
                            context_integration._save_memory_settings()
                            
                            # 显示操作反馈
                            if new_state:
                                st.success("✅ 上下文记忆已启用")
                            else:
                                st.info("⏸️ 上下文记忆已禁用")
                                
                        except Exception as e:
                            st.error(f"保存设置失败: {e}")
                    
                    # 强制刷新页面以更新所有UI组件
                    time.sleep(0.5)  # 短暂延迟确保设置已保存
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 配置设置
            
            st.markdown("**🔧 记忆配置**")
            
            # 记忆深度设置
            st.markdown('<div class="context-setting-label"><i>📏</i> 记忆深度</div>', unsafe_allow_html=True)
            
            # 获取或初始化记忆深度设置
            if "context_memory_depth" not in st.session_state:
                st.session_state.context_memory_depth = 5
            
            memory_depth = st.slider(
                "保留的对话轮数",
                min_value=1,
                max_value=20,
                value=st.session_state.context_memory_depth,
                key="memory_depth_slider",
                label_visibility="collapsed",
                help="设置AI能够记住的最近对话轮数。较大的值会让AI记住更多历史，但可能影响响应速度。推荐值：3-8轮"
            )
            if memory_depth != st.session_state.context_memory_depth:
                st.session_state.context_memory_depth = memory_depth
                # 保存设置
                if CONTEXT_MEMORY_AVAILABLE:
                    try:
                        context_integration = get_context_integration()
                        context_integration._save_memory_settings()
                    except Exception:
                        pass
            
            st.markdown('<div class="context-tooltip">💡 <strong>算法说明</strong>: 系统使用滑动窗口算法保留最近N轮对话，超出范围的对话将被自动清理。较大的值提供更好的上下文连贯性，但会增加计算开销。</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 记忆强度设置
            st.markdown('<div class="context-setting-label"><i>💪</i> 记忆强度</div>', unsafe_allow_html=True)
            
            # 获取或初始化记忆强度设置
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
                help="设置历史对话对当前回答的影响程度。0.0表示完全忽略历史，1.0表示完全依赖历史。推荐值：0.5-0.8"
            )
            if memory_strength != st.session_state.context_memory_strength:
                st.session_state.context_memory_strength = memory_strength
                # 保存设置
                if CONTEXT_MEMORY_AVAILABLE:
                    try:
                        context_integration = get_context_integration()
                        context_integration._save_memory_settings()
                    except Exception:
                        pass
            
            st.markdown('<div class="context-tooltip">💡 <strong>算法说明</strong>: 使用加权融合算法，将历史上下文与当前输入按此权重比例混合。权重越高，AI越倾向于基于历史信息回答；权重越低，AI越专注于当前问题。</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 高级设置
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
                    if CONTEXT_MEMORY_AVAILABLE:
                        try:
                            context_integration = get_context_integration()
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
                    help="将对话记忆保存到本地文件，下次启动时恢复（当前版本已默认启用SQLite持久化）",
                    key="persist_memory_checkbox"
                )
                if persist_memory != st.session_state.get('context_persist_memory', False):
                    st.session_state.context_persist_memory = persist_memory
                    if CONTEXT_MEMORY_AVAILABLE:
                        try:
                            context_integration = get_context_integration()
                            context_integration._save_memory_settings()
                            if persist_memory:
                                st.info("💾 记忆持久化已启用，对话数据将保存到 streamlit_context_memory.db")
                            else:
                                st.info("⚠️ 注意：禁用持久化不会删除已保存的数据，只是不再保存新的对话")
                        except Exception as e:
                            st.error(f"设置持久化失败: {e}")
                
                # 隐私模式
                privacy_mode = st.checkbox(
                    "隐私模式（不保存敏感信息）",
                    value=st.session_state.get('context_privacy_mode', False),
                    help="启用后，系统会自动过滤邮箱、电话、身份证等敏感信息",
                    key="privacy_mode_checkbox"
                )
                if privacy_mode != st.session_state.get('context_privacy_mode', False):
                    st.session_state.context_privacy_mode = privacy_mode
                    if CONTEXT_MEMORY_AVAILABLE:
                        try:
                            context_integration = get_context_integration()
                            context_integration._save_memory_settings()
                            if privacy_mode:
                                st.success("🔒 隐私模式已启用，敏感信息将被自动过滤")
                            else:
                                st.info("🔓 隐私模式已禁用，对话内容将完整保存")
                        except Exception as e:
                            st.error(f"设置隐私模式失败: {e}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 记忆统计信息 - 使用当前状态
            current_memory_enabled = st.session_state.get('context_memory_enabled', True)
            if current_memory_enabled:
                
                st.markdown("**📈 记忆统计**")
                
                # 获取实际的统计信息
                if CONTEXT_MEMORY_AVAILABLE:
                    try:
                        context_integration = get_context_integration()
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
                        # 降级显示
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        with col_stat1:
                            st.metric("已保存对话", "0轮")
                        with col_stat2:
                            st.metric("记忆容量", "0%")
                        with col_stat3:
                            st.metric("关联精度", "0%")
                        st.caption(f"⚠️ 统计数据获取失败: {e}")
                else:
                    # 模拟统计信息
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("已保存对话", "0轮")
                    with col_stat2:
                        st.metric("记忆容量", "0%")
                    with col_stat3:
                        st.metric("关联精度", "0%")
                
                # 清理记忆按钮
                if st.button("🗑️ 清理所有记忆", use_container_width=True, type="secondary"):
                    # 使用确认对话框
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
                            if CONTEXT_MEMORY_AVAILABLE:
                                try:
                                    context_integration = get_context_integration()
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
                            else:
                                st.error("❌ 记忆系统不可用")
                            st.session_state.confirm_clear_memory = False
                    with col_confirm2:
                        if st.button("❌ 取消", use_container_width=True, key="cancel_clear_btn"):
                            st.session_state.confirm_clear_memory = False
                            st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # 使用提示
            st.info("💡 **使用提示**: 启用上下文记忆可以让AI更好地理解多轮对话的上下文，提供更连贯、更准确的回答。")
            
            # 调用原始的render_context_ui函数以确保功能完整性
            # 注意：我们保留了原始的功能调用，但将其包装在隐藏的容器中
            # 这样既保留了功能，又提供了美观的UI
            # with st.container():
            #     st.markdown('<div style="display: none;">', unsafe_allow_html=True)
            #     render_context_ui()
            #     st.markdown('</div>', unsafe_allow_html=True)
    else:
        # 如果上下文记忆系统不可用，显示友好的提示
        with st.expander("🧠 上下文记忆系统", expanded=False):
            st.warning("上下文记忆系统当前不可用")
            st.info("要启用上下文记忆功能，请确保已正确安装并配置相关模块。")
    
    # 监控面板占位符
    monitor_placeholder = st.empty()
    
    # 🧠 Prompt模板配置面板
    if PROMPT_TEMPLATE_AVAILABLE:
        with st.expander("🧠 Prompt模板配置", expanded=False):
            try:
                # 初始化Prompt配置UI
                if 'prompt_config_ui' not in st.session_state:
                    st.session_state.prompt_config_ui = PromptConfigUI()
                
                config_ui = st.session_state.prompt_config_ui
                
                # 获取配置摘要
                summary = config_ui.manager.get_config_summary()
                
                # 只在需要时刷新统计数据，不重新加载配置
                if st.session_state.get('prompt_config_updated', 0) > st.session_state.get('last_summary_update', 0):
                    # 只刷新统计数据，不重新加载配置文件
                    summary = config_ui.manager.get_config_summary()
                    st.session_state.last_summary_update = time.time()
                
                # 配置状态显示
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "业务上下文", 
                        "已配置" if summary['business_context_configured'] else "未配置",
                        f"{summary['business_context_length']}/2000字符"
                    )
                with col2:
                    st.metric("术语词典", f"{summary['term_dictionary_size']}个术语")
                
                st.metric("示例查询", f"{summary['example_queries_count']}个示例")
                
                # 快速配置选项
                st.markdown("**⚙️ 快速配置**")
                
                # LLM模式选择
                current_mode = st.session_state.get('prompt_mode', 'flexible')
                prompt_mode = st.selectbox(
                    "查询策略",
                    options=['professional', 'flexible'],
                    index=0 if current_mode == 'professional' else 1,
                    format_func=lambda x: "标准查询 (严格匹配)" if x == 'professional' else "智能查询 (语义理解)",
                    help="标准查询：严格按照数据库结构生成精确SQL；智能查询：理解业务语义，提供更灵活的查询方案",
                    key="prompt_mode_select"
                )
                
                if prompt_mode != current_mode:
                    st.session_state.prompt_mode = prompt_mode
                    mode_name = "标准查询" if prompt_mode == 'professional' else "智能查询"
                    st.success(f"✅ 已切换到{mode_name}策略")
                
                # 业务上下文快速配置
                st.markdown("**📝 业务上下文**")
                
                current_context = config_ui.manager.business_context
                
                # 行业术语输入
                industry_terms = st.text_area(
                    "行业术语 (用逗号分隔)",
                    value=current_context.industry_terms,
                    height=60,
                    placeholder="例如：零售业、电商、供应链、库存周转率、客单价",
                    help="输入您所在行业的专业术语，系统会自动识别和解释"
                )
                
                # 分析重点
                analysis_focus = st.text_input(
                    "分析重点",
                    value=current_context.analysis_focus,
                    placeholder="例如：销售分析、客户分析、产品分析、运营效率",
                    help="指明您最关注的分析维度"
                )
                
                # 保存按钮
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 保存配置", use_container_width=True, key="save_prompt_config"):
                        try:
                            # 保存时只更新用户修改的字段，保留其他字段的现有值
                            config_ui.manager.update_business_context(
                                industry_terms=industry_terms,
                                analysis_focus=analysis_focus,
                                # 保留现有的business_rules和data_characteristics
                                business_rules=current_context.business_rules,
                                data_characteristics=current_context.data_characteristics
                            )
                            st.success("✅ Prompt配置已保存")
                            # 强制刷新统计数据
                            st.session_state.prompt_config_updated = time.time()
                            time.sleep(0.5)
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ 保存失败: {e}")
                
                with col2:
                    if st.button("🔧 高级配置", use_container_width=True, key="advanced_prompt_config"):
                        st.session_state.show_advanced_prompt_config = True
                        st.rerun()
                
                # 术语词典快速导入
                st.markdown("**📚 术语词典**")
                uploaded_terms = st.file_uploader(
                    "上传术语词典 (CSV格式)",
                    type=['csv'],
                    help="CSV文件需包含 'term' 和 'explanation' 两列",
                    key="terms_upload"
                )
                
                if uploaded_terms is not None:
                    try:
                        import pandas as pd
                        df = pd.read_csv(uploaded_terms)
                        
                        if 'term' in df.columns and 'explanation' in df.columns:
                            # 使用固定的文件名确保一致性
                            csv_path = "data/uploaded_terms_user_uploaded_terms.csv"
                            os.makedirs("data", exist_ok=True)
                            with open(csv_path, 'wb') as f:
                                f.write(uploaded_terms.getbuffer())
                            
                            config_ui.manager.load_term_dictionary(csv_path)
                            st.success(f"✅ 成功导入 {len(df)} 个术语")
                            # 更新统计数据
                            st.session_state.prompt_config_updated = time.time()
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("❌ CSV文件必须包含 'term' 和 'explanation' 列")
                    except Exception as e:
                        st.error(f"❌ 导入失败: {e}")
                
                # 示例查询快速添加
                st.markdown("**💡 示例查询**")
                col1, col2 = st.columns([3, 1])
                with col1:
                    new_example = st.text_input(
                        "添加示例查询",
                        placeholder="例如：查看销售额最高的产品",
                        key="new_example_input"
                    )
                with col2:
                    example_category = st.selectbox(
                        "类别",
                        ["销售分析", "客户分析", "产品分析", "运营分析", "财务分析"],
                        key="example_category_select"
                    )
                
                if st.button("➕ 添加示例", key="add_example_btn") and new_example:
                    config_ui.manager.add_example_query(
                        query=new_example,
                        category=example_category,
                        description=f"{example_category}示例"
                    )
                    st.success("✅ 示例查询已添加")
                    # 更新统计数据
                    st.session_state.prompt_config_updated = time.time()
                    time.sleep(0.5)
                    st.rerun()
                
                # 使用提示
                st.info("💡 **使用提示**: Prompt模板系统可以让AI更好地理解您的业务需求，提供更准确的分析结果。")
                
            except Exception as e:
                st.error(f"Prompt模板配置面板错误: {e}")
    else:
        with st.expander("🧠 Prompt模板配置", expanded=False):
            st.warning("Prompt模板系统当前不可用")
            st.info("要启用Prompt模板功能，请确保已正确安装相关模块。")

    # 硬件优化面板
    if HARDWARE_OPTIMIZATION_AVAILABLE:
        optimization_status = get_optimization_status()
        vendor = optimization_status.get('vendor', 'Unknown')
        
        # 根据硬件厂商显示不同的图标和标题
        if vendor == 'Intel':
            panel_title = "🚀 Intel平台优化"
            panel_icon = "🔧"
        elif vendor == 'NVIDIA':
            panel_title = "⚡ NVIDIA平台优化"
            panel_icon = "🎮"
        elif vendor == 'AMD':
            panel_title = "🔥 AMD平台优化"
            panel_icon = "🚀"
        else:
            panel_title = "🔧 硬件平台优化"
            panel_icon = "⚙️"
        
        with st.expander(panel_title, expanded=True):
            try:
                if optimization_status['enabled']:
                    if optimization_status['optimized']:
                        st.success(f"🎯 {vendor}系统已优化")
                        
                        # 显示优化指标
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("CPU提升", optimization_status['cpu_gain'])
                            st.metric("总体加速", optimization_status['overall_speedup'])
                        with col2:
                            st.metric("GPU加速", optimization_status['gpu_speedup'])
                            st.metric("内存效率", optimization_status['memory_efficiency'])
                        
                        # 显示优化次数
                        if 'optimization_count' in optimization_status:
                            st.caption(f"🔄 已优化查询: {optimization_status['optimization_count']} 次")
                        
                    else:
                        # 未优化状态：显示硬件检测信息但不显示性能指标
                        st.info(f"⏳ {vendor}硬件已检测，等待查询以进行优化")
                        
                        # 显示检测到的硬件信息（但不显示性能指标）
                        hw_info = optimization_status.get('hardware_info', {})
                        if hw_info:
                            st.caption(f"💻 检测到: {hw_info.get('cpu_model', 'Unknown')[:30]}...")
                            
                            # 显示硬件特性（检测结果）
                            features = []
                            if hw_info.get('cpu_cores', 0) > 0:
                                features.append(f"{hw_info['cpu_cores']}核")
                            if hw_info.get('has_avx2'):
                                features.append("AVX2支持")
                            
                            # GPU检测结果
                            gpu_features = []
                            if hw_info.get('has_intel_gpu'):
                                gpu_features.append("Intel GPU")
                            if hw_info.get('has_nvidia_gpu'):
                                gpu_features.append("NVIDIA GPU")
                            if hw_info.get('has_amd_gpu'):
                                gpu_features.append("AMD GPU")
                            if hw_info.get('has_cuda'):
                                gpu_features.append("CUDA支持")
                            
                            if features:
                                st.caption(f"🔧 CPU特性: {' | '.join(features)}")
                            if gpu_features:
                                st.caption(f"🎮 GPU特性: {' | '.join(gpu_features)}")
                            
                            st.caption("💡 开始查询后将显示实际优化效果")
                    
                    # 通用硬件信息显示（优化后的详细信息）
                    if optimization_status['optimized']:
                        hw_info = optimization_status.get('hardware_info', {})
                        
                        # 根据硬件厂商显示不同的特性
                        features = []
                        if hw_info.get('cpu_cores', 0) > 0:
                            features.append(f"{hw_info['cpu_cores']}核")
                        if hw_info.get('has_avx2'):
                            features.append("AVX2: ✅")
                        else:
                            features.append("AVX2: ❌")
                        
                        # GPU特性显示
                        gpu_features = []
                        if hw_info.get('has_intel_gpu'):
                            gpu_features.append("Intel GPU: ✅")
                        if hw_info.get('has_nvidia_gpu'):
                            gpu_features.append("NVIDIA GPU: ✅")
                        if hw_info.get('has_amd_gpu'):
                            gpu_features.append("AMD GPU: ✅")
                        if hw_info.get('has_cuda'):
                            gpu_features.append("CUDA: ✅")
                        
                        if features:
                            st.caption(f"🔧 {' | '.join(features)}")
                        if gpu_features:
                            st.caption(f"🎮 {' | '.join(gpu_features)}")
                            
                else:
                    st.warning(f"⚠️ {vendor}优化器未启用")
                    
            except Exception as e:
                st.error(f"硬件优化面板错误: {e}")
    else:
        with st.expander("⚠️ 硬件优化不可用", expanded=False):
            st.warning("硬件优化模块未正确加载，请检查依赖项")
    
    # 技术卓越性面板 - 前端UI已禁用（用户要求界面简洁）
    if TECHNICAL_EXCELLENCE_AVAILABLE and TECHNICAL_EXCELLENCE_UI_ENABLED:
        with st.expander("🏆 技术卓越性状态", expanded=False):
            try:
                tech_status = tech_manager.get_technical_status()
                
                # 总体状态
                if tech_status.overall_score >= 80:
                    st.success(f"🎯 技术评分: {tech_status.overall_score:.1f}% ({tech_status.maturity_level})")
                elif tech_status.overall_score >= 60:
                    st.info(f"📊 技术评分: {tech_status.overall_score:.1f}% ({tech_status.maturity_level})")
                else:
                    st.warning(f"⚠️ 技术评分: {tech_status.overall_score:.1f}% ({tech_status.maturity_level})")
                
                # 模块状态
                col1, col2 = st.columns(2)
                with col1:
                    intel_status = "✅" if tech_status.intel_integration else "❌"
                    st.caption(f"🚀 Intel集成: {intel_status}")
                    
                    arch_status = "✅" if tech_status.enterprise_architecture else "❌"
                    st.caption(f"🏗️ 企业架构: {arch_status}")
                
                with col2:
                    perf_status = "✅" if tech_status.adaptive_performance else "❌"
                    st.caption(f"⚡ 性能优化: {perf_status}")
                    
                    st.caption(f"🔄 优化次数: {tech_manager.operation_count}")
                
                # 优化建议
                recommendations = get_technical_recommendations()
                if recommendations and len(recommendations) > 0:
                    st.caption("💡 优化建议:")
                    for rec in recommendations[:2]:  # 只显示前2个建议
                        st.caption(f"• {rec}")
                        
            except Exception as e:
                st.error(f"技术卓越性面板错误: {e}")
    # UI面板被禁用，但后端功能继续工作
    
    # 性能趋势图
    with st.expander("📈 性能趋势", expanded=False):
        trend_hours = st.selectbox("时间范围", [1, 3, 6, 12, 24], index=0, key="trend_hours")
        if st.button("刷新趋势图", use_container_width=True):
            trend_fig = performance_monitor.create_performance_trend_chart(trend_hours)
            if trend_fig:
                # 生成唯一的图表key，包含时间范围参数
                chart_key = generate_sidebar_chart_key("performance_trend", f"{trend_hours}h")
                st.plotly_chart(trend_fig, use_container_width=True, key=chart_key)
            else:
                st.info("暂无足够的历史数据生成趋势图")



    with st.expander("🧠 模型设置", expanded=False):
        st.markdown("**🔧 SQL生成API配置**")
        api_url = st.text_input("API URL", st.session_state.config["api_base"])
        api_key = st.text_input("API Key", st.session_state.config["api_key"], type="password")
        model = st.text_input("生成模型 (LLM)", st.session_state.config["model_name"])
        
        st.markdown("**🤖 推荐引擎设置**")
        enable_ai_recommendations = st.checkbox(
            "启用AI智能推荐", 
            value=st.session_state.config.get("enable_ai_recommendations", True),
            help="启用后可以使用AI生成智能问题推荐"
        )
        
        if enable_ai_recommendations:
            use_separate_api = st.checkbox(
                "使用独立的推荐API配置",
                value=st.session_state.config.get("recommendation_use_separate_api", False),
                help="启用后推荐功能将使用独立的API配置，否则与SQL生成共用上述API"
            )
            
            if use_separate_api:
                st.markdown("**📡 推荐API独立配置**")
                rec_api_url = st.text_input(
                    "推荐API URL", 
                    st.session_state.config.get("recommendation_api_base", "https://api.deepseek.com")
                )
                rec_api_key = st.text_input(
                    "推荐API Key", 
                    st.session_state.config.get("recommendation_api_key", ""), 
                    type="password"
                )
                rec_model = st.text_input(
                    "推荐模型名称", 
                    st.session_state.config.get("recommendation_model_name", "deepseek-reasoner")
                )
            else:
                st.info("💡 推荐功能将使用上述SQL生成的API配置")
                rec_api_url = api_url
                rec_api_key = api_key
                rec_model = model
        else:
            st.info("💡 禁用后将使用基于规则的备用推荐")
            use_separate_api = False
            rec_api_url = ""
            rec_api_key = ""
            rec_model = ""
        
        st.markdown("**📁 RAG模型配置**")
        rag_path = st.text_input("RAG 模型路径", st.session_state.config.get("model_path", "models/bge-small-ov"))

    with st.expander("🗄️ 数据库连接", expanded=False):
        # 检测数据库类型是否发生变化
        current_db_type = st.session_state.config.get("db_type", "SQLite")
        db_type_options = ["SQLite", "MySQL"]
        current_index = db_type_options.index(current_db_type) if current_db_type in db_type_options else 0
        
        db_type = st.selectbox("类型", db_type_options, index=current_index, key="db_type_selector")
        
        # 检测数据库类型切换
        if db_type != current_db_type:
            # 数据库类型发生变化，更新session_state以触发界面刷新
            st.session_state.config["db_type"] = db_type
            st.rerun()
        
        final_uris = []
        db_path_val = ""
        
        if db_type == "SQLite":
            # 从分离的SQLite配置中获取默认值
            sqlite_config = st.session_state.config.get("sqlite_config", {})
            default_sqlite_path = sqlite_config.get("db_path", "data/ecommerce.db")
            
            db_path_val = st.text_area("文件路径", value=default_sqlite_path)
            for p in db_path_val.split('\n'):
                if p.strip(): final_uris.append(f"sqlite:///{p.strip()}")
        else:
            # MySQL配置 - 从分离的MySQL配置中获取默认值
            mysql_config = st.session_state.config.get("mysql_config", {})
            default_host = mysql_config.get("host", "localhost")
            default_port = mysql_config.get("port", "3306")
            default_user = mysql_config.get("user", "root")
            default_password = mysql_config.get("password", "")
            default_db_name = mysql_config.get("database", "ecommerce")
            
            c1, c2 = st.columns(2)
            host = c1.text_input("Host", value=default_host)
            port = c2.text_input("Port", value=default_port)
            user = c1.text_input("User", value=default_user)
            pwd = c2.text_input("Password", value=default_password, type="password")
            db_name = st.text_input("DB Name", value=default_db_name)
            
            # MySQL连接测试按钮
            if st.button("🔧 测试MySQL连接", use_container_width=True):
                if host and port and user and db_name:
                    with st.spinner("正在测试MySQL连接..."):
                        try:
                            # 导入测试函数
                            import sys
                            sys.path.append('.')
                            from test_mysql_connection import test_mysql_connection
                            
                            result = test_mysql_connection(host, int(port), user, pwd, db_name)
                            
                            if result["success"]:
                                st.success("✅ MySQL连接测试成功！")
                                details = result["details"]
                                st.info(f"MySQL版本: {details.get('mysql_version', 'N/A')}")
                                st.info(f"数据库: {details.get('current_database', 'N/A')}")
                                st.info(f"表数量: {details.get('table_count', 0)}")
                                if details.get('tables'):
                                    st.info(f"表列表: {', '.join(details['tables'][:5])}{'...' if len(details['tables']) > 5 else ''}")
                            else:
                                st.error(f"❌ MySQL连接失败: {result['message']}")
                                if result.get('details', {}).get('suggestions'):
                                    st.warning("💡 建议解决方案:")
                                    for suggestion in result['details']['suggestions']:
                                        st.write(f"• {suggestion}")
                                        
                        except ImportError:
                            st.error("❌ 缺少依赖库，请安装: pip install pymysql sqlalchemy")
                        except Exception as e:
                            st.error(f"❌ 测试过程中出错: {str(e)}")
                else:
                    st.warning("⚠️ 请填写完整的MySQL连接信息")
            
            if host and db_name:
                uri = f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db_name}"
                final_uris = [uri]; db_path_val = uri 

    with st.expander("📚 知识与策略", expanded=False):
        # 根据当前数据库类型自动适配知识库配置
        current_db_type = st.session_state.config.get("db_type", "SQLite")
        
        if current_db_type == "SQLite":
            sqlite_config = st.session_state.config.get("sqlite_config", {})
            default_schema_path = sqlite_config.get("schema_path", "data/schema_northwind.json")
            st.info("💡 当前使用SQLite数据库，建议使用JSON格式的Schema文件")
            help_text = "SQLite: 推荐使用JSON格式的Schema文件（如data/schema_northwind.json）"
        else:
            mysql_config = st.session_state.config.get("mysql_config", {})
            default_schema_path = mysql_config.get("schema_path", "")
            st.info("💡 当前使用MySQL数据库，可以留空让系统自动从数据库获取Schema")
            help_text = "MySQL: 可留空自动获取Schema，或指定自定义Schema文件"
        
        # 使用key参数确保在数据库类型切换时重新渲染
        kb_input = st.text_area(
            "知识库路径", 
            value=default_schema_path,
            help=help_text,
            key=f"kb_input_{current_db_type}"  # 关键：使用数据库类型作为key的一部分
        )
        
        uploaded_files = st.file_uploader("上传文件", accept_multiple_files=True)
        log_path = st.text_input("日志路径", st.session_state.config.get("log_file", "data/agent.log"))
        max_retries = st.slider("最大重试", 1, 10, st.session_state.config.get("max_retries", 3))
        max_candidates = st.slider("可能性探索 (条)", 1, 5, st.session_state.config.get("max_candidates", 3))
        
        # 新增：空结果处理配置
        st.markdown("**空结果处理策略**")
        allow_empty_results = st.checkbox(
            "允许SQL查询结果为空", 
            value=st.session_state.config.get("allow_empty_results", True),
            help="如果禁用，当查询结果为空时将根据重试机制自动重试"
        )

    if st.button("💾 保存配置", type="primary", use_container_width=True):
        saved_paths = []
        if uploaded_files:
            os.makedirs("data/uploads", exist_ok=True)
            for uf in uploaded_files:
                path = f"data/uploads/{uf.name}"
                with open(path, "wb") as f: f.write(uf.getbuffer())
                saved_paths.append(path)
        kb_paths = list(set([p.strip() for p in kb_input.split('\n') if p.strip()] + saved_paths))
        
        # 更新基础配置
        st.session_state.config.update({
            "api_base": api_url, "api_key": api_key, "model_name": model,
            "db_type": db_type, "db_path": db_path_val, "db_uris": final_uris,
            "schema_path": "\n".join(kb_paths), "kb_paths_list": kb_paths,
            "model_path": rag_path, "log_file": log_path,
            "max_retries": max_retries, "max_candidates": max_candidates,
            "allow_empty_results": allow_empty_results,
            "enable_ai_recommendations": enable_ai_recommendations,
            "recommendation_use_separate_api": use_separate_api,
            "recommendation_api_base": rec_api_url,
            "recommendation_api_key": rec_api_key,
            "recommendation_model_name": rec_model
        })
        
        # 分别保存SQLite和MySQL的配置
        if db_type == "SQLite":
            st.session_state.config["sqlite_config"] = {
                "db_path": db_path_val,
                "schema_path": "\n".join(kb_paths)
            }
        else:  # MySQL
            st.session_state.config["mysql_config"] = {
                "host": host,
                "port": port,
                "user": user,
                "password": pwd,
                "database": db_name,
                "schema_path": "\n".join(kb_paths)
            }
        
        save_config(st.session_state.config)
        st.success("✅ 配置已保存！数据库配置已分别保存，切换数据库类型时会自动恢复对应配置。")
        st.cache_resource.clear(); st.rerun()

    st.markdown("---")
    st.markdown("### 💬 会话管理")
    ids = list(st.session_state.history.keys())[::-1]
    titles = [st.session_state.history[i]["title"] for i in ids]
    try: curr_idx = ids.index(st.session_state.current_session_id)
    except ValueError: curr_idx = 0
    sel = st.selectbox("历史记录", titles, index=curr_idx, key="history_selector")
    if sel:
        tid = ids[titles.index(sel)]
        if tid != st.session_state.current_session_id:
            st.session_state.current_session_id = tid
            st.rerun()
    
    # 会话操作按钮
    c1, c2 = st.columns(2)
    if c1.button("➕ 新建", use_container_width=True):
        sid, hist = create_new_session(st.session_state.history)
        st.session_state.history = hist
        st.session_state.current_session_id = sid
        st.rerun()
    if c2.button("🗑️ 删除", type="secondary", use_container_width=True):
        hist = delete_session(st.session_state.history, st.session_state.current_session_id)
        st.session_state.history = hist
        if not st.session_state.history:
            sid, hist = create_new_session(st.session_state.history)
            st.session_state.history = hist
            st.session_state.current_session_id = sid
        else: st.session_state.current_session_id = list(st.session_state.history.keys())[0]
        st.rerun()
    
    # 分享和导出功能
    st.markdown("#### 📤 分享与导出")
    current_session = st.session_state.history.get(st.session_state.current_session_id, {})
    has_messages = len(current_session.get("messages", [])) > 0
    
    if has_messages:
        # PDF报告导出
        if st.button("📄 导出PDF报告", use_container_width=True):
            with st.spinner("正在生成PDF报告..."):
                pdf_path = export_manager.export_session_to_pdf(
                    current_session, 
                    current_session.get("title", "分析报告")
                )
                if pdf_path:
                    st.success("PDF报告生成成功！")
                    # 提供下载链接
                    with open(pdf_path, "rb") as pdf_file:
                        st.download_button(
                            label="⬇️ 下载PDF报告",
                            data=pdf_file.read(),
                            file_name=os.path.basename(pdf_path),
                            mime="application/pdf",
                            use_container_width=True
                        )
                else:
                    st.error("PDF生成失败，请安装reportlab库")
        
        # DOCX报告导出
        if st.button("📝 导出Word报告", use_container_width=True):
            with st.spinner("正在生成Word报告..."):
                docx_path = export_manager.export_session_to_docx(
                    current_session, 
                    current_session.get("title", "分析报告")
                )
                if docx_path:
                    st.success("Word报告生成成功！")
                    # 提供下载链接
                    with open(docx_path, "rb") as docx_file:
                        st.download_button(
                            label="⬇️ 下载Word报告",
                            data=docx_file.read(),
                            file_name=os.path.basename(docx_path),
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True
                        )
                else:
                    st.error("Word生成失败，请安装python-docx库")
        
        # 创建分享链接
        if st.button("🔗 创建分享链接", use_container_width=True):
            share_id = export_manager.create_shareable_session(
                current_session, 
                st.session_state.current_session_id
            )
            if share_id:
                share_url = f"分享ID: {share_id}"
                st.success("分享链接创建成功！")
                st.code(share_url, language="text")
                st.info("💡 其他用户可以使用此分享ID查看您的分析结果")
            else:
                st.error("分享链接创建失败")
    else:
        st.info("💡 开始对话后可使用分享和导出功能")

# --- 动态监控刷新函数 ---
def update_monitor():
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    total_lat = st.session_state.last_total_latency
    rag_lat = st.session_state.last_rag_latency
    lat_color = "#28a745" if total_lat < 1000 else "#ffc107" if total_lat < 3000 else "#dc3545"
    
    # 收集并保存性能指标
    current_metrics = performance_monitor.collect_current_metrics(rag_lat, total_lat)
    if current_metrics:
        performance_monitor.save_metrics(current_metrics)
    
    # 检测异常
    anomalies = performance_monitor.detect_anomalies(current_metrics)
    suggestions = performance_monitor.get_optimization_suggestions(current_metrics, anomalies)
    
    # 获取性能摘要
    summary = performance_monitor.get_performance_summary()
    
    # 构建监控面板内容 - 只显示基本性能指标
    summary_content = ""
    if summary:
        avg_cpu = summary.get('avg_cpu', 0)
        total_queries = summary.get('total_queries', 0)
        summary_content = f"📈 **1小时摘要**: 平均CPU: {avg_cpu}% | 查询数: {total_queries}"
    
    # 使用Streamlit原生组件显示基本性能指标
    with monitor_placeholder.container():
        st.markdown("**📊 实时性能监控**")
        
        # 性能指标
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CPU 占用", f"{cpu}%")
            st.metric("OpenVINO", f"{rag_lat:.1f} ms")
        with col2:
            st.metric("内存占用", f"{mem}%")
            st.metric("端到端延迟", f"{total_lat:.0f} ms")
        
        # 只显示摘要信息，不显示警告和建议
        if summary_content:
            st.caption(summary_content)

update_monitor()

# --- 推荐引擎客户端创建函数 ---
@st.cache_resource
def get_recommendation_client(cfg):
    """获取推荐引擎的LLM客户端"""
    if not cfg.get("enable_ai_recommendations", True):
        return None, None, "AI推荐已禁用"
    
    # 检查是否使用独立的推荐API配置
    use_separate_api = cfg.get("recommendation_use_separate_api", False)
    
    if use_separate_api:
        # 使用独立的推荐API配置
        api_key = cfg.get("recommendation_api_key", "")
        api_base = cfg.get("recommendation_api_base", "https://api.deepseek.com")
        model_name = cfg.get("recommendation_model_name", "deepseek-reasoner")
        
        if not api_key:
            return None, None, "推荐API Key未配置"
    else:
        # 使用SQL生成的API配置
        api_key = cfg.get("api_key", "")
        api_base = cfg.get("api_base", "https://api.deepseek.com")
        model_name = cfg.get("model_name", "deepseek-reasoner")
        
        if not api_key:
            return None, None, "API Key未配置"
    
    try:
        from openai import OpenAI
        import httpx
        
        # 处理URL格式
        clean_url = api_base.rstrip('/')
        if not clean_url.endswith('/v1'):
            clean_url += "/v1"
        
        client = create_openai_client_safe(api_key, clean_url, 60.0)
        
        return client, model_name, None
        
    except Exception as e:
        return None, None, f"推荐客户端创建失败: {str(e)}"

# --- 懒加载 Agent ---
@st.cache_resource
def get_agent(cfg):
    if not cfg["api_key"]: return None, "请配置 API Key"
    try:
        rag = IntelRAG(model_path=cfg.get("model_path"), db_uris=cfg.get("db_uris", []), kb_paths=cfg.get("kb_paths_list", []))
        
        agent = Text2SQLAgent(
            api_key=cfg["api_key"], base_url=cfg["api_base"], model_name=cfg["model_name"], 
            db_uris=cfg.get("db_uris", []), rag_engine=rag, 
            max_retries=cfg.get("max_retries", 3), max_candidates=cfg.get("max_candidates", 1),
            log_file=cfg.get("log_file", "data/agent.log"),
            config=cfg  # 🧠 传递完整配置给Prompt模板系统
        )
        
        return agent, None
    except Exception as e: return None, str(e)

# --- 页面主逻辑 ---
current_data = st.session_state.history[st.session_state.current_session_id]
messages = current_data["messages"]

# 处理按钮输入
prompt_input = None
if st.session_state.prompt_trigger:
    prompt_input = st.session_state.prompt_trigger
    st.session_state.prompt_trigger = None
elif user_input := st.chat_input("输入业务问题 (支持中英文)..."):
    prompt_input = user_input

# --- 欢迎页 ---
if len(messages) == 0:
    # 主标题区域 - 整体上移并美化
    st.markdown("""
    <div style="text-align: center; margin-top: -10px; margin-bottom: 25px;">
        <h1 style="color: #0068B5; margin: 0; font-weight: 600; font-size: 2.8rem; letter-spacing: -0.5px;">Intel® DeepInsight</h1>
        <p style="font-size: 1.1em; color: #666; margin-top: 12px; line-height: 1.6;">
            基于 OpenVINO™ 的本地化智能零售决策系统<br>
            <span style="font-size: 0.85em; color: #888; font-weight: 500;">全本地运行 · 隐私安全 · 极速推理</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 显示上下文记忆状态 - 使用最新的状态
    if CONTEXT_MEMORY_AVAILABLE:
        current_memory_enabled = st.session_state.get('context_memory_enabled', True)
        if current_memory_enabled:
            st.markdown("""
            <div class="context-status">
                🧠 <strong>上下文记忆已启用</strong> - AI将记住对话历史，提供更智能的回复
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="context-status context-disabled">
                💭 <strong>上下文记忆已禁用</strong> - AI将不会记住对话历史
            </div>
            """, unsafe_allow_html=True)
    
    # 智能推荐问题标题 - 优化样式
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h4 style="color: #333; font-weight: 600; margin-bottom: 15px; display: flex; align-items: center;">
            <span style="margin-right: 8px;">💡</span>智能推荐问题：
        </h4>
    </div>
    """, unsafe_allow_html=True)
    
    # 使用固定的示例问题
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        if st.button("🏆 2016年销售额最高的5个城市是哪里？", use_container_width=True):
            st.session_state.prompt_trigger = "2016年销售额最高5个的城市是哪里"
            st.rerun()
    with c2:
        if st.button("📊 家具类产品的平均利润率是多少？", use_container_width=True):
            st.session_state.prompt_trigger = "家具类产品的平均利润率是多少"
            st.rerun()
    with c3:
        if st.button("📈 库存积压最严重的TOP5产品是？", use_container_width=True):
            st.session_state.prompt_trigger = "库存积压最严重的TOP5产品是？"
            st.rerun()
    with c4:
        if st.button("💻 告诉我，哪几个产品决定了我们的生死？", use_container_width=True):
            st.session_state.prompt_trigger = "告诉我，哪几个产品决定了我们的生死？"
            st.rerun()

# --- 历史消息渲染 (🔥 统一渲染逻辑：确保所有标题常驻) ---
for msg_index, msg in enumerate(messages):
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"]=="user" else "🤖"):
        # 1. 思考过程 (如有)
        if "thought" in msg and msg["thought"]:
            # 获取配置的模型名称用于显示
            model_display_name = st.session_state.config.get("model_name", "AI模型")
            with st.expander(f"🤔 思考过程 ({model_display_name})", expanded=False):
                st.markdown(f"<div class='thought-persist'>{msg['thought']}</div>", unsafe_allow_html=True)
        
        # 判断消息类型
        is_sql_result = "data" in msg and msg["data"] is not None
        
        if is_sql_result:
            # === 类型 A: 数据查询结果 (保持标题顺序) ===
            
            # 0. 表选择过程信息持久化显示 (历史消息)
            if "table_selection_info" in msg and msg["table_selection_info"]:
                table_info = msg["table_selection_info"]
                if any(table_info.values()):
                    with st.expander("🗄️ 智能表选择过程", expanded=False):
                        st.markdown("**📋 表选择详细过程**")
                        
                        # 显示初步筛选结果
                        if table_info.get("initial_analysis"):
                            st.markdown("**第1步：语义相似度初步筛选**")
                            st.info(table_info["initial_analysis"])
                        
                        # 显示Agent推理过程
                        if table_info.get("agent_reasoning"):
                            st.markdown("**第2步：Agent智能筛选推理**")
                            st.success(f"🧠 推理过程: {table_info['agent_reasoning']}")
                        
                        # 显示关联分析
                        if table_info.get("join_analysis"):
                            st.markdown("**第3步：表关联关系分析**")
                            st.info(table_info["join_analysis"])
                        
                        # 显示最终选择结果
                        if table_info.get("final_selection"):
                            final_selection = table_info["final_selection"]
                            selected_tables = final_selection.get("selected_tables", [])
                            analysis = final_selection.get("analysis", {})
                            
                            st.markdown("**🎯 最终选择结果**")
                            
                            if selected_tables:
                                # 显示选择推理
                                selection_reasoning = analysis.get("selection_reasoning", "")
                                if selection_reasoning:
                                    st.info(f"🧠 选择推理: {selection_reasoning}")
                                
                                # 显示是否使用了语义匹配
                                if analysis.get("use_semantic_matching"):
                                    st.success("🚀 使用OpenVINO语义匹配算法")
                                else:
                                    st.warning("⚠️ 使用传统关键词匹配")
                                
                                # 显示处理时间
                                processing_time = analysis.get("processing_time_ms", 0)
                                if processing_time > 0:
                                    st.caption(f"⏱️ 处理时间: {processing_time:.1f}ms")
                                
                                # 显示选中的表（简化版）
                                st.markdown("**📊 相关数据表**:")
                                for i, table_dict in enumerate(selected_tables[:3], 1):
                                    score_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                                    table_name = table_dict.get("table_name", "未知表")
                                    relevance_score = table_dict.get("relevance_score", 0.0)
                                    reasoning = table_dict.get("reasoning", "无推理信息")
                                    st.caption(f"{score_emoji} **{table_name}** (相关性: {relevance_score:.1f}) - {reasoning}")
            
            # 2.1 标题：查询结果
            st.markdown("##### 🔎 查询结果")
            df_hist = pd.DataFrame(msg["data"])
            if not df_hist.empty:
                st.write(f"共查询到 {len(df_hist)} 条数据：")
                
                # 添加数据筛选功能
                if len(df_hist) > 10:  # 数据量较大时提供筛选
                    with st.expander("🔍 数据筛选与排序", expanded=False):
                        # 快速筛选按钮
                        quick_filter = data_filter.create_quick_filter_buttons(df_hist, f"hist_quick_{msg_index}")
                        if quick_filter:
                            df_hist = data_filter.apply_quick_filter(df_hist, quick_filter)
                            st.success(f"已应用筛选: {quick_filter['name']}")
                        
                        # 详细筛选界面
                        filtered_df, filter_config = data_filter.create_filter_interface(df_hist, f"hist_filter_{msg_index}")
                        if filter_config:
                            df_hist = filtered_df
                            
                            # 保存筛选配置选项
                            col1, col2 = st.columns(2)
                            with col1:
                                filter_name = st.text_input("筛选配置名称", placeholder="输入名称保存筛选配置", key=f"filter_name_hist_{msg_index}")
                            with col2:
                                if st.button("💾 保存筛选", key=f"save_filter_hist_{msg_index}") and filter_name:
                                    if data_filter.save_filter_config(filter_config, filter_name):
                                        st.success(f"筛选配置 '{filter_name}' 已保存")
                
                st.dataframe(df_hist, hide_index=True)
                
                # 2.2 标题：可视化 (如果符合条件)
                numeric_cols = df_hist.select_dtypes(include='number').columns
                if len(df_hist) > 1 and len(numeric_cols) > 0:
                    st.markdown("##### 📊 可视化")
                    
                    # 使用新的可视化引擎
                    chart_options = viz_engine.get_chart_options(df_hist, msg.get('content', ''))
                    
                    # 如果有多个图表选项，让用户选择
                    if len(chart_options) > 2:
                        col1, col2 = st.columns([3, 1])
                        with col2:
                            selected_chart = st.selectbox(
                                "图表类型", 
                                options=[opt["type"] for opt in chart_options],
                                format_func=lambda x: next(opt["icon"] + " " + opt["name"] for opt in chart_options if opt["type"] == x),
                                key=f"hist_chart_select_{msg_index}"
                            )
                        with col1:
                            if selected_chart == "table":
                                # 显示数据表格
                                st.dataframe(df_hist, use_container_width=True)
                            else:
                                fig = viz_engine.create_interactive_chart(df_hist, selected_chart, msg.get('content', ''))
                                # 生成历史消息图表的唯一key
                                chart_key = generate_history_chart_key(msg_index, selected_chart, df_hist)
                                st.plotly_chart(fig, use_container_width=True, key=chart_key)
                    else:
                        # 自动选择最佳图表类型
                        auto_chart_type = viz_engine.detect_chart_type(df_hist, msg.get('content', ''))
                        if auto_chart_type == "table":
                            # 显示数据表格
                            st.dataframe(df_hist, use_container_width=True)
                        else:
                            fig = viz_engine.create_interactive_chart(df_hist, query_context=msg.get('content', ''))
                            # 生成历史消息图表的唯一key（自动类型）
                            chart_key = generate_history_chart_key(msg_index, "auto", df_hist)
                            st.plotly_chart(fig, use_container_width=True, key=chart_key)
            
            # 2.3 标题：商业洞察 (这里显式重新渲染标题，确保不消失！)
            st.markdown("##### 💡 商业洞察")
            if msg.get("content"):
                st.markdown(msg["content"])
            
            # 2.3.5 异常检测分析
            if "data" in msg and msg["data"]:
                df_for_anomaly = pd.DataFrame(msg["data"])
                if not df_for_anomaly.empty and len(df_for_anomaly) > 2:
                    # 获取原始查询
                    user_query = ""
                    msg_index = messages.index(msg)
                    if msg_index > 0:
                        user_query = messages[msg_index - 1].get("content", "")
                    
                    anomaly_analysis = anomaly_detector.analyze_anomalies(df_for_anomaly, user_query)
                    
                    if anomaly_analysis["total_anomalies"] > 0:
                        st.markdown("##### ⚠️ 异常检测")
                        
                        # 异常摘要
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("总异常数", anomaly_analysis["total_anomalies"])
                        with col2:
                            st.metric("高风险", anomaly_analysis["high_severity"], delta=None if anomaly_analysis["high_severity"] == 0 else "需关注")
                        with col3:
                            st.metric("中风险", anomaly_analysis["medium_severity"])
                        
                        # 主要异常预览 - 新增功能
                        if "primary_anomaly" in anomaly_analysis and anomaly_analysis["primary_anomaly"]:
                            primary = anomaly_analysis["primary_anomaly"]
                            
                            # 风险等级颜色映射
                            risk_colors = {
                                "high": "🔴",
                                "medium": "🟡", 
                                "low": "🟢"
                            }
                            risk_color = risk_colors.get(primary.impact_level, "🔵")
                            
                            # 显示主要异常预览卡片
                            with st.container():
                                st.markdown("**📋 主要异常预览**")
                                
                                # 异常标题行
                                col_icon, col_desc = st.columns([1, 5])
                                with col_icon:
                                    st.markdown(f"### {primary.icon}")
                                with col_desc:
                                    st.markdown(f"**{risk_color} {primary.type_name}** ({primary.impact_level}风险)")
                                    st.write(primary.short_description)
                                
                                # 异常详情行
                                col_reason, col_sample = st.columns(2)
                                with col_reason:
                                    st.write(f"**原因**: {primary.quick_reason}")
                                    if primary.quick_action:
                                        st.write(f"**建议**: {primary.quick_action}")
                                
                                with col_sample:
                                    if primary.sample_data:
                                        st.write("**异常样本**:")
                                        for sample in primary.sample_data[:2]:
                                            st.write(f"• {sample}")
                                
                                # 置信度显示
                                confidence_pct = int(primary.confidence * 100)
                                confidence_label = "高" if primary.confidence > 0.8 else "中" if primary.confidence > 0.6 else "低"
                                st.caption(f"🎯 检测置信度: {confidence_pct}% ({confidence_label})")
                                
                                # 如果有多个异常，显示其他异常提示
                                if anomaly_analysis["total_anomalies"] > 1:
                                    other_count = anomaly_analysis["total_anomalies"] - 1
                                    st.info(f"💡 还有 {other_count} 个其他异常，点击下方查看详情")
                        
                        # 显示前3个最重要的异常（保持原有的详细展示）
                        # 在历史消息部分和新消息生成部分都修改这个循环：
                        for i, anomaly in enumerate(anomaly_analysis["anomalies"][:3]):
                            severity_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(anomaly["severity"], "🔵")
                            
                            # 构建完整的异常信息
                            message = f"{severity_color} **{anomaly.get('description', '异常')}**\n\n"
                            
                            # 添加统计依据
                            if 'statistical_basis' in anomaly:
                                message += f"📊 **检测依据**: {anomaly['statistical_basis']}\n\n"
                            
                            # 添加具体证据
                            if 'evidence_details' in anomaly:
                                message += f"🔍 **具体证据**:\n{anomaly['evidence_details']}\n\n"
                            elif 'details' in anomaly:
                                message += f"📝 **详细情况**: {anomaly['details']}\n\n"
                            
                            # 添加建议
                            if 'suggestion' in anomaly:
                                message += f"💡 **处理建议**: {anomaly['suggestion']}"
                            
                            st.warning(message)
                            with st.expander(f"{severity_color} **{anomaly['description']}**", expanded=True):
                                # 基本信息
                                col_info1, col_info2 = st.columns(2)
                                with col_info1:
                                    st.write(f"**异常类型**: {anomaly.get('type', 'unknown')}")
                                    st.write(f"**影响字段**: {anomaly.get('column', 'N/A')}")
                                    st.write(f"**异常数量**: {anomaly.get('count', 0)}")
                                with col_info2:
                                    st.write(f"**风险等级**: {anomaly.get('severity', 'unknown')}")
                                    if 'ratio' in anomaly:
                                        st.write(f"**异常比例**: {anomaly['ratio']:.1%}")
                                    if 'total_loss' in anomaly:
                                        st.write(f"**财务影响**: {anomaly['total_loss']:,.2f}")
                                
                                # 检测标准和依据
                                if 'criteria' in anomaly:
                                    st.markdown("**🔍 检测标准**")
                                    criteria = anomaly['criteria']
                                    st.write(f"• **方法**: {criteria.get('method', 'N/A')}")
                                    st.write(f"• **阈值**: {criteria.get('threshold', 'N/A')}")
                                    if 'calculation' in criteria:
                                        st.write(f"• **计算公式**: {criteria['calculation']}")
                                    
                                    # 显示具体的数值标准
                                    if 'lower_bound' in criteria and 'upper_bound' in criteria:
                                        st.write(f"• **正常范围**: {criteria['lower_bound']:.2f} - {criteria['upper_bound']:.2f}")
                                    if 'z_threshold' in criteria:
                                        st.write(f"• **Z-Score阈值**: {criteria['z_threshold']}")
                                
                                # 异常证据和具体数据
                                if 'evidence' in anomaly:
                                    st.markdown("**📊 异常证据**")
                                    evidence = anomaly['evidence']
                                    
                                    # 显示异常记录样本
                                    if 'outlier_records' in evidence and evidence['outlier_records']:
                                        st.write("**异常数据样本**:")
                                        for j, record in enumerate(evidence['outlier_records'][:2]):
                                            st.write(f"  {j+1}. 行{record['row_index']}: 异常值 = {record['anomaly_value']:.2f}")
                                            if len(record['full_record']) <= 5:
                                                st.json(record['full_record'])
                                    
                                    elif 'negative_records' in evidence and evidence['negative_records']:
                                        st.write("**负利润记录样本**:")
                                        for j, record in enumerate(evidence['negative_records'][:2]):
                                            st.write(f"  {j+1}. 行{record['row_index']}: 利润 = {record['profit_value']:.2f}")
                                    
                                    elif 'zero_records' in evidence and evidence['zero_records']:
                                        st.write("**零值记录样本**:")
                                        for j, record in enumerate(evidence['zero_records'][:2]):
                                            st.write(f"  {j+1}. 行{record['row_index']}: 存在零值")
                                    
                                    elif 'high_margin_records' in evidence and evidence['high_margin_records']:
                                        st.write("**高利润率记录样本**:")
                                        for j, record in enumerate(evidence['high_margin_records'][:2]):
                                            st.write(f"  {j+1}. 行{record['row_index']}: 利润率 = {record['profit_margin']:.1%}")
                                    
                                    elif 'price_anomaly_records' in evidence and evidence['price_anomaly_records']:
                                        st.write("**异常单价记录样本**:")
                                        for j, record in enumerate(evidence['price_anomaly_records'][:2]):
                                            st.write(f"  {j+1}. 行{record['row_index']}: 单价 = {record['unit_price']:.2f}")
                                    
                                    elif 'trend_break_points' in evidence and evidence['trend_break_points']:
                                        st.write("**趋势突变点**:")
                                        for j, point in enumerate(evidence['trend_break_points'][:2]):
                                            st.write(f"  {j+1}. 变化: {point['previous_value']:.2f} → {point['current_value']:.2f} ({point['change_percentage']:.1%})")
                                    
                                    elif 'decline_sequence' in evidence and evidence['decline_sequence']:
                                        st.write("**下降趋势序列**:")
                                        for j, point in enumerate(evidence['decline_sequence'][-3:]):  # 显示最后3个点
                                            st.write(f"  期{point['period']}: {point['value']:.2f} (累计下降: {point['cumulative_decline']:.1%})")
                                
                                # 建议
                                st.markdown("**💡 处理建议**")
                                st.write(anomaly.get('suggestion', '建议进一步分析此异常'))
                        
                        # 更多异常详情
                        if len(anomaly_analysis["anomalies"]) > 3:
                            with st.expander(f"查看更多异常 ({len(anomaly_analysis['anomalies']) - 3} 个)", expanded=False):
                                for anomaly in anomaly_analysis["anomalies"][3:]:
                                    severity_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(anomaly["severity"], "🔵")
                                    
                                    st.markdown(f"**{severity_color} {anomaly['description']}**")
                                    
                                    # 简化显示检测标准
                                    if 'criteria' in anomaly:
                                        criteria = anomaly['criteria']
                                        st.write(f"• 检测方法: {criteria.get('method', 'N/A')}")
                                        st.write(f"• 阈值标准: {criteria.get('threshold', 'N/A')}")
                                    
                                    # 简化显示异常数据
                                    if 'evidence' in anomaly:
                                        evidence = anomaly['evidence']
                                        if 'statistical_summary' in evidence:
                                            summary = evidence['statistical_summary']
                                            if 'extreme_range' in summary:
                                                st.write(f"• 异常值范围: {summary['extreme_range']}")
                                            elif 'affected_percentage' in summary:
                                                st.write(f"• 影响比例: {summary['affected_percentage']:.1f}%")
                                    
                                    st.write(f"• 建议: {anomaly.get('suggestion', '需要进一步分析')}")
                                    st.markdown("---")
            
            # 2.3.7 其他可能的理解方式 (历史消息)
            if "alternatives" in msg and msg["alternatives"]:
                st.markdown("##### 🤔 其他可能的理解方式")
                
                with st.expander(f"查看其他 {len(msg['alternatives'])} 种理解方式", expanded=False):
                    st.markdown("*点击下方按钮可以按照不同的理解方式重新执行查询*")
                    
                    for i, alt in enumerate(msg["alternatives"]):
                        with st.container():
                            col1, col2 = st.columns([4, 1])
                            
                            with col1:
                                # 处理可能是字典或对象的情况
                                if isinstance(alt, dict):
                                    rank = alt.get("rank", i + 1)
                                    natural_desc = alt.get("natural_description", alt.get("description", "未知理解方式"))
                                    confidence = alt.get("confidence", 0.0)
                                    key_interpretations = alt.get("key_interpretations", {})
                                else:
                                    rank = getattr(alt, "rank", i + 1)
                                    natural_desc = getattr(alt, "natural_description", getattr(alt, "description", "未知理解方式"))
                                    confidence = getattr(alt, "confidence", 0.0)
                                    key_interpretations = getattr(alt, "key_interpretations", {})
                                
                                st.write(f"**理解方式 {rank}**:")
                                st.write(f"📝 {natural_desc}")
                                st.write(f"🎯 置信度: {confidence:.1%}")
                                
                                if key_interpretations:
                                    with st.expander("查看技术细节", expanded=False):
                                        for term, interp in key_interpretations.items():
                                            interp_desc = interp.get('desc', '') if isinstance(interp, dict) else str(interp)
                                            st.caption(f"• {term}: {interp_desc}")
                            
                            with col2:
                                if st.button(f"🔄 选择此理解", key=f"select_alt_hist_{msg_index}_{i}"):
                                    # 重新执行这种理解方式
                                    st.session_state.prompt_trigger = natural_desc
                                    st.rerun()
                            
                            st.divider()
            
            # 2.4 推荐相关问题
            if "data" in msg and msg["data"]:
                st.markdown("##### 🤔 您可能还想了解")
                
                # 优先使用保存的推荐，如果没有则重新生成（兼容旧消息）
                if "recommendations" in msg and msg["recommendations"]:
                    recommendations = msg["recommendations"]
                else:
                    # 兼容旧消息：重新生成推荐
                    df_for_rec = pd.DataFrame(msg["data"])
                    # 获取原始查询（从历史消息中找到对应的用户问题）
                    user_query = ""
                    msg_index = messages.index(msg)
                    if msg_index > 0:
                        user_query = messages[msg_index - 1].get("content", "")
                    
                    recommendations = recommendation_engine.generate_recommendations(
                        current_query=user_query,
                        result_df=df_for_rec,
                        num_recommendations=3,
                        llm_client=None,  # 历史消息使用备用推荐
                        model_name=None
                    )
                
                if recommendations:
                    rec_cols = st.columns(len(recommendations))
                    for i, rec in enumerate(recommendations):
                        with rec_cols[i]:
                            if st.button(f"💭 {rec}", use_container_width=True, key=f"rec_hist_{msg_index}_{i}"):
                                recommendation_engine.record_question_click(rec)
                                st.session_state.prompt_trigger = rec
                                st.rerun()
            
            # 2.5 标题：数据详情 & SQL
            with st.expander("📝 原始 SQL 与数据导出", expanded=False):
                if not msg["data"]: 
                    st.warning("结果为空")
                else:
                    # 数据导出功能
                    df_export = pd.DataFrame(msg["data"])
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv_data = export_manager.export_data_to_csv(df_export, "query_result")
                        if csv_data and os.path.exists(csv_data):
                            with open(csv_data, "rb") as csv_file:
                                st.download_button(
                                    label="📊 下载CSV",
                                    data=csv_file.read(),
                                    file_name=os.path.basename(csv_data),
                                    mime="text/csv",
                                    key=f"csv_download_hist_{msg_index}"
                                )
                    
                    with col2:
                        excel_data = export_manager.export_data_to_excel(df_export, "query_result")
                        if excel_data and os.path.exists(excel_data):
                            with open(excel_data, "rb") as excel_file:
                                st.download_button(
                                    label="📈 下载Excel",
                                    data=excel_file.read(),
                                    file_name=os.path.basename(excel_data),
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key=f"excel_download_hist_{msg_index}"
                                )
                
                if "sql" in msg: 
                    st.code(msg["sql"], language="sql")
                
        else:
            # === 类型 B: 普通对话 ===
            if msg.get("content"):
                st.markdown(msg["content"])


# --- 新的推理与生成逻辑 ---
if prompt_input:
    # 懒加载
    agent = None
    if not st.session_state.agent_loaded:
        with st.status("🚀 首次运行，正在加载 OpenVINO 加速引擎...", expanded=True) as status:
            agent, err = get_agent(st.session_state.config)
            if err:
                status.update(label="❌ 初始化失败", state="error")
                st.error(err); st.stop()
            st.session_state.agent_loaded = True
            status.update(label="✅ 引擎加载完毕", state="complete", expanded=False)
    else:
        agent, err = get_agent(st.session_state.config)
        if err: st.error(err); st.stop()
    
    # 确保agent已正确加载
    if agent is None:
        st.error("Agent 加载失败，请检查配置")
        st.stop()
    
    # 渲染用户提问
    st.chat_message("user", avatar="🧑‍💻").markdown(prompt_input)
    messages.append({"role": "user", "content": prompt_input})
    
    # 硬件优化预处理
    hardware_optimization_result = None
    if HARDWARE_OPTIMIZATION_AVAILABLE:
        try:
            # 估算查询结果大小（基于查询复杂度）
            estimated_result_size = 100
            if any(keyword in prompt_input.lower() for keyword in ['join', 'group by', 'sum', 'count']):
                estimated_result_size = 500
            if any(keyword in prompt_input.lower() for keyword in ['union', 'subquery', 'window']):
                estimated_result_size = 1000
            
            # 执行硬件优化
            hardware_optimization_result = optimize_query_performance(prompt_input, estimated_result_size)
            
            if hardware_optimization_result:
                vendor = hardware_optimization_result.vendor.value
                st.info(f"🚀 {vendor}优化已启用 - 预期加速比: {hardware_optimization_result.overall_speedup:.2f}x")
        except Exception as e:
            st.warning(f"硬件优化预处理失败: {e}")
    
    # 技术卓越性优化预处理 - 后端功能启用
    if TECHNICAL_EXCELLENCE_AVAILABLE:
        try:
            # 记录查询开始，用于性能监控
            query_start_time = time.perf_counter()
            
            # 估算输入大小
            input_size = len(prompt_input.encode('utf-8'))
            
            # 预测性能（如果有历史数据）
            tech_status = tech_manager.get_technical_status()
            if tech_status.overall_score >= 70:
                # 后端优化处理，不显示UI信息
                pass
            
        except Exception as e:
            logger.warning(f"技术卓越性预处理失败: {e}")
    
    # AI 回答容器
    with st.chat_message("assistant", avatar="🤖"):
        # 🧠 集成上下文记忆系统
        if CONTEXT_MEMORY_AVAILABLE and st.session_state.get('context_memory_enabled', True):
            try:
                # 使用上下文记忆系统处理输入
                contextual_prompt = integrate_with_messages(
                    messages[:-1],  # 不包括刚添加的用户消息
                    prompt_input,
                    system_instruction="你是一个专业的数据分析助手，专门帮助用户分析零售业务数据。"
                )
                
                # 显示上下文状态
                st.caption("🧠 已加载对话上下文")
                
                # 使用上下文感知的提示进行处理
                final_prompt = contextual_prompt
            except Exception as e:
                st.warning(f"⚠️ 上下文记忆系统遇到问题，使用基本模式: {e}")
                final_prompt = prompt_input
        else:
            # 传统方式处理
            final_prompt = prompt_input
        status_box = st.status("🚀 系统启动...", expanded=True)
        code_ph = None
        thought_ph = None
        curr_sql = ""
        curr_thought = ""
        
        start_time = time.perf_counter()
        
        try:
            # 使用上下文感知的提示或传统提示
            stream_gen = agent.generate_and_execute_stream(final_prompt, messages[:-1])
            final_resp, df_result, sql_code, mode = "", None, None, "CHAT"
            selected_possibility, alternatives = None, []
            step_count = 0 
            
            # 保存表选择过程信息，用于持久化显示
            table_selection_info = {
                "initial_analysis": "",
                "agent_reasoning": "",
                "join_analysis": "",
                "final_selection": None
            }

            for step in stream_gen:
                step_count += 1
                if step_count % 5 == 0: update_monitor()

                if step["type"] == "step":
                    status_box.write(f"{step['icon']} {step['msg']}")
                    status_box.update(state=step["status"])
                    if "rag_latency" in step:
                        st.session_state.last_rag_latency = step["rag_latency"]
                        update_monitor()
                
                elif step["type"] == "code_start":
                    status_box.markdown(f"**{step.get('label', 'Code')}**")
                    code_ph = status_box.empty()
                    curr_sql = ""
                
                elif step["type"] == "code_chunk":
                    curr_sql += step["content"]
                    code_ph.code(curr_sql, language="sql")
                
                elif step["type"] == "thought_start":
                    # 获取配置的模型名称用于显示
                    model_display_name = st.session_state.config.get("model_name", "AI模型")
                    status_box.markdown(f"**🤔 语义分析 ({model_display_name} Thinking)...**")
                    thought_ph = status_box.empty()
                    curr_thought = ""
                
                elif step["type"] == "thought_chunk":
                    curr_thought += step["content"]
                    thought_ph.markdown(f"<div class='thought-box'>{curr_thought}</div>", unsafe_allow_html=True)
                
                elif step["type"] == "error_log":
                    status_box.error(f"⚠️ {step['content']}")

                elif step["type"] == "table_analysis":
                    # 显示初步表筛选结果并保存
                    status_box.markdown("**🔍 初步表筛选结果**")
                    status_box.info(step["content"])
                    table_selection_info["initial_analysis"] = step["content"]
                
                elif step["type"] == "agent_reasoning":
                    # 显示Agent推理过程并保存
                    status_box.markdown("**🤖 Agent智能筛选推理**")
                    status_box.success(f"🧠 推理过程: {step['content']}")
                    table_selection_info["agent_reasoning"] = step["content"]
                
                elif step["type"] == "join_analysis":
                    # 显示表关联分析并保存
                    status_box.markdown("**🔗 表关联关系分析**")
                    status_box.info(step["content"])
                    table_selection_info["join_analysis"] = step["content"]

                elif step["type"] == "table_selection":
                    # 显示表选择结果并保存信息
                    status_box.markdown("**🗄️ 智能表选择结果**")
                    
                    selected_tables = step.get("selected_tables", [])
                    analysis = step.get("analysis", {})
                    
                    # 保存最终选择信息
                    table_selection_info["final_selection"] = {
                        "selected_tables": selected_tables,
                        "analysis": analysis
                    }
                    
                    if selected_tables:
                        # 显示选择推理
                        selection_reasoning = analysis.get("selection_reasoning", "")
                        if selection_reasoning:
                            status_box.info(f"🧠 选择推理: {selection_reasoning}")
                        
                        # 显示是否使用了语义匹配
                        if analysis.get("use_semantic_matching"):
                            status_box.success("🚀 使用OpenVINO语义匹配算法")
                        else:
                            status_box.warning("⚠️ 使用传统关键词匹配（建议配置OpenVINO模型以获得更好效果）")
                        
                        # 显示处理时间
                        processing_time = analysis.get("processing_time_ms", 0)
                        if processing_time > 0:
                            status_box.caption(f"⏱️ 处理时间: {processing_time:.1f}ms")
                        
                        # 显示选中的表
                        table_info = "📊 **相关数据表**:\n\n"
                        for i, table_rel in enumerate(selected_tables[:3], 1):  # 显示前3个最相关的表
                            score_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                            table_info += f"{score_emoji} **{table_rel.table_name}** (相关性: {table_rel.relevance_score:.1f})\n"
                            table_info += f"   📝 {table_rel.table_description}\n"
                            table_info += f"   💡 {table_rel.reasoning}\n"
                            
                            # 显示语义相似度（如果有）
                            if hasattr(table_rel, 'semantic_similarity') and table_rel.semantic_similarity > 0:
                                table_info += f"   🎯 语义相似度: {table_rel.semantic_similarity:.2f}\n"
                            
                            # 显示匹配的关键词
                            if hasattr(table_rel, 'keyword_matches') and table_rel.keyword_matches:
                                keywords_text = ", ".join(table_rel.keyword_matches[:3])
                                table_info += f"   🔍 关键词匹配: {keywords_text}\n"
                            
                            # 显示相关字段
                            if table_rel.matched_columns:
                                col_names = []
                                for col in table_rel.matched_columns[:3]:
                                    col_name = col.get('col', '')
                                    if 'similarity' in col:
                                        col_name += f" ({col['similarity']:.2f})"
                                    col_names.append(col_name)
                                if col_names:
                                    table_info += f"   📋 相关字段: {', '.join(col_names)}\n"
                            
                            table_info += "\n"
                        
                        status_box.markdown(table_info)
                        
                        # 显示查询意图分析
                        intent = analysis.get("intent", {})
                        if intent and any(intent.values()):
                            intent_info = "🎯 **查询特征分析**:\n"
                            intent_features = []
                            if intent.get("has_aggregation"):
                                intent_features.append("聚合计算")
                            if intent.get("has_filtering"):
                                intent_features.append("条件筛选")
                            if intent.get("has_grouping"):
                                intent_features.append("分组统计")
                            if intent.get("has_sorting"):
                                intent_features.append("排序排名")
                            if intent.get("has_time"):
                                intent_features.append("时间分析")
                            if intent.get("has_geography"):
                                intent_features.append("地理分析")
                            
                            if intent_features:
                                intent_info += f"• 检测到的查询特征: {', '.join(intent_features)}\n"
                                status_box.markdown(intent_info)
                    else:
                        error_msg = analysis.get("error", "未找到明确相关的表，将使用全部表信息")
                        status_box.warning(f"⚠️ {error_msg}")

                elif step["type"] == "result":
                    mode = "SQL"; df_result = step["df"]; sql_code = step["sql"]
                    # 保存可能性信息用于后续显示
                    selected_possibility = step.get("selected_possibility")
                    alternatives = step.get("alternatives", [])
                    status_box.update(label="✅ 执行完成", state="complete", expanded=False)
                
                elif step["type"] == "final_chat":
                    mode = "CHAT"
                    status_box.update(label="✨ 对话完成", state="complete", expanded=False)
                
                elif step["type"] == "error":
                    status_box.update(label="❌ 发生错误", state="error"); st.error(step["msg"]); st.stop()

            # --- 生成结束，开始渲染最终结果 (保持与历史记录一致的结构) ---
            if mode == "SQL":
                # 0. 表选择过程信息持久化显示
                if any(table_selection_info.values()):
                    with st.expander("🗄️ 智能表选择过程", expanded=False):
                        st.markdown("**📋 表选择详细过程**")
                        
                        # 显示初步筛选结果
                        if table_selection_info["initial_analysis"]:
                            st.markdown("**第1步：语义相似度初步筛选**")
                            st.info(table_selection_info["initial_analysis"])
                        
                        # 显示Agent推理过程
                        if table_selection_info["agent_reasoning"]:
                            st.markdown("**第2步：Agent智能筛选推理**")
                            st.success(f"🧠 推理过程: {table_selection_info['agent_reasoning']}")
                        
                        # 显示关联分析
                        if table_selection_info["join_analysis"]:
                            st.markdown("**第3步：表关联关系分析**")
                            st.info(table_selection_info["join_analysis"])
                        
                        # 显示最终选择结果
                        if table_selection_info["final_selection"]:
                            final_selection = table_selection_info["final_selection"]
                            selected_tables = final_selection.get("selected_tables", [])
                            analysis = final_selection.get("analysis", {})
                            
                            st.markdown("**🎯 最终选择结果**")
                            
                            if selected_tables:
                                # 显示选择推理
                                selection_reasoning = analysis.get("selection_reasoning", "")
                                if selection_reasoning:
                                    st.info(f"🧠 选择推理: {selection_reasoning}")
                                
                                # 显示是否使用了语义匹配
                                if analysis.get("use_semantic_matching"):
                                    st.success("🚀 使用OpenVINO语义匹配算法")
                                else:
                                    st.warning("⚠️ 使用传统关键词匹配")
                                
                                # 显示处理时间
                                processing_time = analysis.get("processing_time_ms", 0)
                                if processing_time > 0:
                                    st.caption(f"⏱️ 处理时间: {processing_time:.1f}ms")
                                
                                # 显示选中的表
                                st.markdown("**📊 相关数据表**:")
                                for i, table_rel in enumerate(selected_tables[:3], 1):
                                    score_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                                    
                                    with st.container():
                                        st.markdown(f"{score_emoji} **{table_rel.table_name}** (相关性: {table_rel.relevance_score:.1f})")
                                        st.caption(f"📝 {table_rel.table_description}")
                                        st.caption(f"💡 {table_rel.reasoning}")
                                        
                                        # 显示语义相似度（如果有）
                                        if hasattr(table_rel, 'semantic_similarity') and table_rel.semantic_similarity > 0:
                                            st.caption(f"🎯 语义相似度: {table_rel.semantic_similarity:.2f}")
                                        
                                        # 显示匹配的关键词
                                        if hasattr(table_rel, 'keyword_matches') and table_rel.keyword_matches:
                                            keywords_text = ", ".join(table_rel.keyword_matches[:3])
                                            st.caption(f"🔍 关键词匹配: {keywords_text}")
                                        
                                        # 显示相关字段
                                        if table_rel.matched_columns:
                                            col_names = []
                                            for col in table_rel.matched_columns[:3]:
                                                col_name = col.get('col', '')
                                                if 'similarity' in col:
                                                    col_name += f" ({col['similarity']:.2f})"
                                                col_names.append(col_name)
                                            if col_names:
                                                st.caption(f"📋 相关字段: {', '.join(col_names)}")
                                        
                                        if i < len(selected_tables[:3]):
                                            st.divider()
                                
                                # 显示查询意图分析
                                intent = analysis.get("intent", {})
                                if intent and any(intent.values()):
                                    st.markdown("**🎯 查询特征分析**:")
                                    intent_features = []
                                    if intent.get("has_aggregation"):
                                        intent_features.append("聚合计算")
                                    if intent.get("has_filtering"):
                                        intent_features.append("条件筛选")
                                    if intent.get("has_grouping"):
                                        intent_features.append("分组统计")
                                    if intent.get("has_sorting"):
                                        intent_features.append("排序排名")
                                    if intent.get("has_time"):
                                        intent_features.append("时间分析")
                                    if intent.get("has_geography"):
                                        intent_features.append("地理分析")
                                    
                                    if intent_features:
                                        st.info(f"• 检测到的查询特征: {', '.join(intent_features)}")
                
                # 1. 查询结果
                st.markdown("##### 🔎 查询结果")
                has_data = df_result is not None and not df_result.empty

                if has_data:
                    st.write(f"共查询到 {len(df_result)} 条数据：")
                    
                    # 添加数据筛选功能
                    if len(df_result) > 10:  # 数据量较大时提供筛选
                        with st.expander("🔍 数据筛选与排序", expanded=False):
                            # 快速筛选按钮
                            quick_filter = data_filter.create_quick_filter_buttons(df_result, "current_quick")
                            if quick_filter:
                                df_result = data_filter.apply_quick_filter(df_result, quick_filter)
                                st.success(f"已应用筛选: {quick_filter['name']}")
                            
                            # 详细筛选界面
                            filtered_df, filter_config = data_filter.create_filter_interface(df_result, "current_filter")
                            if filter_config:
                                df_result = filtered_df
                                
                                # 保存筛选配置选项
                                col1, col2 = st.columns(2)
                                with col1:
                                    filter_name = st.text_input("筛选配置名称", placeholder="输入名称保存筛选配置", key="filter_name_current")
                                with col2:
                                    if st.button("💾 保存筛选", key="save_filter_current") and filter_name:
                                        if data_filter.save_filter_config(filter_config, filter_name):
                                            st.success(f"筛选配置 '{filter_name}' 已保存")
                    
                    st.dataframe(df_result, hide_index=True)
                    
                    # 2. 可视化
                    numeric_cols = df_result.select_dtypes(include='number').columns
                    if len(df_result) > 1 and len(numeric_cols) > 0:
                        st.markdown("##### 📊 可视化")
                        
                        # 使用新的可视化引擎，传入查询上下文
                        chart_options = viz_engine.get_chart_options(df_result, prompt_input)
                        
                        # 如果有多个图表选项，让用户选择
                        if len(chart_options) > 2:
                            col1, col2 = st.columns([3, 1])
                            with col2:
                                selected_chart = st.selectbox(
                                    "图表类型", 
                                    options=[opt["type"] for opt in chart_options],
                                    format_func=lambda x: next(opt["icon"] + " " + opt["name"] for opt in chart_options if opt["type"] == x),
                                    key="current_chart_select"
                                )
                            with col1:
                                if selected_chart == "table":
                                    # 显示数据表格
                                    st.dataframe(df_result, use_container_width=True)
                                else:
                                    fig = viz_engine.create_interactive_chart(df_result, selected_chart, prompt_input)
                                    # 生成新查询图表的唯一key（选定类型）
                                    chart_key = generate_query_chart_key(prompt_input, selected_chart, df_result)
                                    st.plotly_chart(fig, use_container_width=True, key=chart_key)
                        else:
                            # 自动选择最佳图表类型，传入查询上下文
                            auto_chart_type = viz_engine.detect_chart_type(df_result, prompt_input)
                            if auto_chart_type == "table":
                                # 显示数据表格
                                st.dataframe(df_result, use_container_width=True)
                            else:
                                fig = viz_engine.create_interactive_chart(df_result, query_context=prompt_input)
                                # 生成新查询图表的唯一key（自动类型）
                                chart_key = generate_query_chart_key(prompt_input, "auto", df_result)
                                st.plotly_chart(fig, use_container_width=True, key=chart_key)
                    
                    # 3. 商业洞察
                    st.markdown("##### 💡 商业洞察")
                    insight_stream = agent.generate_insight_stream(prompt_input, df_result)
                    final_resp = st.write_stream(insight_stream)
                    
                    # 4. 硬件优化报告
                    if HARDWARE_OPTIMIZATION_AVAILABLE and hardware_optimization_result:
                        vendor = hardware_optimization_result.vendor.value
                        opt_type = hardware_optimization_result.optimization_type.value
                        
                        # 根据硬件厂商显示不同的标题和图标
                        if vendor == 'Intel':
                            report_title = "🚀 Intel平台优化报告"
                        elif vendor == 'NVIDIA':
                            report_title = "⚡ NVIDIA平台优化报告"
                        elif vendor == 'AMD':
                            report_title = "🔥 AMD平台优化报告"
                        else:
                            report_title = "🔧 硬件平台优化报告"
                        
                        with st.expander(report_title, expanded=False):
                            st.markdown("##### 📊 性能优化详情")
                            
                            # 优化指标展示
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric(
                                    "CPU性能提升", 
                                    f"{hardware_optimization_result.cpu_performance_gain:.1%}",
                                    help=f"基于{vendor}平台的CPU优化性能提升"
                                )
                            with col2:
                                st.metric(
                                    "GPU加速比", 
                                    f"{hardware_optimization_result.gpu_acceleration_gain:.2f}x",
                                    help=f"{vendor}GPU并行计算加速比"
                                )
                            with col3:
                                st.metric(
                                    "内存效率", 
                                    f"{hardware_optimization_result.memory_efficiency:.1%}",
                                    help="内存访问模式和缓存优化效率"
                                )
                            with col4:
                                st.metric(
                                    "总体加速比", 
                                    f"{hardware_optimization_result.overall_speedup:.2f}x",
                                    help="综合优化后的整体性能提升"
                                )
                            
                            # 硬件利用情况
                            hw_details = hardware_optimization_result.optimization_details
                            if hw_details:
                                st.markdown("**🔧 硬件优化详情**")
                                
                                # 显示优化策略
                                st.info(f"🎯 优化策略: {opt_type} | 硬件平台: {vendor}")
                                
                                # 显示具体优化信息
                                opt_info = []
                                if hw_details.get('cpu_cores_used', 0) > 0:
                                    opt_info.append(f"CPU核心: {hw_details['cpu_cores_used']}")
                                if hw_details.get('gpu_devices_used', 0) > 0:
                                    opt_info.append(f"GPU设备: {hw_details['gpu_devices_used']}")
                                if hw_details.get('vectorization_enabled'):
                                    opt_info.append("向量化: ✅")
                                if hw_details.get('memory_optimization'):
                                    opt_info.append("内存优化: ✅")
                                
                                if opt_info:
                                    st.caption(" | ".join(opt_info))
                                
                                # 显示优化建议
                                if hardware_optimization_result.recommendations:
                                    st.markdown("**💡 优化建议**")
                                    for rec in hardware_optimization_result.recommendations:
                                        st.write(f"• {rec}")
                    
                    
                    
                    # 3.7 其他可能的理解方式 (如果有)
                    if alternatives and len(alternatives) > 0:
                        st.markdown("##### 🤔 其他可能的理解方式")
                        
                        with st.expander(f"查看其他 {len(alternatives)} 种理解方式", expanded=False):
                            st.markdown("*点击下方按钮可以按照不同的理解方式重新执行查询*")
                            
                            for i, alt in enumerate(alternatives):
                                with st.container():
                                    col1, col2 = st.columns([4, 1])
                                    
                                    with col1:
                                        # 处理可能是字典或对象的情况
                                        if isinstance(alt, dict):
                                            rank = alt.get("rank", i + 1)
                                            natural_desc = alt.get("natural_description", alt.get("description", "未知理解方式"))
                                            confidence = alt.get("confidence", 0.0)
                                            key_interpretations = alt.get("key_interpretations", {})
                                        else:
                                            rank = getattr(alt, "rank", i + 1)
                                            natural_desc = getattr(alt, "natural_description", getattr(alt, "description", "未知理解方式"))
                                            confidence = getattr(alt, "confidence", 0.0)
                                            key_interpretations = getattr(alt, "key_interpretations", {})
                                        
                                        st.write(f"**理解方式 {rank}**:")
                                        st.write(f"📝 {natural_desc}")
                                        st.write(f"🎯 置信度: {confidence:.1%}")
                                        
                                        if key_interpretations:
                                            with st.expander("查看技术细节", expanded=False):
                                                for term, interp in key_interpretations.items():
                                                    interp_desc = interp.get('desc', '') if isinstance(interp, dict) else str(interp)
                                                    st.caption(f"• {term}: {interp_desc}")
                                    
                                    with col2:
                                        if st.button(f"🔄 选择此理解", key=f"select_alt_current_{i}"):
                                            # 重新执行这种理解方式
                                            st.session_state.prompt_trigger = natural_desc
                                            st.rerun()
                                    
                                    st.divider()
                    
                    # 4. 推荐相关问题
                    st.markdown("##### 🤔 您可能还想了解")
                    
                    # 根据配置获取推荐引擎客户端
                    use_ai_recommendations = st.session_state.config.get("enable_ai_recommendations", True)
                    use_separate_api = st.session_state.config.get("recommendation_use_separate_api", False)
                    
                    llm_client_for_rec = None
                    model_name_for_rec = None
                    rec_status = ""
                    
                    if use_ai_recommendations:
                        if use_separate_api:
                            # 使用独立的推荐API配置
                            rec_client, rec_model, rec_error = get_recommendation_client(st.session_state.config)
                            if rec_client:
                                llm_client_for_rec = rec_client
                                model_name_for_rec = rec_model
                                rec_status = "🤖 AI智能推荐 (独立API配置)"
                            else:
                                rec_status = f"📋 规则推荐 (独立API错误: {rec_error})"
                        else:
                            # 使用SQL生成的API配置
                            if hasattr(agent, 'client') and agent.client:
                                llm_client_for_rec = agent.client
                                model_name_for_rec = agent.model_name if hasattr(agent, 'model_name') else None
                                rec_status = "🤖 AI智能推荐 (共用SQL API)"
                            else:
                                rec_status = "📋 规则推荐 (SQL API不可用)"
                    else:
                        rec_status = "📋 规则推荐 (AI推荐已禁用)"
                    
                    recommendations = recommendation_engine.generate_recommendations(
                        current_query=prompt_input,
                        result_df=df_result,
                        num_recommendations=3,
                        llm_client=llm_client_for_rec,
                        model_name=model_name_for_rec
                    )
                    
                    # 显示推荐模式状态
                    st.caption(rec_status)
                    
                    if recommendations:
                        rec_cols = st.columns(len(recommendations))
                        for i, rec in enumerate(recommendations):
                            with rec_cols[i]:
                                if st.button(f"💭 {rec}", use_container_width=True, key=f"rec_current_{i}"):
                                    recommendation_engine.record_question_click(rec)
                                    st.session_state.prompt_trigger = rec
                                    st.rerun()
                    
                    # 构建完整消息体
                    # 将QueryPossibility对象转换为字典以便序列化
                    alternatives_dict = []
                    if alternatives:
                        for alt in alternatives:
                            if hasattr(alt, '__dict__'):
                                alternatives_dict.append({
                                    "rank": alt.rank,
                                    "description": alt.description,
                                    "confidence": alt.confidence,
                                    "key_interpretations": alt.key_interpretations,
                                    "ambiguity_resolutions": alt.ambiguity_resolutions,
                                    "natural_description": getattr(alt, 'natural_description', alt.description)
                                })
                            else:
                                alternatives_dict.append(alt)
                    
                    selected_possibility_dict = None
                    if selected_possibility and hasattr(selected_possibility, '__dict__'):
                        selected_possibility_dict = {
                            "rank": selected_possibility.rank,
                            "description": selected_possibility.description,
                            "confidence": selected_possibility.confidence,
                            "key_interpretations": selected_possibility.key_interpretations,
                            "ambiguity_resolutions": selected_possibility.ambiguity_resolutions,
                            "natural_description": getattr(selected_possibility, 'natural_description', selected_possibility.description)
                        }
                    
                    # 将TableRelevance对象转换为可序列化的字典格式
                    serializable_table_info = {}
                    for key, value in table_selection_info.items():
                        if key == "final_selection" and value:
                            # 处理final_selection中的selected_tables
                            selected_tables = value.get("selected_tables", [])
                            serializable_tables = []
                            for table_rel in selected_tables:
                                if hasattr(table_rel, '__dict__'):
                                    # 将TableRelevance对象转换为字典
                                    table_dict = {
                                        "table_name": table_rel.table_name,
                                        "table_description": table_rel.table_description,
                                        "relevance_score": table_rel.relevance_score,
                                        "semantic_similarity": getattr(table_rel, 'semantic_similarity', 0.0),
                                        "keyword_matches": getattr(table_rel, 'keyword_matches', []),
                                        "matched_columns": getattr(table_rel, 'matched_columns', []),
                                        "reasoning": table_rel.reasoning,
                                        "is_primary": getattr(table_rel, 'is_primary', False),
                                        "is_join_required": getattr(table_rel, 'is_join_required', False)
                                    }
                                    serializable_tables.append(table_dict)
                                else:
                                    serializable_tables.append(table_rel)
                            
                            serializable_table_info[key] = {
                                "selected_tables": serializable_tables,
                                "analysis": value.get("analysis", {})
                            }
                        else:
                            serializable_table_info[key] = value
                    
                    # 生成图表数据用于导出
                    chart_export_data = []
                    numeric_cols = df_result.select_dtypes(include='number').columns
                    if len(df_result) > 1 and len(numeric_cols) > 0:
                        # 获取图表导出数据
                        chart_export_data = viz_engine.get_chart_export_data(df_result, query_context=prompt_input)
                    
                    msg_data = {
                        "role": "assistant", 
                        "content": final_resp, 
                        "data": df_result.to_dict(orient="records"), 
                        "sql": sql_code,
                        "thought": curr_thought,
                        "selected_possibility": selected_possibility_dict,
                        "alternatives": alternatives_dict,
                        "table_selection_info": serializable_table_info,  # 使用可序列化的版本
                        "charts": chart_export_data,  # 添加图表数据
                        "recommendations": recommendations  # 保存推荐到消息中
                    }
                else:
                    # 处理空结果
                    if not st.session_state.config.get("allow_empty_results", True):
                        st.warning("⚠️ 查询结果为空，系统将根据重试机制尝试重新生成查询。")
                        final_resp = "查询结果为空，建议调整查询条件或检查数据范围。系统已记录此次查询，可尝试重新提问。"
                    else:
                        st.warning("⚠️ 查询结果为空。")
                        final_resp = "未查询到符合条件的数据。这可能是因为：\n\n1. 查询条件过于严格\n2. 数据库中不存在相关数据\n3. 时间范围或筛选条件需要调整\n\n建议尝试放宽查询条件或检查数据范围。"
                    
                    # 将TableRelevance对象转换为可序列化的字典格式（空结果情况）
                    serializable_table_info = {}
                    for key, value in table_selection_info.items():
                        if key == "final_selection" and value:
                            # 处理final_selection中的selected_tables
                            selected_tables = value.get("selected_tables", [])
                            serializable_tables = []
                            for table_rel in selected_tables:
                                if hasattr(table_rel, '__dict__'):
                                    # 将TableRelevance对象转换为字典
                                    table_dict = {
                                        "table_name": table_rel.table_name,
                                        "table_description": table_rel.table_description,
                                        "relevance_score": table_rel.relevance_score,
                                        "semantic_similarity": getattr(table_rel, 'semantic_similarity', 0.0),
                                        "keyword_matches": getattr(table_rel, 'keyword_matches', []),
                                        "matched_columns": getattr(table_rel, 'matched_columns', []),
                                        "reasoning": table_rel.reasoning,
                                        "is_primary": getattr(table_rel, 'is_primary', False),
                                        "is_join_required": getattr(table_rel, 'is_join_required', False)
                                    }
                                    serializable_tables.append(table_dict)
                                else:
                                    serializable_tables.append(table_rel)
                            
                            serializable_table_info[key] = {
                                "selected_tables": serializable_tables,
                                "analysis": value.get("analysis", {})
                            }
                        else:
                            serializable_table_info[key] = value
                    
                    msg_data = {
                        "role": "assistant", 
                        "content": final_resp, 
                        "data": [], 
                        "sql": sql_code, 
                        "thought": curr_thought,
                        "table_selection_info": serializable_table_info  # 使用可序列化的版本
                    }
                
                # 5. 原始数据折叠栏 (在生成阶段也显示出来)
                with st.expander("📝 原始 SQL 与数据导出", expanded=False):
                    st.code(sql_code, language="sql")
                    
                    # 数据导出功能
                    if has_data:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            csv_data = export_manager.export_data_to_csv(df_result, "query_result")
                            if csv_data and os.path.exists(csv_data):
                                with open(csv_data, "rb") as csv_file:
                                    st.download_button(
                                        label="📊 下载CSV",
                                        data=csv_file.read(),
                                        file_name=os.path.basename(csv_data),
                                        mime="text/csv",
                                        key="csv_download_current"
                                    )
                        
                        with col2:
                            excel_data = export_manager.export_data_to_excel(df_result, "query_result")
                            if excel_data and os.path.exists(excel_data):
                                with open(excel_data, "rb") as excel_file:
                                    st.download_button(
                                        label="📈 下载Excel",
                                        data=excel_file.read(),
                                        file_name=os.path.basename(excel_data),
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        key="excel_download_current"
                                    )

            else:
                # 聊天模式 - 使用上下文感知的提示
                final_resp = st.write_stream(agent.chat_stream(final_prompt, messages[:-1]))
                msg_data = {"role": "assistant", "content": final_resp, "thought": curr_thought}

            end_time = time.perf_counter()
            st.session_state.last_total_latency = (end_time - start_time) * 1000
            
            # 技术卓越性后处理 - 后端功能启用
            if TECHNICAL_EXCELLENCE_AVAILABLE:
                try:
                    # 记录操作性能
                    total_latency = (end_time - start_time) * 1000
                    
                    # 确定操作类型
                    operation_type = "text2sql"
                    if df_result is not None and len(df_result) > 0:
                        operation_type = "sql_execution"
                    elif curr_thought:
                        operation_type = "reasoning"
                    
                    # 记录性能数据（后端处理）
                    tech_manager.record_operation_performance(
                        operation_type=operation_type,
                        operation_id=f"query_{int(time.time())}",
                        latency_ms=total_latency,
                        error_occurred=False,
                        cache_hit=False,  # 可以根据实际情况调整
                        input_size=len(prompt_input.encode('utf-8')),
                        context={
                            'has_sql': sql_code is not None,
                            'has_data': df_result is not None,
                            'result_rows': len(df_result) if df_result is not None else 0,
                            'query_complexity': estimated_result_size if 'estimated_result_size' in locals() else 100
                        }
                    )
                    
                except Exception as e:
                    logger.warning(f"技术卓越性后处理失败: {e}")
            
            messages.append(msg_data)
            
            # 🧠 更新上下文记忆
            if CONTEXT_MEMORY_AVAILABLE and st.session_state.get('context_memory_enabled', True):
                try:
                    # 获取最终的回复内容
                    final_response = msg_data.get("content", "")
                    update_context_after_response(prompt_input, final_response)
                except Exception as e:
                    # 记录错误但不中断流程
                    logger.warning(f"上下文记忆更新失败: {e}")
            
            update_session_messages(st.session_state.current_session_id, messages, st.session_state.history)
            
            update_monitor()
            st.rerun()

        except Exception as e:
            status_box.update(label="❌ 致命错误", state="error")
            st.error(str(e))
            
            # 🧠 跟踪错误
            if CONTEXT_MEMORY_AVAILABLE and st.session_state.get('context_memory_enabled', True):
                try:
                    context_integration = get_context_integration()
                    context_integration.track_error_resolution(
                        str(e), 
                        "显示错误信息给用户", 
                        success=False
                    )
                except Exception as context_error:
                    # 避免错误处理中的错误导致系统崩溃
                    logger.warning(f"错误跟踪失败: {context_error}")
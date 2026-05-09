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
    from hardware.universal_hardware_optimizer import (
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
    from prompt_template_system import PromptTemplateManager, PromptMode, LLMProvider, EnhancedPromptBuilder
    from prompt_config_ui import PromptConfigUI
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

# 🎨 UI面板组件
try:
    from ui.panels import render_context_memory_panel
    UI_PANELS_AVAILABLE = True
except ImportError:
    UI_PANELS_AVAILABLE = False



# 性能优化配置
st.set_page_config(
    page_title="DeepInsight-text2sql", 
    layout="wide", 
    page_icon="assets/团队Logo.png",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "DeepInsight - 基于OpenVINO™的智能零售决策系统"
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

# --- CSS样式和JavaScript加载 (从外部文件加载) ---
try:
    from ui.style_loader import inject_styles_and_scripts
    inject_styles_and_scripts()
except ImportError as e:
    # 如果导入失败，使用内联样式回退
    st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    .main .block-container { padding-top: 1rem !important; max-width: 1200px !important; }
    .stChatMessage { padding: 1.2rem; border-radius: 15px; border: 1px solid #eef0f3; background: white; }
    h5 { color: #0068B5; font-weight: 600; }
    .thought-persist { background: #f0f7ff; padding: 16px; border-radius: 12px; border-left: 5px solid #0068B5; margin-bottom: 18px; }
    .thought-box { background: #f8f9fa; padding: 14px; border-radius: 10px; border-left: 4px solid #6c757d; margin: 12px 0; }
    .monitor-box { background: white; padding: 15px; border-radius: 10px; border: 1px solid #eee; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)
    print(f"⚠️ 样式加载模块未找到，使用回退样式: {e}")

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
        
        # 🗄️ 数据库选择器 - 添加在高级配置页面顶部
        available_dbs = config_ui.manager.get_available_databases()
        if len(available_dbs) > 1:
            st.markdown("### 🗄️ 数据库配置")
            current_db = config_ui.manager.get_current_database()
            
            db_display_names = {
                'northwind': '🍎 Northwind (食品贸易)',
                'adventureworks': '🚴 AdventureWorks (自行车制造)'
            }
            
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_db = st.selectbox(
                    "选择数据库",
                    options=list(available_dbs.keys()),
                    index=list(available_dbs.keys()).index(current_db) if current_db in available_dbs else 0,
                    format_func=lambda x: db_display_names.get(x, x),
                    help="切换数据库将加载对应的业务上下文、术语词典和示例查询",
                    key="advanced_db_selector"
                )
            
            with col2:
                if selected_db != current_db:
                    if st.button("🔄 切换", type="primary", use_container_width=True, key="switch_db_btn"):
                        success = config_ui.manager.switch_database(selected_db)
                        if success:
                            st.success(f"✅ 已切换到 {db_display_names.get(selected_db, selected_db)}")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("❌ 数据库切换失败")
            
            st.divider()
        
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
        st.image("assets/intel.svg", width=144)
        # st.image("assets/团队logo.png", width=144)
    else:
        st.markdown("### DeepInsight")
    
    # 🧠 渲染上下文记忆UI (使用模块化面板)
    if UI_PANELS_AVAILABLE:
        render_context_memory_panel()
    elif CONTEXT_MEMORY_AVAILABLE:
        # 回退到简化版本
        with st.expander("🧠 上下文记忆系统", expanded=False):
            st.info("上下文记忆系统已启用")
            memory_enabled = st.session_state.get('context_memory_enabled', True)
            st.write(f"状态: {'✅ 已启用' if memory_enabled else '⏸️ 已禁用'}")
    else:
        with st.expander("🧠 上下文记忆系统", expanded=False):
            st.warning("上下文记忆系统当前不可用")


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
        
        # ⭐ Reasoner 自愈模式配置
        st.markdown("**🧠 智能自愈配置**")
        use_reasoner_for_healing = st.checkbox(
            "启用 Reasoner 自愈模式",
            value=st.session_state.config.get("use_reasoner_for_healing", True),
            help="启用后，SQL 生成失败时将使用 DeepSeek Reasoner 进行智能修复"
        )
        
        if use_reasoner_for_healing:
            reasoner_model = st.text_input(
                "Reasoner 模型名称",
                st.session_state.config.get("reasoner_model", "deepseek-reasoner"),
                help="推理模型名称，用于自愈重试时的 SQL 生成"
            )
        else:
            reasoner_model = st.session_state.config.get("reasoner_model", "deepseek-reasoner")
            st.info("💡 禁用后，自愈时将继续使用主模型")

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
            "recommendation_model_name": rec_model,
            # ⭐ Reasoner 自愈模式配置
            "use_reasoner_for_healing": use_reasoner_for_healing,
            "reasoner_model": reasoner_model
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
            st.metric("端到端延迟", f"{total_lat/1000:.2f} s") # 这里我从ms转换成了s，美观一些。
        
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
            config=cfg,  # 🧠 传递完整配置给Prompt模板系统
            # ⭐ Reasoner 自愈模式配置
            reasoner_model=cfg.get("reasoner_model", "deepseek-reasoner"),
            use_reasoner_for_healing=cfg.get("use_reasoner_for_healing", True)
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
    # 主标题区域 - 现代化渐变设计
    st.markdown("""
    <style>
        @keyframes shimmer {
            0% { background-position: -200% center; }
            100% { background-position: 200% center; }
        }
        .hero-title {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(135deg, #0068B5 0%, #00a8e8 50%, #0068B5 100%);
            background-size: 200% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: shimmer 3s linear infinite;
            margin: 0;
            letter-spacing: -1px;
        }
        .hero-badges {
            display: flex;
            justify-content: center;
            gap: 12px;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        .hero-badge {
            padding: 6px 16px;
            border-radius: 50px;
            font-size: 0.85rem;
            font-weight: 600;
            transition: transform 0.3s ease;
        }
        .hero-badge:hover { transform: translateY(-2px); }
        .badge-blue { background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); color: #1d4ed8; }
        .badge-green { background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); color: #15803d; }
        .badge-orange { background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); color: #b45309; }
    </style>
    <div style="text-align: center; padding: 30px 20px; margin-bottom: 20px;">
        <h1 class="hero-title">DeepInsight-text2sql</h1>
        <p style="font-size: 1.15rem; color: #64748b; margin-top: 16px; line-height: 1.6;">
            基于 OpenVINO™ 的本地化智能零售决策系统
        </p>
        <div class="hero-badges">
            <span class="hero-badge badge-blue">🚀 全本地运行</span>
            <span class="hero-badge badge-green">🔒 隐私安全</span>
            <span class="hero-badge badge-orange">⚡ 极速推理</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 显示上下文记忆状态 - 使用最新的状态
    if CONTEXT_MEMORY_AVAILABLE:
        current_memory_enabled = st.session_state.get('context_memory_enabled', True)
        if current_memory_enabled:
            st.markdown("""
            <div style="max-width: 650px; margin: 0 auto 25px; padding: 14px 24px; border-radius: 14px;
                        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
                        border: 1px solid rgba(34, 197, 94, 0.3); text-align: center;
                        display: flex; align-items: center; justify-content: center; gap: 12px;">
                <span style="font-size: 1.4rem;">🧠</span>
                <span style="color: #166534;"><strong>上下文记忆已启用</strong> — AI将记住对话历史</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="max-width: 650px; margin: 0 auto 25px; padding: 14px 24px; border-radius: 14px;
                        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
                        border: 1px solid rgba(245, 158, 11, 0.3); text-align: center;
                        display: flex; align-items: center; justify-content: center; gap: 12px;">
                <span style="font-size: 1.4rem;">💭</span>
                <span style="color: #92400e;"><strong>上下文记忆已禁用</strong> — AI将不会记住历史</span>
            </div>
            """, unsafe_allow_html=True)
    
    # 智能推荐问题标题
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h4 style="color: #1e293b; font-weight: 700; display: inline-flex; align-items: center; gap: 8px;
                   padding-bottom: 10px; border-bottom: 3px solid; border-image: linear-gradient(90deg, #0068B5, #00a8e8) 1;">
            💡 您可能想问...
        </h4>
    </div>
    """, unsafe_allow_html=True)
    
    # 推荐问题 - 2x2布局更美观
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 家具类产品的平均利润率是多少？", use_container_width=True):
            st.session_state.prompt_trigger = "家具类产品的平均利润率是多少"
            st.rerun()
        if st.button("🏆 哪些供应商提供的产品种类超过了 3 种？", use_container_width=True):
            st.session_state.prompt_trigger = "哪些供应商提供的产品种类超过了 3 种？"
            st.rerun()
    with col2:
        if st.button("📈 库存积压最严重的TOP5产品是？", use_container_width=True):
            st.session_state.prompt_trigger = "库存积压最严重的TOP5产品是？"
            st.rerun()
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
            
            # 🆕 0. 知识检索与Agent思考过程展示（历史消息）
            if "knowledge_retrieval" in msg and msg["knowledge_retrieval"]:
                retrieval_info = msg["knowledge_retrieval"]
                with st.expander("🧠 Agent 思考与知识检索", expanded=False):
                    st.markdown("**📊 二阶段混合检索过程**")
                    
                    # 第1步：粗排候选表
                    st.markdown("**第1步：向量粗排 (OpenVINO加速)**")
                    if retrieval_info.get('rough_candidates_display'):
                        st.info(retrieval_info['rough_candidates_display'])
                    else:
                        st.caption("无粗排结果")
                    
                    # 第2步：精排核心表
                    st.markdown("**第2步：LLM精排 (核心表筛选)**")
                    if retrieval_info.get('core_tables_display'):
                        st.success(f"✅ 核心表: {retrieval_info['core_tables_display']}")
                    else:
                        st.caption("无精排结果")
                    
                    # 第3步：术语匹配
                    st.markdown("**第3步：业务术语匹配**")
                    if retrieval_info.get('matched_terms_display') and retrieval_info['matched_terms_display'] != "无匹配术语":
                        st.info(retrieval_info['matched_terms_display'])
                    else:
                        st.caption("未匹配到相关术语")
                    
                    # 第4步：Few-Shot示例
                    st.markdown("**第4步：Few-Shot 示例匹配**")
                    if retrieval_info.get('matched_examples_display') and retrieval_info['matched_examples_display'] != "无匹配示例":
                        st.info(retrieval_info['matched_examples_display'])
                    else:
                        st.caption("未匹配到相关示例")
                    
                    # 性能指标
                    if retrieval_info.get('metrics_display'):
                        st.caption(f"⏱️ {retrieval_info['metrics_display']}")
            
            # 1. 表选择过程信息持久化显示 (历史消息)
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

            # 1.5 Token 与缓存信息（历史持久化显示）
            if msg.get("from_cache"):
                st.caption("♻️ 本次结果来自查询缓存（未触发新的 LLM 调用）")

            if msg.get("cumulative_token_usage"):
                usage = msg.get("cumulative_token_usage", {})
                st.caption(
                    f"📈 累计Token: prompt={usage.get('prompt_tokens', 0)}, "
                    f"completion={usage.get('completion_tokens', 0)}, "
                    f"total={usage.get('total_tokens', 0)}, "
                    f"calls={usage.get('call_count', 0)}"
                )

            if msg.get("token_usage") and not msg.get("from_cache"):
                usage = msg.get("token_usage", {})
                st.caption(
                    f"📊 最近一次调用Token: prompt={usage.get('prompt_tokens', 0)}, "
                    f"completion={usage.get('completion_tokens', 0)}, "
                    f"total={usage.get('total_tokens', 0)}"
                )
            
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
            # 推理使用上下文增强后的 prompt；缓存键使用原始用户问题，确保命中稳定
            stream_gen = agent.generate_and_execute_stream(
                final_prompt,
                messages[:-1],
                cache_query_key=prompt_input,
            )
            final_resp, df_result, sql_code, mode = "", None, None, "CHAT"
            selected_possibility, alternatives = None, []
            latest_token_usage = None
            cumulative_token_usage = None
            result_from_cache = False
            # 表选择信息初始化（RAG重新设计后可能为空）
            table_selection_info = {
                "initial_analysis": None,
                "agent_reasoning": None,
                "join_analysis": None,
                "final_selection": None
            }
            # 🆕 知识检索信息初始化（二阶段检索结果展示）
            knowledge_retrieval_info = {
                "rough_candidates": [],  # 粗排候选表
                "core_tables": [],       # 精排核心表
                "matched_terms": [],     # 匹配的术语
                "matched_examples": [],  # 匹配的示例
                "metrics": {}            # 性能指标
            }
            step_count = 0 

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

                elif step["type"] == "token_usage":
                    usage = step.get("usage", {})
                    latest_token_usage = usage
                    status_box.caption(
                        f"📊 本次Token: prompt={usage.get('prompt_tokens', 0)}, "
                        f"completion={usage.get('completion_tokens', 0)}, "
                        f"total={usage.get('total_tokens', 0)}"
                    )

                elif step["type"] == "cumulative_token_usage":
                    usage = step.get("usage", {})
                    cumulative_token_usage = usage
                    status_box.caption(
                        f"📈 累计Token: prompt={usage.get('prompt_tokens', 0)}, "
                        f"completion={usage.get('completion_tokens', 0)}, "
                        f"total={usage.get('total_tokens', 0)}, "
                        f"calls={usage.get('call_count', 0)}"
                    )

                elif step["type"] == "rag_enhancement":
                    # 显示 RAG 语义增强信息
                    pattern_count = step.get("pattern_count", 0)
                    rule_count = step.get("rule_count", 0)
                    
                    with status_box.expander(f"🧠 RAG 语义增强 (匹配 {pattern_count} 模式, {rule_count} 规则)", expanded=False):
                        st.markdown(step.get("content", ""))

                elif step["type"] == "result":
                    mode = "SQL"; df_result = step["df"]; sql_code = step["sql"]
                    result_from_cache = bool(step.get("from_cache", False))
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
            
            # 🆕 立即捕获检索信息（在任何可能的rerun之前）
            retrieval_display = None
            if hasattr(agent, 'get_retrieval_display_info'):
                retrieval_display = agent.get_retrieval_display_info()
                
            if mode == "SQL":
                # 🆕 0.0 知识检索与Agent思考过程展示（二阶段混合检索）
                with st.expander("🧠 Agent 思考与知识检索", expanded=False):
                    if retrieval_display:
                        st.markdown("**📊 二阶段混合检索过程**")
                        
                        # 第1步：粗排候选表
                        st.markdown("**第1步：向量粗排 (OpenVINO加速)**")
                        if retrieval_display.get('rough_candidates_display'):
                            st.info(retrieval_display['rough_candidates_display'])
                        else:
                            st.caption("无粗排结果")
                        
                        # 第2步：精排核心表
                        st.markdown("**第2步：LLM精排 (核心表筛选)**")
                        if retrieval_display.get('core_tables_display'):
                            st.success(f"✅ 核心表: {retrieval_display['core_tables_display']}")
                        else:
                            st.caption("无精排结果")
                        
                        # 第3步：术语匹配
                        st.markdown("**第3步：业务术语匹配**")
                        if retrieval_display.get('matched_terms_display') and retrieval_display['matched_terms_display'] != "无匹配术语":
                            st.info(retrieval_display['matched_terms_display'])
                        else:
                            st.caption("未匹配到相关术语")
                        
                        # 第4步：Few-Shot示例
                        st.markdown("**第4步：Few-Shot 示例匹配**")
                        if retrieval_display.get('matched_examples_display') and retrieval_display['matched_examples_display'] != "无匹配示例":
                            st.info(retrieval_display['matched_examples_display'])
                        else:
                            st.caption("未匹配到相关示例")
                        
                        # 性能指标
                        if retrieval_display.get('metrics_display'):
                            st.caption(f"⏱️ {retrieval_display['metrics_display']}")
                    else:
                        st.caption("ℹ️ 二阶段知识检索未启用或无可用数据")
                
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
                if result_from_cache:
                    st.caption("♻️ 本次结果来自查询缓存（未触发新的 LLM 调用）")
                if cumulative_token_usage:
                    st.caption(
                        f"📈 累计Token: prompt={cumulative_token_usage.get('prompt_tokens', 0)}, "
                        f"completion={cumulative_token_usage.get('completion_tokens', 0)}, "
                        f"total={cumulative_token_usage.get('total_tokens', 0)}, "
                        f"calls={cumulative_token_usage.get('call_count', 0)}"
                    )
                if latest_token_usage and not result_from_cache:
                    st.caption(
                        f"📊 最近一次调用Token: prompt={latest_token_usage.get('prompt_tokens', 0)}, "
                        f"completion={latest_token_usage.get('completion_tokens', 0)}, "
                        f"total={latest_token_usage.get('total_tokens', 0)}"
                    )
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
                        "from_cache": result_from_cache,
                        "token_usage": latest_token_usage,
                        "cumulative_token_usage": cumulative_token_usage,
                        "selected_possibility": selected_possibility_dict,
                        "alternatives": alternatives_dict,
                        "table_selection_info": serializable_table_info,  # 使用可序列化的版本
                        "knowledge_retrieval": retrieval_display,  # 🆕 保存知识检索信息
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
                        "from_cache": result_from_cache,
                        "token_usage": latest_token_usage,
                        "cumulative_token_usage": cumulative_token_usage,
                        "table_selection_info": serializable_table_info,  # 使用可序列化的版本
                        "knowledge_retrieval": retrieval_display  # 🆕 保存知识检索信息
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

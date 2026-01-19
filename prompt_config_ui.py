"""
Intel® DeepInsight - Prompt配置管理UI组件
提供用户友好的界面来配置业务上下文、术语词典和示例查询
"""

import streamlit as st
import pandas as pd
import time
from typing import Dict, List, Optional
import io
import csv
from prompt_template_system import (
    PromptTemplateManager, PromptMode, LLMProvider, 
    BusinessContext, ExampleQuery
)

class PromptConfigUI:
    """Prompt配置UI管理器"""
    
    def __init__(self):
        self.manager = PromptTemplateManager()
    
    def render_config_sidebar(self):
        """渲染配置侧边栏"""
        with st.sidebar:
            st.header("🔧 Prompt配置")
            
            # 配置摘要
            summary = self.manager.get_config_summary()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("业务上下文", 
                         "已配置" if summary['business_context_configured'] else "未配置",
                         f"{summary['business_context_length']}/2000字符")
            with col2:
                st.metric("术语词典", f"{summary['term_dictionary_size']}个术语")
            
            st.metric("示例查询", f"{summary['example_queries_count']}个示例")
            
            # 快速配置按钮
            if st.button("📝 配置业务上下文", use_container_width=True):
                st.session_state.show_business_config = True
            
            if st.button("📚 管理术语词典", use_container_width=True):
                st.session_state.show_term_config = True
            
            if st.button("💡 管理示例查询", use_container_width=True):
                st.session_state.show_example_config = True
    
    def render_business_context_config(self):
        """渲染业务上下文配置"""
        st.subheader("📝 业务上下文配置")
        
        # 配置示例按钮区域
        st.markdown("### 🎯 快速配置示例")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🛒 电商零售", use_container_width=True):
                self._apply_ecommerce_config()
                st.success("✅ 已应用电商零售配置")
                st.rerun()
        
        with col2:
            if st.button("🏭 制造业", use_container_width=True):
                self._apply_manufacturing_config()
                st.success("✅ 已应用制造业配置")
                st.rerun()
        
        with col3:
            if st.button("🏦 金融服务", use_container_width=True):
                self._apply_finance_config()
                st.success("✅ 已应用金融服务配置")
                st.rerun()
        
        with col4:
            if st.button("🎓 教育培训", use_container_width=True):
                self._apply_education_config()
                st.success("✅ 已应用教育培训配置")
                st.rerun()

        with st.expander("配置说明", expanded=False):
            st.info("""
            **业务上下文配置说明：**
            - **行业术语**：输入您所在行业的专业术语，用逗号分隔
            - **业务规则**：描述您的业务特殊规则和约定
            - **数据特征**：说明您的数据特点和结构
            - **分析重点**：指明您最关注的分析维度
            
            **使用建议：**
            - 每个字段建议控制在500字符以内
            - 使用简洁明确的描述
            - 可以留空不填，系统会使用默认配置
            """)
        
        # 当前配置显示
        current_context = self.manager.business_context
        
        col1, col2 = st.columns(2)
        
        with col1:
            industry_terms = st.text_area(
                "行业术语",
                value=current_context.industry_terms,
                height=100,
                help="输入行业专业术语，用逗号分隔",
                placeholder="例如：零售业、电商、供应链、库存周转率、客单价"
            )
            
            business_rules = st.text_area(
                "业务规则",
                value=current_context.business_rules,
                height=100,
                help="描述业务特殊规则和约定",
                placeholder="例如：关注季节性销售趋势，重视客户留存率分析"
            )
        
        with col2:
            data_characteristics = st.text_area(
                "数据特征",
                value=current_context.data_characteristics,
                height=100,
                help="说明数据特点和结构",
                placeholder="例如：包含订单、产品、客户、员工等核心业务数据"
            )
            
            analysis_focus = st.text_area(
                "分析重点",
                value=current_context.analysis_focus,
                height=100,
                help="指明最关注的分析维度",
                placeholder="例如：销售分析、客户分析、产品分析、运营效率"
            )
        
        # 字符统计
        total_chars = len(industry_terms + business_rules + data_characteristics + analysis_focus)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            if total_chars > 2000:
                st.error(f"总字符数超限：{total_chars}/2000")
            else:
                st.success(f"字符数：{total_chars}/2000")
        
        with col2:
            if st.button("💾 保存配置", type="primary"):
                try:
                    self.manager.update_business_context(
                        industry_terms=industry_terms,
                        business_rules=business_rules,
                        data_characteristics=data_characteristics,
                        analysis_focus=analysis_focus
                    )
                    st.success("✅ 业务上下文配置已保存")
                    st.rerun()
                except ValueError as e:
                    st.error(f"❌ 配置保存失败：{e}")
        
        with col3:
            if st.button("🔄 重置配置"):
                try:
                    self.manager.reset_to_default()
                    st.success("✅ 配置已重置为默认值")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 重置失败: {e}")
    
    def render_term_dictionary_config(self):
        """渲染术语词典配置"""
        st.subheader("📚 术语词典管理")
        
        # 上传CSV文件
        with st.expander("📤 上传术语词典", expanded=True):
            st.info("""
            **CSV文件格式要求：**
            - 必须包含 `term` 和 `explanation` 两列
            - `term`：术语名称
            - `explanation`：术语解释
            - 编码：UTF-8
            """)
            
            uploaded_file = st.file_uploader(
                "选择CSV文件",
                type=['csv'],
                help="上传包含术语和解释的CSV文件"
            )
            
            if uploaded_file is not None:
                try:
                    # 读取CSV文件
                    df = pd.read_csv(uploaded_file)
                    
                    # 验证列名
                    if 'term' not in df.columns or 'explanation' not in df.columns:
                        st.error("❌ CSV文件必须包含 'term' 和 'explanation' 列")
                    else:
                        st.success(f"✅ 成功读取 {len(df)} 个术语")
                        
                        # 预览数据
                        st.dataframe(df.head(10), use_container_width=True)
                        
                        if st.button("💾 导入术语词典", type="primary"):
                            # 使用固定的文件名确保一致性
                            import os
                            os.makedirs("data", exist_ok=True)
                            csv_path = "data/uploaded_terms_user_uploaded_terms.csv"
                            
                            # 保存文件
                            with open(csv_path, 'wb') as f:
                                f.write(uploaded_file.getbuffer())
                            
                            # 加载术语词典
                            self.manager.load_term_dictionary(csv_path)
                            st.success("✅ 术语词典已导入并保存")
                            st.rerun()
                
                except Exception as e:
                    st.error(f"❌ 文件读取失败：{e}")
        
        # 当前术语词典显示和管理
        if self.manager.term_dictionary.terms:
            with st.expander("📖 当前术语词典", expanded=True):
                # 搜索功能
                search_keyword = st.text_input("🔍 搜索术语", placeholder="输入关键词搜索术语或解释")
                
                if search_keyword:
                    terms_to_show = self.manager.search_terms(search_keyword)
                    st.info(f"找到 {len(terms_to_show)} 个相关术语")
                else:
                    terms_to_show = self.manager.term_dictionary.terms
                
                if terms_to_show:
                    terms_df = pd.DataFrame([
                        {"术语": term, "解释": explanation}
                        for term, explanation in terms_to_show.items()
                    ])
                    
                    st.dataframe(terms_df, use_container_width=True)
                    
                    # 下载当前词典
                    csv_buffer = io.StringIO()
                    terms_df.to_csv(csv_buffer, index=False, encoding='utf-8')
                    
                    st.download_button(
                        label="📥 下载当前词典",
                        data=csv_buffer.getvalue(),
                        file_name="current_term_dictionary.csv",
                        mime="text/csv"
                    )
        
        # 术语管理标签页
        tab1, tab2, tab3 = st.tabs(["➕ 添加术语", "✏️ 修改术语", "🗑️ 删除术语"])
        
        with tab1:
            self._render_add_term_tab()
        
        with tab2:
            self._render_edit_term_tab()
        
        with tab3:
            self._render_delete_term_tab()
    
    def _render_add_term_tab(self):
        """渲染添加术语标签页"""
        st.markdown("### ➕ 添加新术语")
        
        with st.form("add_term_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_term = st.text_input("术语名称", placeholder="例如：客单价")
            
            with col2:
                new_explanation = st.text_input("术语解释", placeholder="例如：平均每个客户的消费金额")
            
            submitted = st.form_submit_button("➕ 添加术语", type="primary")
            
            if submitted:
                if new_term and new_explanation:
                    try:
                        if new_term in self.manager.term_dictionary.terms:
                            st.warning(f"⚠️ 术语 '{new_term}' 已存在")
                            if st.button("🔄 更新解释", key="update_existing"):
                                self.manager.update_term(new_term, new_explanation)
                                st.success("✅ 术语解释已更新")
                                st.rerun()
                        else:
                            self.manager.add_term(new_term, new_explanation)
                            st.success(f"✅ 已添加术语: {new_term}")
                            st.rerun()
                    except Exception as e:
                        st.error(f"❌ 添加失败: {e}")
                else:
                    st.error("❌ 请填写术语名称和解释")
    
    def _render_edit_term_tab(self):
        """渲染修改术语标签页"""
        st.markdown("### ✏️ 修改术语")
        
        if not self.manager.term_dictionary.terms:
            st.info("暂无术语可修改")
            return
        
        # 选择要修改的术语
        term_to_edit = st.selectbox(
            "选择要修改的术语",
            options=list(self.manager.term_dictionary.terms.keys()),
            format_func=lambda x: f"{x} - {self.manager.term_dictionary.terms[x][:50]}..."
        )
        
        if term_to_edit:
            current_explanation = self.manager.term_dictionary.terms[term_to_edit]
            
            with st.form("edit_term_form"):
                st.text_input("术语名称", value=term_to_edit, disabled=True)
                new_explanation = st.text_area(
                    "术语解释",
                    value=current_explanation,
                    height=100
                )
                
                submitted = st.form_submit_button("💾 保存修改", type="primary")
                
                if submitted:
                    if new_explanation and new_explanation != current_explanation:
                        try:
                            self.manager.update_term(term_to_edit, new_explanation)
                            st.success("✅ 术语解释已更新")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ 修改失败: {e}")
                    else:
                        st.info("ℹ️ 解释内容没有变化")
    
    def _render_delete_term_tab(self):
        """渲染删除术语标签页"""
        st.markdown("### 🗑️ 删除术语")
        
        if not self.manager.term_dictionary.terms:
            st.info("暂无术语可删除")
            return
        
        # 选择要删除的术语
        terms_to_delete = st.multiselect(
            "选择要删除的术语（可多选）",
            options=list(self.manager.term_dictionary.terms.keys()),
            format_func=lambda x: f"{x} - {self.manager.term_dictionary.terms[x][:50]}..."
        )
        
        if terms_to_delete:
            st.warning(f"⚠️ 确定要删除以下 {len(terms_to_delete)} 个术语吗？")
            
            for term in terms_to_delete:
                st.write(f"• **{term}**: {self.manager.term_dictionary.terms[term]}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🗑️ 确认删除", type="primary"):
                    try:
                        for term in terms_to_delete:
                            self.manager.delete_term(term)
                        
                        st.success(f"✅ 已删除 {len(terms_to_delete)} 个术语")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 删除失败: {e}")
            
            with col2:
                if st.button("❌ 取消"):
                    st.rerun()
    
    def render_example_queries_config(self):
        """渲染示例查询配置"""
        st.subheader("💡 示例查询管理")
        
        # 添加新示例
        with st.expander("➕ 添加示例查询", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                new_query = st.text_area(
                    "查询内容",
                    height=80,
                    placeholder="例如：查看去年销售额最高的产品"
                )
                new_category = st.selectbox(
                    "查询类别",
                    ["销售分析", "客户分析", "产品分析", "运营分析", "财务分析", "其他"],
                    help="选择查询所属的业务类别"
                )
            
            with col2:
                new_description = st.text_area(
                    "查询描述",
                    height=80,
                    placeholder="例如：产品销售排名分析，帮助了解热销产品"
                )
                new_sql_pattern = st.text_input(
                    "SQL模式（可选）",
                    placeholder="例如：SELECT ... FROM products ORDER BY sales DESC"
                )
            
            if st.button("➕ 添加示例", type="primary") and new_query and new_category:
                self.manager.add_example_query(
                    query=new_query,
                    category=new_category,
                    sql_pattern=new_sql_pattern,
                    description=new_description
                )
                st.success("✅ 示例查询已添加")
                st.rerun()
        
        # 当前示例查询显示
        if self.manager.example_queries:
            with st.expander("📋 当前示例查询", expanded=True):
                for i, example in enumerate(self.manager.example_queries):
                    with st.container():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.write(f"**{example.query}**")
                            if example.description:
                                st.caption(example.description)
                        
                        with col2:
                            st.badge(example.category)
                        
                        with col3:
                            if st.button("🗑️", key=f"delete_{i}", help="删除此示例"):
                                if self.manager.remove_example_query(i):
                                    st.success("✅ 示例查询已删除")
                                    time.sleep(0.5)
                                    st.rerun()
                                else:
                                    st.error("❌ 删除失败")
                        
                        if example.sql_pattern:
                            st.code(example.sql_pattern, language="sql")
                        
                        st.divider()
    
    def render_prompt_preview(self, user_query: str = ""):
        """渲染Prompt预览"""
        st.subheader("👁️ Prompt预览")
        
        col1, col2 = st.columns(2)
        
        with col1:
            mode = st.selectbox(
                "Prompt模式",
                [PromptMode.PROFESSIONAL, PromptMode.FLEXIBLE],
                index=1,  # 默认选择智能查询
                format_func=lambda x: "标准查询" if x == PromptMode.PROFESSIONAL else "智能查询"
            )
        
        with col2:
            llm_provider = st.selectbox(
                "LLM提供商",
                [LLMProvider.DEEPSEEK, LLMProvider.OPENAI, LLMProvider.CLAUDE, LLMProvider.QWEN],
                format_func=lambda x: x.value.upper()
            )
        
        test_query = st.text_input(
            "测试查询",
            value=user_query or "分析一下最近三个月的销售趋势",
            help="输入一个测试查询来预览生成的Prompt"
        )
        
        if st.button("🔍 生成预览", type="primary"):
            try:
                prompt = self.manager.build_complete_prompt(
                    user_query=test_query,
                    schema_info="orders表包含订单信息，products表包含产品信息...",
                    rag_context="销售趋势分析通常关注时间序列变化...",
                    mode=mode,
                    llm_provider=llm_provider
                )
                
                st.text_area(
                    "生成的完整Prompt",
                    value=prompt,
                    height=400,
                    help="这是发送给LLM的完整Prompt内容"
                )
                
                # 统计信息
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("总字符数", len(prompt))
                with col2:
                    st.metric("总词数", len(prompt.split()))
                with col3:
                    # 估算token数（粗略估计：中文1字符≈1token，英文1词≈1.3token）
                    estimated_tokens = len([c for c in prompt if ord(c) > 127]) + len(prompt.split()) * 1.3
                    st.metric("估算Token数", f"{estimated_tokens:.0f}")
            
            except Exception as e:
                st.error(f"❌ Prompt生成失败：{e}")
    
    def _apply_ecommerce_config(self):
        """应用电商零售配置示例"""
        self.manager.update_business_context(
            industry_terms="电商、零售、供应链、库存周转率、客单价、GMV、SKU、转化率、复购率、同比、环比、ROI、ARPU、LTV",
            business_rules="关注季节性销售趋势，重视客户留存率分析，注重产品类别间的关联销售，考虑地域差异对销售的影响，重点监控库存周转和现金流",
            data_characteristics="包含订单、产品、客户、员工、供应商等核心业务数据，数据时间跨度较长，涵盖多个销售渠道和地域市场，包含用户行为轨迹数据",
            analysis_focus="销售分析、客户分析、产品分析、运营效率、地域分析、时间趋势分析、用户行为分析、营销效果分析"
        )
        
        # 添加电商相关示例查询
        ecommerce_examples = [
            ("查看最近30天销售额排名前10的产品", "产品分析", "产品热销排行分析"),
            ("分析不同地区的客户购买偏好", "客户分析", "地域客户行为差异分析"),
            ("统计各渠道的转化率和ROI", "营销分析", "多渠道营销效果对比"),
            ("查看库存周转率低于平均值的产品", "库存分析", "滞销产品识别和库存优化"),
            ("分析用户复购率和生命周期价值", "客户分析", "客户价值和忠诚度分析")
        ]
        
        # 清空现有示例并添加新的
        self.manager.example_queries = []
        for query, category, description in ecommerce_examples:
            self.manager.add_example_query(query, category, "", description)
    
    def _apply_manufacturing_config(self):
        """应用制造业配置示例"""
        self.manager.update_business_context(
            industry_terms="制造业、生产线、产能利用率、良品率、设备效率、OEE、供应链、质量控制、成本控制、交期、工艺流程",
            business_rules="注重生产效率和质量控制，关注设备维护和产能优化，重视供应链稳定性，严格控制生产成本和交期",
            data_characteristics="包含生产订单、设备运行、质量检测、物料消耗、人员排班等数据，实时性要求高，数据量大且连续",
            analysis_focus="生产效率分析、质量分析、设备分析、成本分析、供应链分析、人员效率分析"
        )
        
        manufacturing_examples = [
            ("分析各生产线的产能利用率", "生产分析", "生产线效率对比和优化建议"),
            ("统计产品质量问题的主要原因", "质量分析", "质量问题根因分析"),
            ("查看设备故障频率和维护成本", "设备分析", "设备健康状况和维护策略"),
            ("分析原材料成本变化趋势", "成本分析", "原材料成本控制和采购优化"),
            ("统计各班次的生产效率差异", "人员分析", "班次效率对比和人员配置优化")
        ]
        
        self.manager.example_queries = []
        for query, category, description in manufacturing_examples:
            self.manager.add_example_query(query, category, "", description)
    
    def _apply_finance_config(self):
        """应用金融服务配置示例"""
        self.manager.update_business_context(
            industry_terms="金融、银行、保险、投资、风险管理、合规、KYC、反洗钱、信贷、资产管理、流动性、资本充足率、不良率",
            business_rules="严格遵守监管要求，注重风险控制和合规管理，关注客户资产安全，重视数据隐私保护",
            data_characteristics="包含客户信息、交易记录、风险评估、合规检查等敏感数据，对数据安全和隐私保护要求极高",
            analysis_focus="风险分析、合规分析、客户分析、产品分析、市场分析、运营效率分析"
        )
        
        finance_examples = [
            ("分析客户信贷风险等级分布", "风险分析", "信贷风险评估和管控"),
            ("统计各产品的盈利能力", "产品分析", "金融产品收益性分析"),
            ("查看合规检查中的异常交易", "合规分析", "反洗钱和异常交易监控"),
            ("分析客户资产配置偏好", "客户分析", "客户投资行为和偏好分析"),
            ("统计各渠道的获客成本和转化率", "营销分析", "获客渠道效果评估")
        ]
        
        self.manager.example_queries = []
        for query, category, description in finance_examples:
            self.manager.add_example_query(query, category, "", description)
    
    def _apply_education_config(self):
        """应用教育培训配置示例"""
        self.manager.update_business_context(
            industry_terms="教育、培训、学员、课程、师资、教学质量、学习效果、完课率、满意度、认证、考试、学分",
            business_rules="注重教学质量和学员体验，关注师资水平和课程设计，重视学习效果评估和持续改进",
            data_characteristics="包含学员信息、课程数据、学习记录、考试成绩、教师评价等教育相关数据",
            analysis_focus="学习效果分析、课程分析、师资分析、学员分析、教学质量分析、运营分析"
        )
        
        education_examples = [
            ("分析各课程的完课率和满意度", "课程分析", "课程质量评估和优化建议"),
            ("统计学员学习进度和成绩分布", "学员分析", "学员学习状况和个性化辅导"),
            ("查看教师授课效果和学员反馈", "师资分析", "教师教学质量评估"),
            ("分析不同学习方式的效果差异", "教学分析", "教学方法效果对比"),
            ("统计各专业的就业率和薪资水平", "就业分析", "专业就业前景和市场需求分析")
        ]
        
        self.manager.example_queries = []
        for query, category, description in education_examples:
            self.manager.add_example_query(query, category, "", description)
    
    def render_main_config_page(self):
        """渲染主配置页面"""
        st.title("🔧 Prompt模板配置")
        
        # 标签页
        tab1, tab2, tab3, tab4 = st.tabs([
            "📝 业务上下文", "📚 术语词典", "💡 示例查询", "👁️ Prompt预览"
        ])
        
        with tab1:
            self.render_business_context_config()
        
        with tab2:
            self.render_term_dictionary_config()
        
        with tab3:
            self.render_example_queries_config()
        
        with tab4:
            self.render_prompt_preview()

# 集成到主应用的函数
def integrate_prompt_config_to_main_app():
    """集成Prompt配置到主应用"""
    
    # 在侧边栏添加配置入口
    if 'prompt_config_ui' not in st.session_state:
        st.session_state.prompt_config_ui = PromptConfigUI()
    
    # 渲染配置侧边栏
    st.session_state.prompt_config_ui.render_config_sidebar()
    
    # 处理配置页面显示
    if st.session_state.get('show_business_config', False):
        st.session_state.prompt_config_ui.render_business_context_config()
        if st.button("✅ 完成配置"):
            st.session_state.show_business_config = False
            st.rerun()
    
    elif st.session_state.get('show_term_config', False):
        st.session_state.prompt_config_ui.render_term_dictionary_config()
        if st.button("✅ 完成配置"):
            st.session_state.show_term_config = False
            st.rerun()
    
    elif st.session_state.get('show_example_config', False):
        st.session_state.prompt_config_ui.render_example_queries_config()
        if st.button("✅ 完成配置"):
            st.session_state.show_example_config = False
            st.rerun()

def get_configured_prompt_manager() -> PromptTemplateManager:
    """获取配置好的Prompt管理器"""
    if 'prompt_manager' not in st.session_state:
        st.session_state.prompt_manager = PromptTemplateManager()
    return st.session_state.prompt_manager

# 使用示例
if __name__ == "__main__":
    st.set_page_config(
        page_title="Prompt配置管理",
        page_icon="🔧",
        layout="wide"
    )
    
    ui = PromptConfigUI()
    ui.render_main_config_page()
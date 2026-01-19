"""
Intel® DeepInsight - Prompt模板系统集成模块
将新的Prompt模板系统集成到现有的agent_core.py中
"""

from typing import Optional, Dict, Any
import logging
from prompt_template_system import PromptTemplateManager, PromptMode, LLMProvider

logger = logging.getLogger(__name__)

class EnhancedPromptBuilder:
    """增强的Prompt构建器，集成到现有系统"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化Prompt构建器"""
        self.config = config
        self.template_manager = PromptTemplateManager()
        
        # 从配置中获取LLM提供商
        self.llm_provider = self._detect_llm_provider(config)
        
        # 默认使用智能查询模式
        self.default_mode = PromptMode.FLEXIBLE
    
    @property
    def manager(self) -> PromptTemplateManager:
        """兼容性属性：返回template_manager，供agent_core.py使用"""
        return self.template_manager
    
    def _detect_llm_provider(self, config: Dict[str, Any]) -> LLMProvider:
        """根据配置自动检测LLM提供商"""
        api_base = config.get('api_base', '').lower()
        model_name = config.get('model_name', '').lower()
        
        if 'deepseek' in api_base or 'deepseek' in model_name:
            return LLMProvider.DEEPSEEK
        elif 'openai' in api_base or 'gpt' in model_name:
            return LLMProvider.OPENAI
        elif 'claude' in api_base or 'claude' in model_name:
            return LLMProvider.CLAUDE
        elif 'qwen' in api_base or 'qwen' in model_name:
            return LLMProvider.QWEN
        else:
            return LLMProvider.CUSTOM
    
    def build_sql_generation_prompt(self,
                                  user_query: str,
                                  schema_info: str,
                                  rag_context: str = "",
                                  selected_tables: Optional[list] = None,
                                  query_possibilities: Optional[list] = None,
                                  retry_context: Optional[Dict] = None,
                                  mode: Optional[PromptMode] = None) -> str:
        """
        构建SQL生成的Prompt
        
        Args:
            user_query: 用户查询
            schema_info: 数据库Schema信息
            rag_context: RAG检索上下文
            selected_tables: 选中的表信息
            query_possibilities: 查询可能性列表
            retry_context: 重试上下文（包含错误信息）
            mode: Prompt模式
        
        Returns:
            完整的Prompt字符串
        """
        
        # 使用指定模式或默认模式
        prompt_mode = mode or self.default_mode
        
        # 构建增强的Schema信息
        enhanced_schema = self._build_enhanced_schema_info(schema_info, selected_tables)
        
        # 构建增强的RAG上下文
        enhanced_rag_context = self._build_enhanced_rag_context(rag_context, query_possibilities)
        
        # 构建基础Prompt
        base_prompt = self.template_manager.build_complete_prompt(
            user_query=user_query,
            schema_info=enhanced_schema,
            rag_context=enhanced_rag_context,
            mode=prompt_mode,
            llm_provider=self.llm_provider
        )
        
        # 如果有重试上下文，添加错误信息
        if retry_context:
            base_prompt = self._add_retry_context(base_prompt, retry_context)
        
        return base_prompt
    
    def _build_enhanced_schema_info(self, schema_info: str, selected_tables: Optional[list]) -> str:
        """构建增强的Schema信息"""
        enhanced_info = schema_info
        
        if selected_tables:
            enhanced_info += "\n\n【重点关注表】\n"
            for table_info in selected_tables:
                if hasattr(table_info, 'table_name'):
                    enhanced_info += f"- {table_info.table_name}: {getattr(table_info, 'reasoning', '相关表')}\n"
                else:
                    enhanced_info += f"- {table_info}\n"
        
        return enhanced_info
    
    def _build_enhanced_rag_context(self, rag_context: str, query_possibilities: Optional[list]) -> str:
        """构建增强的RAG上下文"""
        enhanced_context = rag_context
        
        if query_possibilities:
            enhanced_context += "\n\n【查询理解可能性】\n"
            for i, possibility in enumerate(query_possibilities[:3], 1):  # 只取前3个
                if hasattr(possibility, 'natural_description'):
                    enhanced_context += f"{i}. {possibility.natural_description}\n"
                elif hasattr(possibility, 'description'):
                    enhanced_context += f"{i}. {possibility.description}\n"
                else:
                    enhanced_context += f"{i}. {possibility}\n"
        
        return enhanced_context
    
    def _add_retry_context(self, base_prompt: str, retry_context: Dict) -> str:
        """添加重试上下文信息"""
        retry_info = "\n\n【错误修复指导】\n"
        
        # 添加错误历史
        if 'errors' in retry_context:
            retry_info += "之前的错误：\n"
            for error in retry_context['errors']:
                retry_info += f"- {error.get('category', 'UNKNOWN')}: {error.get('message', '')}\n"
        
        # 添加修复建议
        if 'suggestions' in retry_context:
            retry_info += "\n修复建议：\n"
            for suggestion in retry_context['suggestions']:
                retry_info += f"- {suggestion}\n"
        
        # 添加重试次数信息
        retry_count = retry_context.get('retry_count', 0)
        if retry_count > 0:
            retry_info += f"\n这是第 {retry_count + 1} 次尝试，请特别注意避免之前的错误。\n"
        
        return base_prompt + retry_info
    
    def build_insight_generation_prompt(self,
                                      user_query: str,
                                      sql_query: str,
                                      query_results: Any,
                                      anomalies: Optional[list] = None,
                                      visualizations: Optional[Dict] = None) -> str:
        """
        构建洞察生成的Prompt
        
        Args:
            user_query: 原始用户查询
            sql_query: 执行的SQL查询
            query_results: 查询结果
            anomalies: 检测到的异常
            visualizations: 可视化信息
        
        Returns:
            洞察生成的Prompt
        """
        
        # 获取业务上下文
        business_context = self.template_manager.get_business_context_section(user_query)
        
        # 构建洞察生成Prompt
        insight_prompt = f"""
你是Intel® DeepInsight的商业洞察分析师，请基于查询结果提供深度的商业洞察。

{business_context}

【原始查询】
{user_query}

【执行的SQL】
```sql
{sql_query}
```

【查询结果概要】
- 数据行数: {len(query_results) if hasattr(query_results, '__len__') else '未知'}
- 主要字段: {', '.join(query_results.columns.tolist()) if hasattr(query_results, 'columns') else '未知'}
"""
        
        # 添加异常信息
        if anomalies:
            insight_prompt += f"\n【检测到的异常】\n"
            for anomaly in anomalies[:5]:  # 只显示前5个异常
                insight_prompt += f"- {anomaly.get('type', '未知异常')}: {anomaly.get('description', '')}\n"
        
        # 添加可视化信息
        if visualizations:
            insight_prompt += f"\n【可视化类型】\n{visualizations.get('chart_type', '未知')}\n"
        
        insight_prompt += """
【请提供以下洞察】
1. **数据概览**: 简述查询结果的主要发现
2. **关键指标**: 识别重要的业务指标和趋势
3. **异常分析**: 解释检测到的异常及其可能原因
4. **商业建议**: 基于数据提供可操作的业务建议
5. **风险提示**: 指出需要关注的潜在风险

请用简洁明了的语言，重点关注业务价值和可操作性。
"""
        
        return insight_prompt
    
    def set_prompt_mode(self, mode: PromptMode):
        """设置Prompt模式"""
        self.default_mode = mode
    
    def update_business_context(self, **kwargs):
        """更新业务上下文"""
        self.template_manager.update_business_context(**kwargs)
    
    def add_example_query(self, query: str, category: str, description: str = ""):
        """添加示例查询"""
        self.template_manager.add_example_query(query, category, description=description)
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        summary = self.template_manager.get_config_summary()
        summary['llm_provider'] = self.llm_provider.value
        summary['default_mode'] = self.default_mode.value
        return summary
    
    # 术语词典操作方法 - 代理到template_manager
    def add_term(self, term: str, explanation: str):
        """添加术语到词典"""
        self.template_manager.add_term(term, explanation)
    
    def update_term(self, term: str, new_explanation: str):
        """更新术语解释"""
        self.template_manager.update_term(term, new_explanation)
    
    def delete_term(self, term: str):
        """删除术语"""
        return self.template_manager.delete_term(term)
    
    def search_terms(self, keyword: str):
        """搜索术语"""
        return self.template_manager.search_terms(keyword)
    
    def load_term_dictionary(self, csv_path: str):
        """加载术语词典"""
        self.template_manager.load_term_dictionary(csv_path)

# 用于替换现有agent_core.py中的prompt构建逻辑
def create_enhanced_prompt_builder(config: Dict[str, Any]) -> EnhancedPromptBuilder:
    """创建增强的Prompt构建器"""
    return EnhancedPromptBuilder(config)

# 兼容性函数，用于逐步迁移现有代码
def build_legacy_prompt(user_query: str, 
                       schema_info: str, 
                       rag_context: str = "",
                       config: Optional[Dict[str, Any]] = None) -> str:
    """
    兼容性函数，用于逐步迁移现有的prompt构建逻辑
    """
    if config is None:
        config = {}
    
    builder = EnhancedPromptBuilder(config)
    return builder.build_sql_generation_prompt(
        user_query=user_query,
        schema_info=schema_info,
        rag_context=rag_context
    )

# 使用示例
if __name__ == "__main__":
    # 示例配置
    config = {
        'api_base': 'http://aidemo.intel.cn/v1',
        'model_name': 'DeepSeek-V3.1'
    }
    
    # 创建Prompt构建器
    builder = EnhancedPromptBuilder(config)
    
    # 配置业务上下文
    builder.update_business_context(
        industry_terms="零售业、电商、供应链",
        business_rules="关注季节性销售趋势",
        analysis_focus="销售分析、客户分析"
    )
    
    # 构建SQL生成Prompt
    prompt = builder.build_sql_generation_prompt(
        user_query="分析最近三个月的销售趋势",
        schema_info="orders表包含订单信息...",
        rag_context="销售趋势分析..."
    )
    
    print("生成的Prompt:")
    print(prompt)
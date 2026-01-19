"""
Intel® DeepInsight - 多LLM支持的Prompt模板管理系统
支持专业模式和灵活模式，可自定义业务上下文和术语词典
"""

import json
import csv
import time
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import re
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PromptMode(Enum):
    """查询策略枚举"""
    PROFESSIONAL = "professional"  # 标准查询：严格按照数据库结构匹配
    FLEXIBLE = "flexible"         # 智能查询：理解业务语义，提供灵活方案

class LLMProvider(Enum):
    """支持的LLM提供商"""
    DEEPSEEK = "deepseek"
    OPENAI = "openai"
    CLAUDE = "claude"
    QWEN = "qwen"
    CUSTOM = "custom"

@dataclass
class BusinessContext:
    """业务上下文配置"""
    industry_terms: str = ""           # 行业特定术语，500字符限制
    business_rules: str = ""           # 业务规则说明
    data_characteristics: str = ""     # 数据特征描述
    analysis_focus: str = ""           # 分析重点
    
    def validate(self) -> Tuple[bool, str]:
        """验证业务上下文配置"""
        total_length = len(self.industry_terms + self.business_rules + 
                          self.data_characteristics + self.analysis_focus)
        if total_length > 2000:  # 总长度限制
            return False, f"业务上下文总长度超限: {total_length}/2000"
        return True, "验证通过"

@dataclass
class TermDictionary:
    """术语词典"""
    terms: Dict[str, str] = field(default_factory=dict)  # 术语 -> 解释
    
    @classmethod
    def from_csv(cls, csv_path: str) -> 'TermDictionary':
        """从CSV文件加载术语词典"""
        terms = {}
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    term = row.get('term', '').strip()
                    explanation = row.get('explanation', '').strip()
                    if term and explanation:
                        terms[term] = explanation
        except Exception as e:
            logger.error(f"加载术语词典失败: {e}")
        return cls(terms=terms)
    
    def get_relevant_terms(self, query: str) -> Dict[str, str]:
        """获取查询相关的术语"""
        relevant = {}
        query_lower = query.lower()
        for term, explanation in self.terms.items():
            if term.lower() in query_lower:
                relevant[term] = explanation
        return relevant
    
    def add_term(self, term: str, explanation: str):
        """添加术语"""
        if not term or not explanation:
            raise ValueError("术语和解释不能为空")
        self.terms[term] = explanation
    
    def update_term(self, term: str, new_explanation: str):
        """修改术语解释"""
        if term not in self.terms:
            raise ValueError(f"术语 '{term}' 不存在")
        self.terms[term] = new_explanation
    
    def delete_term(self, term: str):
        """删除术语"""
        if term not in self.terms:
            raise ValueError(f"术语 '{term}' 不存在")
        return self.terms.pop(term)
    
    def search_terms(self, keyword: str) -> Dict[str, str]:
        """搜索术语"""
        keyword_lower = keyword.lower()
        results = {}
        for term, explanation in self.terms.items():
            if (keyword_lower in term.lower() or 
                keyword_lower in explanation.lower()):
                results[term] = explanation
        return results
    
    def save_to_csv(self, csv_path: str):
        """保存术语到CSV文件"""
        import os
        import csv
        
        # 确保目录存在
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['term', 'explanation'])
            for term, explanation in self.terms.items():
                writer.writerow([term, explanation])

@dataclass
class ExampleQuery:
    """示例查询"""
    query: str
    category: str
    sql_pattern: str = ""
    description: str = ""

class PromptTemplateManager:
    """Prompt模板管理器"""
    
    def __init__(self, config_path: str = "data/prompt_config.json"):
        self.config_path = config_path
        self.business_context = BusinessContext()
        self.term_dictionary = TermDictionary()
        self.example_queries: List[ExampleQuery] = []
        self.load_config()
    
    def load_config(self):
        """加载配置"""
        # 首先加载默认配置
        self._load_default_config()
        
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                # 加载业务上下文
                if 'business_context' in config:
                    bc = config['business_context']
                    self.business_context = BusinessContext(
                        industry_terms=bc.get('industry_terms', ''),
                        business_rules=bc.get('business_rules', ''),
                        data_characteristics=bc.get('data_characteristics', ''),
                        analysis_focus=bc.get('analysis_focus', '')
                    )
                
                # 加载术语词典 - 改进加载逻辑
                if 'term_dictionary_path' in config and config['term_dictionary_path']:
                    term_dict_path = config['term_dictionary_path']
                    if os.path.exists(term_dict_path):
                        try:
                            self.term_dictionary = TermDictionary.from_csv(term_dict_path)
                            self._term_dict_path = term_dict_path
                            logger.info(f"术语词典加载成功: {len(self.term_dictionary.terms)}个术语")
                        except Exception as e:
                            logger.warning(f"术语词典加载失败: {e}")
                            # 加载失败时使用默认词典
                            self._load_default_term_dictionary()
                    else:
                        logger.warning(f"术语词典文件不存在: {term_dict_path}")
                        # 文件不存在时使用默认词典
                        self._load_default_term_dictionary()
                else:
                    # 没有配置术语词典路径时使用默认词典
                    self._load_default_term_dictionary()
                
                # 加载示例查询
                if 'example_queries' in config:
                    self.example_queries = [
                        ExampleQuery(**eq) for eq in config['example_queries']
                    ]
                    
                logger.info(f"配置加载成功: {len(self.example_queries)}个示例查询, {len(self.term_dictionary.terms)}个术语")
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            # 出错时使用默认配置
            self._load_default_config()
    
    def _load_default_config(self):
        """加载默认配置"""
        # 默认业务上下文
        self.business_context = BusinessContext(
            industry_terms="零售业、供应链、库存周转率、客单价、同比、环比",
            business_rules="关注季节性销售趋势，注重产品类别间的关联销售，考虑地域差异对销售的影响",
            data_characteristics="[待补充]",
            analysis_focus="销售分析、客户分析、产品分析、运营效率、地域分析、时间趋势分析"
        )
        
        # 默认术语词典（从模板文件加载）
        self._load_default_term_dictionary()
        
        # 默认示例查询
        self.example_queries = [
            ExampleQuery(
                query="查看去年销售额最高的产品",
                category="销售分析",
                sql_pattern="SELECT ProductName, SUM(UnitPrice * Quantity) as TotalSales FROM orderdetails od JOIN products p ON od.ProductID = p.ProductID JOIN orders o ON od.OrderID = o.OrderID WHERE YEAR(o.OrderDate) = YEAR(CURDATE()) - 1 GROUP BY p.ProductID ORDER BY TotalSales DESC",
                description="产品销售排名分析，帮助了解热销产品和市场偏好"
            ),
            ExampleQuery(
                query="分析客户的购买频率和消费金额",
                category="客户分析",
                sql_pattern="SELECT c.CustomerID, c.CompanyName, COUNT(o.OrderID) as OrderCount, AVG(od.UnitPrice * od.Quantity) as AvgOrderValue FROM customers c JOIN orders o ON c.CustomerID = o.CustomerID JOIN orderdetails od ON o.OrderID = od.OrderID GROUP BY c.CustomerID",
                description="客户价值分析，识别高价值客户和购买模式"
            )
        ]
    
    def _load_default_term_dictionary(self):
        """加载默认术语词典"""
        template_path = "data/term_dictionary_template.csv"
        if os.path.exists(template_path):
            try:
                self.term_dictionary = TermDictionary.from_csv(template_path)
                self._term_dict_path = template_path
                logger.info(f"默认术语词典加载成功: {len(self.term_dictionary.terms)}个术语")
            except Exception as e:
                logger.warning(f"加载默认术语词典失败: {e}")
                self.term_dictionary = TermDictionary()
                self._term_dict_path = ""
        else:
            self.term_dictionary = TermDictionary()
            self._term_dict_path = ""
    
    def save_config(self):
        """保存配置"""
        config = {
            'business_context': {
                'industry_terms': self.business_context.industry_terms,
                'business_rules': self.business_context.business_rules,
                'data_characteristics': self.business_context.data_characteristics,
                'analysis_focus': self.business_context.analysis_focus
            },
            'example_queries': [
                {
                    'query': eq.query,
                    'category': eq.category,
                    'sql_pattern': eq.sql_pattern,
                    'description': eq.description
                } for eq in self.example_queries
            ],
            'term_dictionary_path': getattr(self, '_term_dict_path', ''),
            'last_updated': time.time()
        }
        
        try:
            Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
            # 先写入临时文件，再重命名，确保原子性操作
            temp_path = self.config_path + '.tmp'
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            # 原子性重命名
            import os
            if os.path.exists(self.config_path):
                os.replace(temp_path, self.config_path)
            else:
                os.rename(temp_path, self.config_path)
                
            logger.info(f"配置已保存到: {self.config_path}")
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            # 清理临时文件
            temp_path = self.config_path + '.tmp'
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
    
    def get_system_instruction(self, mode: PromptMode, llm_provider: LLMProvider) -> str:
        """获取系统指令（固定部分）"""
        base_instruction = """你是Intel® DeepInsight智能数据分析助手，专门帮助用户通过自然语言查询分析业务数据。

核心能力：
1. 理解自然语言查询意图
2. 生成准确的SQL查询语句
3. 提供数据洞察和业务建议
4. 识别数据异常和风险点

安全规则：
- 只能生成SELECT、WITH、EXPLAIN类型的查询语句
- 严禁生成任何修改数据的语句（INSERT、UPDATE、DELETE、DROP等）
- 对不确定的查询要求用户澄清

"""
        
        if mode == PromptMode.PROFESSIONAL:
            mode_instruction = """
查询策略：标准查询
- 严格按照数据库Schema结构生成SQL
- 确保查询语法的准确性和性能优化
- 提供精确的数据匹配和标准化输出
- 适合明确的业务查询需求
"""
        else:  # FLEXIBLE
            mode_instruction = """
查询策略：智能查询  
- 理解业务语义，提供灵活的查询方案
- 支持模糊匹配和近似查询
- 智能推断用户的真实查询意图
- 适合探索性和复杂业务场景
"""
        
        # LLM特定的指令调整
        llm_specific = self._get_llm_specific_instruction(llm_provider)
        
        return base_instruction + mode_instruction + llm_specific
    
    def _get_llm_specific_instruction(self, llm_provider: LLMProvider) -> str:
        """获取LLM特定的指令"""
        instructions = {
            LLMProvider.DEEPSEEK: """
DeepSeek优化指令：
- 充分利用你的推理能力进行复杂查询分析
- 使用CoT（思维链）方式解释查询逻辑
- 对歧义查询提供多种理解可能性
""",
            LLMProvider.OPENAI: """
OpenAI优化指令：
- 利用你的代码理解能力生成高质量SQL
- 提供清晰的步骤分解
- 注重用户体验和解释的易懂性
""",
            LLMProvider.CLAUDE: """
Claude优化指令：
- 发挥你的分析能力提供深度洞察
- 注重安全性和准确性
- 提供结构化的回答格式
""",
            LLMProvider.QWEN: """
Qwen优化指令：
- 充分理解中文语境和业务场景
- 提供本土化的业务分析视角
- 注重实用性和可操作性
""",
            LLMProvider.CUSTOM: ""
        }
        return instructions.get(llm_provider, "")
    
    def get_business_context_section(self, query: str) -> str:
        """获取业务上下文部分（可定制）"""
        if not any([self.business_context.industry_terms,
                   self.business_context.business_rules,
                   self.business_context.data_characteristics,
                   self.business_context.analysis_focus]):
            return ""
        
        context_parts = []
        
        if self.business_context.industry_terms:
            context_parts.append(f"行业术语：{self.business_context.industry_terms}")
        
        if self.business_context.business_rules:
            context_parts.append(f"业务规则：{self.business_context.business_rules}")
        
        if self.business_context.data_characteristics:
            context_parts.append(f"数据特征：{self.business_context.data_characteristics}")
        
        if self.business_context.analysis_focus:
            context_parts.append(f"分析重点：{self.business_context.analysis_focus}")
        
        # 添加相关术语
        relevant_terms = self.term_dictionary.get_relevant_terms(query)
        if relevant_terms:
            terms_text = "、".join([f"{term}({explanation})" 
                                  for term, explanation in relevant_terms.items()])
            context_parts.append(f"相关术语：{terms_text}")
        
        if context_parts:
            return "\n\n【业务上下文】\n" + "\n".join(context_parts)
        return ""
    
    def get_query_processing_logic(self, mode: PromptMode) -> str:
        """获取查询处理逻辑（固定）"""
        base_logic = """
【查询处理流程】
1. 分析用户查询意图和关键信息
2. 识别涉及的数据表和字段
3. 考虑可能的歧义和多种理解方式
4. 生成SQL查询语句
5. 提供查询说明和预期结果描述
"""
        
        if mode == PromptMode.PROFESSIONAL:
            specific_logic = """
【标准查询要求】
- SQL语句必须语法正确且性能优化
- 严格遵循数据库Schema结构
- 提供精确的字段匹配和JOIN逻辑
- 确保查询结果的准确性和一致性
"""
        else:  # FLEXIBLE
            specific_logic = """
【智能查询要求】
- 理解业务需求的核心意图
- 对模糊表达提供最合理的查询解释
- 如有歧义，提供多种可能的查询方案
- 用业务语言解释查询逻辑和结果
"""
        
        return base_logic + specific_logic
    
    def get_example_queries_section(self, query: str, limit: int = 3) -> str:
        """获取相关示例查询"""
        if not self.example_queries:
            return ""
        
        # 简单的相关性匹配（可以后续优化为向量相似度）
        query_lower = query.lower()
        relevant_examples = []
        
        for example in self.example_queries:
            # 计算相关性得分
            score = 0
            example_words = set(example.query.lower().split())
            query_words = set(query_lower.split())
            
            # 词汇重叠度
            overlap = len(example_words & query_words)
            if overlap > 0:
                score = overlap / len(example_words | query_words)
                relevant_examples.append((score, example))
        
        # 按相关性排序并取前N个
        relevant_examples.sort(key=lambda x: x[0], reverse=True)
        top_examples = relevant_examples[:limit]
        
        if top_examples:
            examples_text = "\n【相关查询示例】\n"
            for i, (score, example) in enumerate(top_examples, 1):
                examples_text += f"{i}. {example.query}"
                if example.description:
                    examples_text += f" - {example.description}"
                examples_text += "\n"
            return examples_text
        
        return ""
    
    def build_complete_prompt(self, 
                            user_query: str,
                            schema_info: str,
                            rag_context: str,
                            mode: PromptMode = PromptMode.FLEXIBLE,
                            llm_provider: LLMProvider = LLMProvider.DEEPSEEK) -> str:
        """构建完整的Prompt"""
        
        # 1. 系统指令（固定）
        system_instruction = self.get_system_instruction(mode, llm_provider)
        
        # 2. 业务上下文（可定制）
        business_context = self.get_business_context_section(user_query)
        
        # 3. 查询处理逻辑（固定）
        processing_logic = self.get_query_processing_logic(mode)
        
        # 4. 示例查询（动态）
        example_queries = self.get_example_queries_section(user_query)
        
        # 5. 数据库Schema信息
        schema_section = f"\n【数据库Schema】\n{schema_info}"
        
        # 6. RAG检索上下文
        rag_section = f"\n【相关知识】\n{rag_context}" if rag_context else ""
        
        # 7. 用户查询
        user_section = f"\n【用户查询】\n{user_query}"
        
        # 8. 输出格式要求
        output_format = """
【输出格式】
请按以下格式回答：

1. **查询理解**：简述对用户查询的理解
2. **涉及表格**：列出需要查询的数据表
3. **SQL查询**：
```sql
-- 生成的SQL查询语句
```
4. **查询说明**：解释SQL逻辑和预期结果
5. **注意事项**：提醒用户可能需要注意的问题
"""
        
        # 组装完整Prompt
        complete_prompt = (
            system_instruction +
            business_context +
            processing_logic +
            example_queries +
            schema_section +
            rag_section +
            user_section +
            output_format
        )
        
        return complete_prompt
    
    def update_business_context(self, **kwargs):
        """更新业务上下文"""
        for key, value in kwargs.items():
            if hasattr(self.business_context, key):
                setattr(self.business_context, key, value)
        
        # 验证更新后的配置
        is_valid, message = self.business_context.validate()
        if not is_valid:
            raise ValueError(f"业务上下文配置无效: {message}")
        
        # 立即保存配置
        self.save_config()
        logger.info("业务上下文已更新并保存")
    
    def add_example_query(self, query: str, category: str, 
                         sql_pattern: str = "", description: str = ""):
        """添加示例查询"""
        example = ExampleQuery(
            query=query,
            category=category,
            sql_pattern=sql_pattern,
            description=description
        )
        self.example_queries.append(example)
        # 立即保存配置
        self.save_config()
        logger.info(f"示例查询已添加: {query}")
    
    def remove_example_query(self, index: int):
        """删除示例查询"""
        if 0 <= index < len(self.example_queries):
            removed = self.example_queries.pop(index)
            self.save_config()
            logger.info(f"示例查询已删除: {removed.query}")
            return True
        return False
    
    def load_term_dictionary(self, csv_path: str):
        """加载术语词典"""
        try:
            # 先备份当前词典
            old_dictionary = self.term_dictionary
            old_path = getattr(self, '_term_dict_path', '')
            
            # 加载新词典
            new_dictionary = TermDictionary.from_csv(csv_path)
            
            # 验证加载成功
            if len(new_dictionary.terms) > 0:
                self.term_dictionary = new_dictionary
                self._term_dict_path = csv_path
                
                # 立即保存配置，确保路径和数据都被保存
                self.save_config()
                
                logger.info(f"术语词典已加载并保存: {len(self.term_dictionary.terms)}个术语，路径: {csv_path}")
            else:
                logger.warning("加载的术语词典为空，保持原有词典")
                
        except Exception as e:
            logger.error(f"加载术语词典失败: {e}")
            # 恢复原有词典
            if 'old_dictionary' in locals():
                self.term_dictionary = old_dictionary
                if 'old_path' in locals():
                    self._term_dict_path = old_path
            raise
    
    def add_term(self, term: str, explanation: str):
        """添加术语"""
        self.term_dictionary.add_term(term, explanation)
        # 确保CSV路径存在
        if not hasattr(self, '_term_dict_path') or not self._term_dict_path:
            # 使用默认路径
            self._term_dict_path = "data/uploaded_terms_user_uploaded_terms.csv"
            os.makedirs("data", exist_ok=True)
        
        # 保存到CSV文件
        self.term_dictionary.save_to_csv(self._term_dict_path)
        # 保存配置
        self.save_config()
        logger.info(f"术语已添加: {term}")
    
    def update_term(self, term: str, new_explanation: str):
        """修改术语"""
        self.term_dictionary.update_term(term, new_explanation)
        # 确保CSV路径存在
        if not hasattr(self, '_term_dict_path') or not self._term_dict_path:
            # 使用默认路径
            self._term_dict_path = "data/uploaded_terms_user_uploaded_terms.csv"
            os.makedirs("data", exist_ok=True)
        
        # 保存到CSV文件
        self.term_dictionary.save_to_csv(self._term_dict_path)
        # 保存配置
        self.save_config()
        logger.info(f"术语已修改: {term}")
    
    def delete_term(self, term: str):
        """删除术语"""
        deleted_explanation = self.term_dictionary.delete_term(term)
        # 确保CSV路径存在
        if not hasattr(self, '_term_dict_path') or not self._term_dict_path:
            # 使用默认路径
            self._term_dict_path = "data/uploaded_terms_user_uploaded_terms.csv"
            os.makedirs("data", exist_ok=True)
        
        # 保存到CSV文件
        self.term_dictionary.save_to_csv(self._term_dict_path)
        # 保存配置
        self.save_config()
        logger.info(f"术语已删除: {term}")
        return deleted_explanation
    
    def search_terms(self, keyword: str):
        """搜索术语"""
        return self.term_dictionary.search_terms(keyword)
    
    def reset_to_default(self):
        """重置为默认配置"""
        self._load_default_config()
        self.save_config()
        logger.info("配置已重置为默认值")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        # 重新计算统计数据
        business_context_length = len(
            self.business_context.industry_terms +
            self.business_context.business_rules +
            self.business_context.data_characteristics +
            self.business_context.analysis_focus
        )
        
        business_context_configured = bool(
            self.business_context.industry_terms.strip() or
            self.business_context.business_rules.strip() or
            self.business_context.data_characteristics.strip() or
            self.business_context.analysis_focus.strip()
        )
        
        return {
            'business_context_configured': business_context_configured,
            'term_dictionary_size': len(self.term_dictionary.terms),
            'example_queries_count': len(self.example_queries),
            'business_context_length': business_context_length,
            'config_file_exists': Path(self.config_path).exists(),
            'last_updated': getattr(self, '_last_updated', None)
        }

# 使用示例和测试
if __name__ == "__main__":
    # 创建模板管理器
    manager = PromptTemplateManager()
    
    # 配置业务上下文
    manager.update_business_context(
        industry_terms="零售业、电商、供应链、库存周转率、客单价",
        business_rules="关注季节性销售趋势，重视客户留存率分析",
        data_characteristics="包含订单、产品、客户、员工等核心业务数据",
        analysis_focus="销售分析、客户分析、产品分析、运营效率"
    )
    
    # 添加示例查询
    manager.add_example_query(
        query="查看去年销售额最高的产品",
        category="销售分析",
        description="产品销售排名分析"
    )
    
    # 构建完整Prompt
    prompt = manager.build_complete_prompt(
        user_query="分析一下最近三个月的销售趋势",
        schema_info="orders表包含订单信息，products表包含产品信息...",
        rag_context="销售趋势分析通常关注时间序列变化...",
        mode=PromptMode.PROFESSIONAL,
        llm_provider=LLMProvider.DEEPSEEK
    )
    
    print("生成的完整Prompt:")
    print("=" * 50)
    print(prompt)

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

# 中文分词支持
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    print("⚠️ jieba未安装，示例查询匹配将使用字符级匹配")

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
    
    # 默认示例查询相似度阈值（15%，过滤掉低相似度的噪声匹配）
    DEFAULT_EXAMPLE_THRESHOLD = 0.15
    
    def __init__(self, config_path: str = "data/prompt_config.json"):
        self.config_path = config_path
        self.business_context = BusinessContext()
        self.term_dictionary = TermDictionary()
        self.example_queries: List[ExampleQuery] = []
        self.example_query_threshold = self.DEFAULT_EXAMPLE_THRESHOLD
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
                terms_loaded = False
                
                # 优先从CSV文件加载
                if 'term_dictionary_path' in config and config['term_dictionary_path']:
                    term_dict_path = config['term_dictionary_path']
                    if os.path.exists(term_dict_path):
                        try:
                            self.term_dictionary = TermDictionary.from_csv(term_dict_path)
                            self._term_dict_path = term_dict_path
                            logger.info(f"术语词典从CSV加载成功: {len(self.term_dictionary.terms)}个术语")
                            terms_loaded = True
                        except Exception as e:
                            logger.warning(f"从CSV加载术语词典失败: {e}")
                
                # 如果CSV加载失败，尝试从JSON配置中加载术语
                if not terms_loaded and 'term_dictionary' in config and config['term_dictionary']:
                    try:
                        self.term_dictionary = TermDictionary(terms=config['term_dictionary'])
                        self._term_dict_path = config.get('term_dictionary_path', '')
                        logger.info(f"术语词典从JSON配置加载成功: {len(self.term_dictionary.terms)}个术语")
                        terms_loaded = True
                    except Exception as e:
                        logger.warning(f"从JSON配置加载术语词典失败: {e}")
                
                # 如果都失败了，保持默认配置中的术语词典（已在_load_default_config中设置）
                if not terms_loaded:
                    logger.info("使用默认术语词典")
                
                # 加载示例查询
                if 'example_queries' in config:
                    self.example_queries = [
                        ExampleQuery(**eq) for eq in config['example_queries']
                    ]
                
                # 加载示例查询相似度阈值
                if 'example_query_threshold' in config:
                    self.example_query_threshold = config['example_query_threshold']
                    
                logger.info(f"配置加载成功: {len(self.example_queries)}个示例查询, {len(self.term_dictionary.terms)}个术语, 阈值{self.example_query_threshold:.0%}")
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            # 出错时使用默认配置
            self._load_default_config()
    
    def _load_default_config(self):
        """加载默认配置 - 基于Northwind数据库"""
        # 默认业务上下文 - Northwind食品饮料贸易公司
        self.business_context = BusinessContext(
            industry_terms="食品饮料贸易、B2B批发、库存管理、供应链、运费成本、订单履约、产品类别、供应商管理、销售区域、员工业绩",
            business_rules="关注产品类别销售趋势，重视供应商稳定性和交货及时性，分析不同国家和地区的客户需求差异，监控运费成本优化物流，注重订单履约率和客户满意度",
            data_characteristics="包含产品(products)、类别(categories)、供应商(suppliers)、客户(customers)、员工(employees)、订单(orders)、订单明细(orderdetails)、承运商(shippers)等核心业务数据，支持订单到发货的完整业务链分析",
            analysis_focus="产品销售分析、客户价值分析、供应商绩效分析、员工业绩分析、区域市场分析、运费成本分析、订单时效分析"
        )
        
        # 默认术语词典 - Northwind业务术语
        self.term_dictionary = TermDictionary(terms={
            # 核心业务概念
            "订单金额": "orderdetails表中UnitPrice * Quantity * (1 - Discount)的计算结果",
            "运费": "orders表中的Freight字段，表示每笔订单的物流运输费用",
            "产品类别": "categories表，包含Beverages(饮料)、Condiments(调味品)、Confections(糖果)、Dairy Products(乳制品)、Grains/Cereals(谷物/麦片)、Meat/Poultry(肉类/家禽)、Produce(农产品)、Seafood(海鲜)等8大类",
            "再订购点": "products表中的ReorderLevel字段，库存低于此值时需要补货",
            "已停产": "products表中Discontinued=1表示产品已停产",
            "库存数量": "products表中的UnitsInStock字段",
            "在途订购": "products表中的UnitsOnOrder字段，已订购但未到货的数量",
            # 客户相关
            "客户公司": "customers表中的CompanyName字段",
            "联系人": "customers表中的ContactName和ContactTitle字段",
            # 员工相关
            "上级主管": "employees表中ReportsTo字段引用的员工",
            "销售区域": "通过employeeterritories和territories表关联的区域信息",
            # 订单相关
            "要求发货日期": "orders表中的RequiredDate字段，客户期望的发货日期",
            "实际发货日期": "orders表中的ShippedDate字段，订单实际发出日期",
            "承运商": "shippers表，包含Speedy Express、United Package、Federal Shipping三家物流公司"
        })
        self._term_dict_path = ""
        
        # 默认示例查询 - Northwind业务场景
        self.example_queries = [
            ExampleQuery(
                query="查看销售额最高的产品",
                category="产品分析",
                sql_pattern="SELECT p.ProductName, SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) as TotalSales FROM orderdetails od JOIN products p ON od.ProductID = p.ProductID GROUP BY p.ProductID, p.ProductName ORDER BY TotalSales DESC LIMIT 10",
                description="分析产品销售排名，识别畅销产品"
            ),
            ExampleQuery(
                query="各产品类别的销售情况",
                category="类别分析",
                sql_pattern="SELECT c.CategoryName, COUNT(DISTINCT od.ProductID) as ProductCount, SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) as TotalSales FROM categories c JOIN products p ON c.CategoryID = p.CategoryID JOIN orderdetails od ON p.ProductID = od.ProductID GROUP BY c.CategoryID, c.CategoryName ORDER BY TotalSales DESC",
                description="按产品类别分析销售额和产品数量分布"
            ),
            ExampleQuery(
                query="订单量最大的客户",
                category="客户分析",
                sql_pattern="SELECT c.CompanyName, c.Country, COUNT(o.OrderID) as OrderCount, SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) as TotalAmount FROM customers c JOIN orders o ON c.CustomerID = o.CustomerID JOIN orderdetails od ON o.OrderID = od.OrderID GROUP BY c.CustomerID, c.CompanyName, c.Country ORDER BY TotalAmount DESC LIMIT 10",
                description="分析高价值客户，了解主要客户来源国家"
            ),
            ExampleQuery(
                query="员工销售业绩排名",
                category="员工分析",
                sql_pattern="SELECT CONCAT(e.FirstName, ' ', e.LastName) as EmployeeName, e.Title, COUNT(o.OrderID) as OrderCount, SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) as TotalSales FROM employees e JOIN orders o ON e.EmployeeID = o.EmployeeID JOIN orderdetails od ON o.OrderID = od.OrderID GROUP BY e.EmployeeID ORDER BY TotalSales DESC",
                description="评估员工销售绩效，支持业绩考核"
            ),
            ExampleQuery(
                query="需要补货的产品",
                category="库存分析",
                sql_pattern="SELECT p.ProductName, c.CategoryName, s.CompanyName as Supplier, p.UnitsInStock, p.ReorderLevel, p.UnitsOnOrder FROM products p JOIN categories c ON p.CategoryID = c.CategoryID JOIN suppliers s ON p.SupplierID = s.SupplierID WHERE p.UnitsInStock <= p.ReorderLevel AND p.Discontinued = 0 ORDER BY (p.ReorderLevel - p.UnitsInStock) DESC",
                description="识别库存不足需要补货的产品"
            ),
            ExampleQuery(
                query="各承运商的运费统计",
                category="物流分析",
                sql_pattern="SELECT s.CompanyName as ShipperName, COUNT(o.OrderID) as ShipmentCount, AVG(o.Freight) as AvgFreight, SUM(o.Freight) as TotalFreight FROM orders o JOIN shippers s ON o.ShipVia = s.ShipperID WHERE o.ShippedDate IS NOT NULL GROUP BY s.ShipperID, s.CompanyName ORDER BY ShipmentCount DESC",
                description="分析各承运商的使用频率和运费成本"
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
            # 同时保存术语词典内容到JSON，确保没有CSV文件时也能恢复
            'term_dictionary': self.term_dictionary.terms if self.term_dictionary else {},
            'example_query_threshold': self.example_query_threshold,
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
        """获取相关示例查询
        
        使用jieba中文分词进行智能匹配，支持中英文混合查询。
        """
        if not self.example_queries:
            return ""
        
        query_lower = query.lower()
        relevant_examples = []
        
        # 获取查询的分词结果
        query_words = self._tokenize_text(query_lower)
        
        for example in self.example_queries:
            # 获取示例的分词结果
            example_words = self._tokenize_text(example.query.lower())
            
            # 计算词汇重叠度（Jaccard相似度）
            overlap = len(example_words & query_words)
            union = len(example_words | query_words)
            
            if overlap > 0 and union > 0:
                score = overlap / union
                # 只有相似度超过阈值才添加
                if score >= self.example_query_threshold:
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
    
    def _tokenize_text(self, text: str) -> set:
        """智能分词：支持中英文混合文本
        
        对于中文使用jieba分词，对于英文使用空格分词，
        同时过滤掉单字符和标点符号。
        """
        if JIEBA_AVAILABLE:
            # 使用jieba进行中文分词
            words = set(jieba.lcut(text))
        else:
            # 降级方案：使用字符级2-gram
            words = set()
            # 空格分词（处理英文）
            words.update(text.split())
            # 字符级2-gram（处理中文）
            if len(text) >= 2:
                words.update(text[i:i+2] for i in range(len(text) - 1))
        
        # 过滤掉单字符、空白和标点符号
        stop_chars = set(' \t\n\r，。！？、；：""''（）【】《》')
        filtered_words = {
            w for w in words 
            if len(w) > 1 and not all(c in stop_chars for c in w)
        }
        
        return filtered_words if filtered_words else words
    
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
            'last_updated': getattr(self, '_last_updated', None),
            'current_database': self.get_current_database()
        }
    
    def get_current_database(self) -> str:
        """获取当前配置的数据库名称"""
        if 'adventureworks' in self.config_path.lower():
            return 'adventureworks'
        return 'northwind'
    
    def switch_database(self, database: str) -> bool:
        """
        切换数据库配置
        
        Args:
            database: 数据库名称 ("northwind" 或 "adventureworks")
            
        Returns:
            切换是否成功
        """
        database = database.lower().strip()
        
        # 定义数据库配置路径映射
        db_config_map = {
            'northwind': 'data/prompt_config.json',
            'adventureworks': 'data/prompt_config_adventureworks.json'
        }
        
        if database not in db_config_map:
            logger.error(f"不支持的数据库: {database}")
            return False
        
        new_config_path = db_config_map[database]
        
        # 检查配置文件是否存在
        if not Path(new_config_path).exists():
            logger.error(f"数据库配置文件不存在: {new_config_path}")
            return False
        
        # 如果已经是当前数据库，无需切换
        if self.config_path == new_config_path:
            logger.info(f"已经是 {database} 数据库配置")
            return True
        
        try:
            # 保存当前配置（可选）
            # self.save_config()
            
            # 更新配置路径并重新加载
            self.config_path = new_config_path
            self.load_config()
            
            logger.info(f"数据库配置已切换到: {database}")
            return True
        except Exception as e:
            logger.error(f"切换数据库配置失败: {e}")
            return False
    
    @staticmethod
    def get_available_databases() -> Dict[str, str]:
        """
        获取可用的数据库列表
        
        Returns:
            数据库名称到配置路径的映射
        """
        databases = {}
        
        # Northwind
        if Path('data/prompt_config.json').exists():
            databases['northwind'] = 'data/prompt_config.json'
        
        # AdventureWorks
        if Path('data/prompt_config_adventureworks.json').exists():
            databases['adventureworks'] = 'data/prompt_config_adventureworks.json'
        
        return databases

# ============================================
# EnhancedPromptBuilder - 增强的Prompt构建器
# (从 prompt_integration.py 合并)
# ============================================

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
        
        # 添加 SQL-only 输出约束，确保 LLM 只输出纯 SQL
        sql_only_constraint = """

【输出格式要求】
请直接输出 SQL 语句，不要包含任何解释、分析或说明文字。
- 直接以 SELECT、WITH、INSERT、UPDATE 或 DELETE 开头
- 不要使用 markdown 代码块 (```)
- 不要添加步骤说明或查询分析
- 只输出一条完整的 SQL 语句

【Top N查询提示】
如果用户问题包含"前N个"、"最高的N个"、"top N"等表述，请务必在SQL末尾添加 LIMIT N 子句。"""
        
        return base_prompt + sql_only_constraint
    
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


# ============================================
# 便捷函数
# ============================================

def create_enhanced_prompt_builder(config: Dict[str, Any]) -> EnhancedPromptBuilder:
    """创建增强的Prompt构建器"""
    return EnhancedPromptBuilder(config)


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

# ============================================
# 检索结果格式化工具类
# ============================================

class RetrievalResultFormatter:
    """
    检索结果格式化工具类
    
    用于将 retrieve_context() 的结果格式化为人类可读的展示形式，
    供前端 UI 展示"Agent 思考过程"。
    """
    
    @staticmethod
    def format_for_display(retrieval_result: Dict[str, Any]) -> Dict[str, str]:
        """
        将检索结果格式化为前端展示格式
        
        Args:
            retrieval_result: retrieve_context() 返回的结果
            
        Returns:
            Dict: {
                'rough_candidates_display': str,
                'core_tables_display': str,
                'matched_terms_display': str,
                'matched_examples_display': str,
                'metrics_display': str
            }
        """
        result = {}
        
        # 粗排候选表
        rough = retrieval_result.get('rough_candidates', [])
        if rough:
            lines = [f"{i+1}. **{c['table_name']}** (分数: {c['score']:.3f})\n   {c['description'][:60]}..." 
                    for i, c in enumerate(rough[:10])]
            result['rough_candidates_display'] = "\n".join(lines)
        else:
            result['rough_candidates_display'] = "无候选表"
        
        # 精排核心表
        core = retrieval_result.get('core_tables', [])
        if core:
            result['core_tables_display'] = ", ".join([f"`{t}`" for t in core])
        else:
            result['core_tables_display'] = "无核心表"
        
        # 匹配术语
        terms = retrieval_result.get('matched_terms', [])
        if terms:
            lines = [f"- **{t['term']}**: {t['explanation'][:50]}..." for t in terms]
            result['matched_terms_display'] = "\n".join(lines)
        else:
            result['matched_terms_display'] = "无匹配术语"
        
        # 匹配示例
        examples = retrieval_result.get('matched_examples', [])
        if examples:
            lines = [f"- {ex['query']}" for ex in examples]
            result['matched_examples_display'] = "\n".join(lines)
        else:
            result['matched_examples_display'] = "无匹配示例"
        
        # 性能指标
        metrics = retrieval_result.get('metrics', {})
        result['metrics_display'] = (
            f"粗排: {metrics.get('rough_latency_ms', 0):.1f}ms | "
            f"精排: {metrics.get('pruning_latency_ms', 0):.1f}ms | "
            f"总计: {metrics.get('total_latency_ms', 0):.1f}ms"
        )
        
        return result


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
    
    # 测试EnhancedPromptBuilder
    config = {
        'api_base': 'http://aidemo.intel.cn/v1',
        'model_name': 'DeepSeek-V3.1'
    }
    
    builder = EnhancedPromptBuilder(config)
    prompt = builder.build_sql_generation_prompt(
        user_query="分析最近三个月的销售趋势",
        schema_info="orders表包含订单信息...",
        rag_context="销售趋势分析..."
    )
    
    print("生成的Prompt:")
    print(prompt[:500] + "...")


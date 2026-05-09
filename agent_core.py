import pandas as pd
import os
import httpx
import json
import time
import logging
from datetime import datetime
from typing import Optional, Generator, Tuple, Dict, Any, List
from openai import OpenAI

# 导入错误上下文重试机制
from error_context_system import (
    ErrorCollector, ErrorContextManager, PromptEnhancer, 
    ErrorInfo, RetryContext, ErrorCategory, ErrorSeverity
)


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
                # 注意：新版本的 httpx 不支持 proxies 参数，使用 proxy 或不设置
                return OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    http_client=httpx.Client(),
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

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError, OperationalError, ProgrammingError
from query_possibility_generator import QueryPossibilityGenerator, QueryPossibility

try:
    from hardware.universal_hardware_optimizer import universal_optimizer
    CACHE_AVAILABLE = True
except ImportError:
    universal_optimizer = None
    CACHE_AVAILABLE = False

# 🧠 集成Prompt模板系统（包含二阶段检索结果格式化器）
try:
    from prompt_template_system import EnhancedPromptBuilder, RetrievalResultFormatter
    PROMPT_TEMPLATE_AVAILABLE = True
    KNOWLEDGE_RAG_AVAILABLE = True
except ImportError:
    PROMPT_TEMPLATE_AVAILABLE = False
    KNOWLEDGE_RAG_AVAILABLE = False
    print("⚠️ Prompt模板系统不可用，使用传统Prompt构建")
    print("⚠️ 二阶段检索 Prompt 构建器不可用")

# --- 全局配置 ---
# 强制移除系统代理，防止连接 DeepSeek/OpenAI 时的 SSL 错误
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['all_proxy'] = ''

class Text2SQLAgent:
    """
    智能 Text-to-SQL 代理核心类。
    负责协调 RAG 检索、LLM 推理、SQL 执行以及结果解释的全流程。
    """

    def __init__(
        self, 
        api_key: str, 
        base_url: str, 
        model_name: str, 
        db_uris: List[str], 
        rag_engine: Any, 
        max_retries: int = 3, 
        max_candidates: int = 3, 
        log_file: str = "data/agent.log",
        config: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
        # ⭐ 新增: Reasoner 模型配置 (用于自愈重试)
        reasoner_model: Optional[str] = None,
        use_reasoner_for_healing: bool = False
    ):
        """
        初始化 Agent。

        :param api_key: LLM API 密钥
        :param base_url: LLM API 基础地址
        :param model_name: 模型名称 (如 deepseek-reasoner)
        :param db_uris: 数据库连接字符串列表
        :param rag_engine: 已初始化的 IntelRAG 实例
        :param max_retries: SQL 生成的最大重试次数
        :param max_candidates: 歧义分析时的候选数量
        :param log_file: 日志文件路径
        :param config: 完整的配置字典，用于Prompt模板系统
        """
        # 1. API 客户端初始化
        clean_url = base_url.strip().rstrip('/')
        if not clean_url.endswith("v1"):
            clean_url += "/v1"
            
        self.client = create_openai_client_safe(api_key, clean_url, 60.0)
        
        # 2. 核心属性
        self.model_name = model_name
        self.db_uris = db_uris
        self.rag = rag_engine
        self.max_retries = max_retries
        self.max_candidates = max_candidates
        self.log_file = log_file
        self._config_provided = config is not None  # 记录是否显式传入config
        self.config = config or {}
        self.temperature = temperature  # LLM 温度参数，0.0 为确定性输出
        
        # ⭐ 新增: Reasoner 模型配置 (用于自愈重试)
        self.reasoner_model = reasoner_model  # 如: "deepseek-reasoner"
        self.use_reasoner_for_healing = use_reasoner_for_healing
        if self.use_reasoner_for_healing and self.reasoner_model:
            self._write_log(f"✨ Reasoner 自愈模式已启用: {self.reasoner_model}")
        
        # 3. 🧠 初始化Prompt模板系统
        # ⭐ 只有显式传入非空config时才启用KECA (EnhancedPromptBuilder)
        # 评测时传入 config=None 将禁用KECA，保持 prompt_builder=None
        if PROMPT_TEMPLATE_AVAILABLE and self._config_provided:
            try:
                self.prompt_builder = EnhancedPromptBuilder(self.config)
                self._write_log("✅ Prompt模板系统初始化成功(KECA已启用)")
            except Exception as e:
                self.prompt_builder = None
                self._write_log(f"⚠️ Prompt模板系统初始化失败: {e}")
        else:
            self.prompt_builder = None
            if not self._config_provided:
                self._write_log("ℹ️ KECA已禁用(未传入config)")
        
        # 3.1 🆕 最近一次检索结果缓存（供前端展示"Agent思考过程"）
        # RetrievalResultFormatter 已整合到 prompt_template_system 中
        self.last_retrieval_result = None
        
        # 4. 新增：可能性生成器（集成LLM和术语词典）
        term_dict = None
        if hasattr(self, 'prompt_builder') and self.prompt_builder:
            term_dict = getattr(self.prompt_builder.manager, 'term_dictionary', None)
        elif PROMPT_TEMPLATE_AVAILABLE:
            try:
                from prompt_template_system import PromptTemplateManager
                temp_manager = PromptTemplateManager()
                term_dict = temp_manager.term_dictionary
            except Exception:
                pass
        
        # 初始化可能性生成器，传入LLM客户端用于智能歧义分析
        self.possibility_generator = QueryPossibilityGenerator(
            llm_client=self.client,
            model_name=self.model_name,
            term_dictionary=term_dict
        )
        
        # 5. 新增：错误上下文重试机制
        self.error_context_manager = ErrorContextManager(max_history=10)
        self.prompt_enhancer = PromptEnhancer(max_context_length=1000)
        
        # 6. 日志与文件系统准备
        self._setup_logging()
        
        # 7. 数据库引擎初始化
        self.engine = None
        self._init_db_connection()
    
    def reset_state(self) -> None:
        """
        重置 Agent 状态，用于评测时确保每个 case 独立运行
        
        重置内容：
        - 错误上下文历史
        """
        if hasattr(self, 'error_context_manager') and self.error_context_manager:
            self.error_context_manager.clear()
        self._write_log("🔄 Agent 状态已重置")

    def _get_model_for_attempt(self, attempt: int) -> str:
        """
        根据重试次数选择合适的模型
        
        策略：
        - attempt == 1 (首次生成): 使用主模型 (deepseek-chat)
        - attempt > 1 (自愈重试): 如果启用 Reasoner 模式，使用推理模型
        
        :param attempt: 当前尝试次数 (1-based)
        :return: 模型名称
        """
        # 首次生成使用主模型
        if attempt == 1:
            return self.model_name
        
        # 自愈重试: 检查是否启用 Reasoner 模式
        if self.use_reasoner_for_healing and self.reasoner_model:
            self._write_log(f"🧠 [自愈] 第{attempt}次重试，切换至 Reasoner: {self.reasoner_model}")
            return self.reasoner_model
        
        # 默认使用主模型
        return self.model_name

    def _setup_logging(self):
        """确保日志目录存在"""
        try:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        except OSError as e:
            print(f"⚠️ [System] 无法创建日志目录: {e}")

    def _init_db_connection(self):
        """初始化数据库连接池"""
        if self.db_uris:
            try:
                # 默认使用第一个数据库连接
                self.engine = create_engine(self.db_uris[0])
                # 测试连接
                with self.engine.connect() as conn:
                    pass
                self._write_log(f"数据库连接成功: {self.db_uris[0]}")
            except Exception as e:
                self._write_log(f"❌ 数据库连接失败: {e}")
                self.engine = None

    def _detect_db_type(self) -> str:
        """
        检测当前连接的数据库类型
        
        Returns:
            数据库类型字符串: 'mysql', 'sqlite', 'sqlserver', 'postgresql'
        """
        if not self.db_uris:
            return "mysql"  # 默认
        
        uri = self.db_uris[0].lower()
        
        if 'sqlite' in uri:
            return 'sqlite'
        elif 'mssql' in uri or 'sqlserver' in uri or 'pyodbc' in uri:
            return 'sqlserver'
        elif 'postgresql' in uri or 'postgres' in uri:
            return 'postgresql'
        else:
            return 'mysql'

    def retrieve_context_two_stage(self, query: str, enable_pruning: bool = True) -> Dict:
        """
        二阶段混合检索 (Knowledge Scheduler)
        
        调用 RAG Engine 的三维统一检索，获取:
        - Schema 精排结果 (粗排 -> LLM精排)
        - Few-Shot 匹配示例
        - Terms 术语匹配
        
        Args:
            query: 用户自然语言查询
            enable_pruning: 是否启用 LLM 精排（实验模式控制）
            
        Returns:
            Dict: retrieve_context() 返回的检索结果
        """
        if not hasattr(self.rag, 'retrieve_context'):
            self._write_log("⚠️ RAG Engine 不支持 retrieve_context，回退到传统检索")
            return None
        
        # 构建配置（从 prompt_builder 获取 KECA 配置）
        keca_config = None
        if hasattr(self, 'prompt_builder') and self.prompt_builder:
            keca_config = {
                'example_queries': getattr(self.prompt_builder.manager, 'example_queries', []),
                'term_dictionary': {}
            }
            # 获取术语词典
            term_dict = getattr(self.prompt_builder.manager, 'term_dictionary', None)
            if term_dict and hasattr(term_dict, 'terms'):
                keca_config['term_dictionary'] = term_dict.terms
        
        # 调用三维统一检索
        try:
            retrieval_result = self.rag.retrieve_context(
                query=query,
                config=keca_config,
                llm_client=self.client,
                model_name=self.model_name,
                db_engine=self.engine,
                enable_pruning=enable_pruning,
                rough_top_k=12,
                include_sample_data=True
            )
            
            # 缓存检索结果（供前端展示）
            self.last_retrieval_result = retrieval_result
            
            self._write_log(f"二阶段检索完成: {len(retrieval_result.get('core_tables', []))} 核心表, "
                          f"{len(retrieval_result.get('matched_examples', []))} 示例, "
                          f"{len(retrieval_result.get('matched_terms', []))} 术语")
            
            return retrieval_result
            
        except Exception as e:
            self._write_log(f"❌ 二阶段检索失败: {e}")
            return None

    def get_retrieval_display_info(self) -> Optional[Dict[str, str]]:
        """
        获取检索结果的展示信息（供前端 UI 展示"Agent思考过程"）
        
        Returns:
            格式化的展示信息字典，包含:
            - rough_candidates_display: 粗排候选表
            - core_tables_display: 精排核心表
            - matched_terms_display: 匹配术语
            - matched_examples_display: 匹配示例
            - metrics_display: 性能指标
        """
        if not self.last_retrieval_result:
            return None
        
        if not KNOWLEDGE_RAG_AVAILABLE:
            return None
        
        return RetrievalResultFormatter.format_for_display(self.last_retrieval_result)

    # =====================================================================
    # 🔧 P1: 扩展错误类型检测
    # =====================================================================
    
    def _detect_sql_error_type(self, error_message: str) -> Tuple[str, str]:
        """
        P1: 检测 SQL 错误类型并返回修复建议
        
        扩展自愈范围：不仅处理"表不存在"，还处理列错误、语法错误等。
        
        Args:
            error_message: SQL 执行错误信息
            
        Returns:
            (错误类型, 修复建议)
        """
        import re
        
        error_patterns = {
            "unknown_table": [
                r"Table '.*' doesn't exist",
                r"no such table:",
                r"Unknown table",
            ],
            "unknown_column": [
                r"Unknown column '(\w+)'",
                r"no such column: (\w+)",
                r"column \"(\w+)\" does not exist",
            ],
            "ambiguous_column": [
                r"Column '(\w+)' in .* is ambiguous",
                r"ambiguous column name: (\w+)",
            ],
            "syntax_error": [
                r"You have an error in your SQL syntax",
                r"syntax error at or near",
                r"near \".*\": syntax error",
            ],
            "join_error": [
                r"Unknown table '(\w+)' in.*join",
                r"ON clause contains column that is not in any table",
            ],
            "aggregation_error": [
                r"isn't in GROUP BY",
                r"must appear in the GROUP BY clause",
            ],
        }
        
        fix_suggestions = {
            "unknown_table": "请核实表名，参考 Database Index 中的正确表名",
            "unknown_column": "请检查列名拼写，参考表 Schema 中的正确列名",
            "ambiguous_column": "请在列名前添加表别名，如 `t.column`",
            "syntax_error": "请检查 SQL 语法，特别是括号、逗号、引号的匹配",
            "join_error": "请确认 JOIN 条件中的表名和列名正确",
            "aggregation_error": "请确保 SELECT 中的非聚合列都在 GROUP BY 中",
        }
        
        for error_type, patterns in error_patterns.items():
            for pattern in patterns:
                if re.search(pattern, error_message, re.IGNORECASE):
                    suggestion = fix_suggestions.get(error_type, "请检查 SQL 语法")
                    self._write_log(f"🔍 [自愈] 检测到错误类型: {error_type}")
                    return error_type, suggestion
        
        return "other", "请仔细检查 SQL 语法"

    # =====================================================================
    # 🔧 动态自愈补全机制 (Dynamic Self-Healing)
    # =====================================================================
    
    def _detect_table_not_found_error(self, error_message: str) -> Optional[str]:
        """
        检测是否为「表不存在」类型的错误，并提取目标表名
        
        支持的错误模式:
        - MySQL: Table 'database.tablename' doesn't exist
        - MySQL: Unknown table 'tablename'
        - SQLite: no such table: tablename
        - SQL Server: Invalid object name 'tablename'
        
        Args:
            error_message: SQL 执行错误信息
            
        Returns:
            检测到的目标表名，未检测到则返回 None
        """
        import re
        
        patterns = [
            # MySQL: Table 'db.table' doesn't exist
            r"Table '(?:[\w]+\.)?(\w+)' doesn't exist",
            # MySQL: Unknown table 'table'
            r"Unknown table '(\w+)'",
            # SQLite: no such table: table
            r"no such table:\s*(\w+)",
            # SQL Server: Invalid object name 'table'
            r"Invalid object name '(\w+)'",
            # Generic: table 'tablename' not found
            r"table '(\w+)' not found",
            # Generic: relation "tablename" does not exist (PostgreSQL)
            r'relation "(\w+)" does not exist',
        ]
        
        error_lower = error_message.lower()
        
        for pattern in patterns:
            match = re.search(pattern, error_message, re.IGNORECASE)
            if match:
                table_name = match.group(1)
                self._write_log(f"🔍 [自愈] 检测到表不存在错误: '{table_name}'")
                return table_name
        
        return None
    
    def _match_table_from_index(self, target_table: str) -> Optional[str]:
        """
        从 Database Index 中模糊匹配实际存在的表名
        
        匹配策略（按优先级）:
        1. 精确匹配（忽略大小写）
        2. 去除常见前缀后匹配（tbl_, tb_, t_）
        3. 包含关系匹配
        4. 编辑距离最小匹配（阈值 ≤ 2）
        
        Args:
            target_table: 错误消息中提取的目标表名
            
        Returns:
            匹配到的实际表名，未匹配到则返回 None
        """
        if not hasattr(self.rag, '_get_database_index'):
            self._write_log("⚠️ [自愈] RAG 引擎不支持 _get_database_index")
            return None
        
        database_index = self.rag._get_database_index()
        if not database_index:
            self._write_log("⚠️ [自愈] Database Index 为空")
            return None
        
        target_lower = target_table.lower()
        
        # 1. 精确匹配（忽略大小写）
        for table in database_index:
            if table.lower() == target_lower:
                self._write_log(f"✅ [自愈] 精确匹配: {target_table} -> {table}")
                return table
        
        # 2. 去除常见前缀后匹配
        prefixes = ['tbl_', 'tb_', 't_', 'dim_', 'fact_', 'vw_']
        for prefix in prefixes:
            if target_lower.startswith(prefix):
                stripped = target_lower[len(prefix):]
                for table in database_index:
                    if table.lower() == stripped or table.lower().endswith(stripped):
                        self._write_log(f"✅ [自愈] 前缀去除匹配: {target_table} -> {table}")
                        return table
        
        # 3. 包含关系匹配（目标名是实际表名的子串或反之）
        for table in database_index:
            table_lower = table.lower()
            if target_lower in table_lower or table_lower in target_lower:
                self._write_log(f"✅ [自愈] 包含匹配: {target_table} -> {table}")
                return table
        
        # 4. 编辑距离匹配（Levenshtein，阈值 2）
        def levenshtein_distance(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]
        
        best_match = None
        best_distance = float('inf')
        
        for table in database_index:
            distance = levenshtein_distance(target_lower, table.lower())
            if distance <= 2 and distance < best_distance:
                best_distance = distance
                best_match = table
        
        if best_match:
            self._write_log(f"✅ [自愈] 编辑距离匹配 (距离={best_distance}): {target_table} -> {best_match}")
            return best_match
        
        self._write_log(f"⚠️ [自愈] 未能匹配表名: {target_table}")
        return None
    
    def _get_dynamic_table_schema(self, table_name: str) -> Optional[str]:
        """
        动态获取指定表的详细 Schema 信息
        
        用于自愈机制：当检测到表不存在错误时，获取正确表的 Schema
        并追加到下次重试的 context 中
        
        Args:
            table_name: 表名
            
        Returns:
            格式化的 Schema 信息字符串，获取失败返回 None
        """
        if not hasattr(self.rag, '_extract_table_details'):
            self._write_log("⚠️ [自愈] RAG 引擎不支持 _extract_table_details")
            return None
        
        try:
            # 调用 RAG 引擎获取表详情
            table_details = self.rag._extract_table_details([table_name])
            
            if not table_details:
                self._write_log(f"⚠️ [自愈] 未找到表详情: {table_name}")
                return None
            
            detail = table_details[0]
            
            # 格式化为 Schema 字符串
            schema_lines = [
                f"\n## [自愈补全] 表 `{table_name}` 的详细 Schema",
                f"**描述**: {detail.get('description', '无描述')}",
                "",
                "**字段列表**:"
            ]
            
            for col in detail.get('columns', []):
                col_line = f"  - `{col['name']}` ({col['type']}): {col['description']}"
                schema_lines.append(col_line)
            
            if detail.get('foreign_keys'):
                schema_lines.append("")
                schema_lines.append("**外键关系**:")
                for fk in detail['foreign_keys']:
                    schema_lines.append(f"  - {fk}")
            
            # 尝试获取样本数据
            if hasattr(self.rag, '_get_sample_data') and self.engine:
                sample = self.rag._get_sample_data(table_name, self.engine, limit=3)
                if sample:
                    schema_lines.append("")
                    schema_lines.append("**样本数据 (前3行)**:")
                    for i, row in enumerate(sample[:3], 1):
                        schema_lines.append(f"  {i}. {row}")
            
            schema_text = "\n".join(schema_lines)
            self._write_log(f"✅ [自愈] 获取表 Schema 成功: {table_name} ({len(detail.get('columns', []))} 列)")
            
            return schema_text
            
        except Exception as e:
            self._write_log(f"❌ [自愈] 获取表 Schema 失败: {e}")
            return None
    
    def _apply_self_healing(self, error_message: str, current_context: str, retry_attempt: int = 1) -> Tuple[bool, str]:
        """
        应用动态自愈补全机制（P1 增强版）
        
        支持多种错误类型的修复建议：
        - unknown_table: 表不存在 → 动态 Schema 补全
        - unknown_column: 列不存在 → 补充相关表的列信息
        - ambiguous_column: 歧义列 → 提示添加表别名
        - syntax_error: 语法错误 → 通用修复建议
        - join_error: JOIN 错误 → 修复 JOIN 条件
        - aggregation_error: 聚合错误 → GROUP BY 修复建议
        
        Args:
            error_message: SQL 执行错误信息
            current_context: 当前的 Schema 上下文
            retry_attempt: 当前重试次数（用于 P3 差异化策略）
            
        Returns:
            (是否应用了自愈, 更新后的上下文)
        """
        # P1: 检测错误类型
        error_type, fix_suggestion = self._detect_sql_error_type(error_message)
        
        # P3: 差异化重试策略
        retry_hint = self._get_retry_enhancement(error_type, retry_attempt)
        
        # 情况 1: 表不存在 → 动态 Schema 补全 + FK 关联表
        if error_type == "unknown_table":
            target_table = self._detect_table_not_found_error(error_message)
            if target_table:
                matched_table = self._match_table_from_index(target_table)
                if matched_table:
                    schema_supplement = self._get_dynamic_table_schema(matched_table)
                    if schema_supplement:
                        # 🆕 增强：补充 FK 关联表 (最多 2 个)
                        fk_supplements = []
                        if self.rag and hasattr(self.rag, '_complete_table_dependencies'):
                            try:
                                related_tables = self.rag._complete_table_dependencies([matched_table])
                                # 跳过第一个（主表本身），取最多 2 个关联表
                                for related in related_tables[1:3]:
                                    if related.lower() != matched_table.lower():
                                        related_schema = self._get_dynamic_table_schema(related)
                                        if related_schema:
                                            fk_supplements.append(related_schema)
                                            self._write_log(f"📎 [自愈] 补充关联表: {related}")
                            except Exception as e:
                                self._write_log(f"⚠️ [自愈] 关联表补全失败: {e}")
                        
                        healing_header = "\n\n# 🔧 动态自愈补全 (Schema Injection)\n"
                        healing_note = f"> 系统检测到您可能想查询表 `{matched_table}`，以下是该表的详细信息：\n"
                        
                        # 组合主表和关联表 Schema
                        full_supplement = schema_supplement
                        if fk_supplements:
                            full_supplement += "\n\n> 以下是可能需要 JOIN 的关联表：\n" + "\n\n".join(fk_supplements)
                        
                        enhanced_context = current_context + healing_header + healing_note + full_supplement + retry_hint
                        self._write_log(f"✅ [自愈] Schema 补全: {target_table} -> {matched_table} (+{len(fk_supplements)} 关联表)")
                        return True, enhanced_context
        
        # 情况 2: 其他错误类型 → 添加修复建议
        if error_type != "other":
            healing_header = "\n\n# 🔧 自愈修复建议\n"
            healing_note = f"> ⚠️ **错误类型**: {error_type}\n> **修复建议**: {fix_suggestion}\n"
            enhanced_context = current_context + healing_header + healing_note + retry_hint
            self._write_log(f"✅ [自愈] 生成修复建议: {error_type} - {fix_suggestion}")
            return True, enhanced_context
        
        # 情况 3: 未知错误类型 → 仅添加通用提示
        general_hint = "\n\n# 🔧 请仔细检查 SQL 语法\n> 系统检测到执行错误，请参照 Schema 重新生成 SQL。\n"
        return True, current_context + general_hint + retry_hint
    
    def _get_retry_enhancement(self, error_type: str, attempt: int) -> str:
        """
        P3: 根据错误类型和重试次数返回差异化增强提示
        
        避免每次重试都使用相同策略，减少重复错误。
        
        Args:
            error_type: 错误类型
            attempt: 当前尝试次数
            
        Returns:
            增强提示字符串
        """
        enhancements = {
            "unknown_table": "请仔细核对表名，参考 Database Index",
            "unknown_column": "请参考表的详细 Schema 确认列名",
            "syntax_error": "请检查 SQL 语法，特别是 CTE、窗口函数的写法",
            "join_error": "请确认 JOIN 条件中的表名和列名正确",
            "ambiguous_column": "请在列名前添加表别名 (如 `t.column`)",
            "aggregation_error": "请确保 SELECT 中的非聚合列都在 GROUP BY 中",
        }
        
        base = enhancements.get(error_type, "请仔细检查 SQL")
        
        # 第 2 次以上重试，增加更强的约束
        if attempt >= 2:
            base += "。**第 3 次尝试：请使用最简单的 SQL 写法，避免复杂子查询和 CTE。**"
        
        return f"\n\n> 🔧 **修复提示 (尝试 {attempt})**: {base}\n"


    def _write_log(self, content: str):
        """
        写入本地运行日志，带时间戳。
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {content}\n"
        # 打印到控制台方便调试
        print(log_entry.strip())
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception:
            pass

    def _estimate_query_complexity(self, query: str) -> str:
        """
        智能查询复杂度评估（语义相似度 + 规则兜底）
        
        使用 RAG Embedding 模型将查询与预定义的复杂度原型比较，
        返回语义最相似的复杂度级别。当置信度低时回退到规则方法。
        
        Returns:
            'simple' - 单表简单查询，跳过 RAG
            'medium' - 中等复杂度，使用有限 RAG (top_k=2)
            'complex' - 高度复杂，使用完整 RAG (top_k=5)
        """
        import numpy as np
        
        # 尝试语义方法
        semantic_result = self._estimate_complexity_semantic(query)
        if semantic_result:
            return semantic_result
        
        # 回退到规则方法
        return self._estimate_complexity_rules(query)
    
    def _estimate_complexity_semantic(self, query: str) -> str:
        """
        基于语义相似度的复杂度评估
        
        将查询与预定义的复杂度原型进行余弦相似度比较，
        返回最匹配的复杂度级别。
        """
        import numpy as np
        
        # 检查 RAG 引擎是否可用
        if not hasattr(self, 'rag_engine') or self.rag_engine is None:
            return None
        if not hasattr(self.rag_engine, '_get_embedding'):
            return None
        if self.rag_engine.model is None:
            return None
        
        # 初始化原型缓存（仅首次调用时计算）
        if not hasattr(self, '_complexity_prototype_cache'):
            self._init_complexity_prototype_cache()
        
        if not self._complexity_prototype_cache:
            return None
        
        try:
            # 获取查询的 embedding
            query_emb = self.rag_engine._get_embedding(query)
            query_norm = np.linalg.norm(query_emb)
            if query_norm < 1e-8:
                return None
            
            # 与各复杂度原型比较，找最高相似度
            best_complexity = None
            best_score = -1
            
            for complexity, proto_embeddings in self._complexity_prototype_cache.items():
                for proto_emb in proto_embeddings:
                    proto_norm = np.linalg.norm(proto_emb)
                    if proto_norm < 1e-8:
                        continue
                    # 余弦相似度
                    sim = np.dot(query_emb, proto_emb) / (query_norm * proto_norm)
                    if sim > best_score:
                        best_score = sim
                        best_complexity = complexity
            
            # 置信度检查：分数太低时返回 None，使用规则兜底
            if best_score < 0.55:
                self._write_log(f"复杂度语义匹配置信度低 ({best_score:.3f})，使用规则兜底")
                return None
            
            self._write_log(f"复杂度语义匹配: {best_complexity} (相似度={best_score:.3f})")
            return best_complexity
            
        except Exception as e:
            self._write_log(f"语义复杂度评估异常: {e}")
            return None
    
    def _init_complexity_prototype_cache(self):
        """
        初始化复杂度原型的 Embedding 缓存
        
        从 config 中读取 complexity_prototypes，预计算所有原型的 embedding，
        避免每次查询时重复计算。
        """
        import numpy as np
        self._complexity_prototype_cache = {}
        
        # 从 config 读取原型
        prototypes = None
        if self.config and 'complexity_prototypes' in self.config:
            prototypes = self.config['complexity_prototypes']
        elif hasattr(self, 'prompt_builder') and self.prompt_builder:
            manager = getattr(self.prompt_builder, 'manager', None)
            if manager and hasattr(manager, 'config'):
                prototypes = manager.config.get('complexity_prototypes', None)
        
        if not prototypes:
            self._write_log("未找到 complexity_prototypes 配置，语义复杂度检测不可用")
            return
        
        # 预计算所有原型的 embedding
        for complexity, proto_list in prototypes.items():
            embeddings = []
            for proto_query in proto_list:
                try:
                    emb = self.rag_engine._get_embedding(proto_query)
                    if np.linalg.norm(emb) > 1e-8:
                        embeddings.append(emb)
                except Exception:
                    continue
            if embeddings:
                self._complexity_prototype_cache[complexity] = embeddings
        
        total_protos = sum(len(v) for v in self._complexity_prototype_cache.values())
        self._write_log(f"复杂度原型缓存初始化完成: {total_protos} 个原型")
    
    def _estimate_complexity_rules(self, query: str) -> str:
        """
        基于规则的复杂度评估（兜底方法）
        
        使用关键词匹配和实体计数来估算复杂度。
        """
        import re
        score = 0
        
        # 维度1：复杂关键词检测
        complex_keywords = [
            (r'每个.+的|各.+的', 3),
            (r'排名|排序|前\d+|top\s*\d+', 3),
            (r'最高|最低|最大|最小', 2),
            (r'占比|百分比|比例', 4),
            (r'连续|环比|同比|趋势', 5),
            (r'累计|累积|running', 4),  # 新增：累计计算
            (r'关联|关系|对应', 3),
            (r'对比|比较|差异', 3),
            (r'增长|下降|变化', 3),
            (r'购买过|订购过', 4),
            (r'分[组类别]', 2),
        ]
        
        for pattern, weight in complex_keywords:
            if re.search(pattern, query, re.IGNORECASE):
                score += weight
        
        # 维度2：实体数量（改进版：避免复合词误判）
        # "产品类别" 应该被视为一个概念而非两个
        compound_terms = ['产品类别', '销售订单', '采购订单', '产品库存', '订单明细']
        query_for_entity = query
        for term in compound_terms:
            query_for_entity = query_for_entity.replace(term, '_compound_')
        
        distinct_entities = set()
        entity_map = {
            r'产品|商品': 'product',
            r'订单': 'order',
            r'销售': 'sales',
            r'客户|顾客|用户': 'customer',
            r'员工|雇员': 'employee',
            r'区域|地区': 'region',
        }
        for pattern, entity in entity_map.items():
            if re.search(pattern, query_for_entity):
                distinct_entities.add(entity)
        
        if len(distinct_entities) >= 2:
            score += len(distinct_entities)
        
        # 维度3：问句长度
        if len(query) > 30:
            score += 1
        if len(query) > 50:
            score += 1
        
        # 复杂度分级
        if score <= 1:
            return 'simple'
        elif score <= 3:
            return 'medium'
        else:
            return 'complex'
    
    def _is_simple_query(self, query: str) -> bool:
        """
        判断是否为简单查询（兼容性封装）
        
        简单查询特征：
        - 单表查询 (查询所有X、统计X总数)
        - 不涉及多表关联
        - 不需要复杂的业务逻辑理解
        
        Returns:
            True 如果是简单查询，应跳过 RAG 增强
        """
        complexity = self._estimate_query_complexity(query)
        is_simple = complexity == 'simple'
        
        if is_simple:
            self._write_log(f"🎯 检测到简单查询 (复杂度={complexity})，跳过 RAG 增强: {query[:30]}...")
        
        return is_simple

    def _get_table_map(self) -> str:
        """
        生成轻量级数据库表清单（Hybrid RAG 的"全局地图"）
        
        作用：让 LLM 知道数据库中存在哪些表，即使 RAG 检索遗漏了某张表，
        LLM 也可以在自愈阶段请求查看该表的详细结构。
        
        Returns:
            格式化的表名列表字符串，Token 消耗极低（约 200 tokens）
        """
        if not hasattr(self.rag, 'documents') or not self.rag.documents:
            return ""
        
        tables = []
        for doc in self.rag.documents:
            if doc.startswith('【表名】'):
                # 提取表名（格式：【表名】TableName）
                first_line = doc.split('\n')[0]
                table_name = first_line.replace('【表名】', '').strip()
                if table_name:
                    tables.append(table_name)
        
        if not tables:
            return ""
        
        return f"=== 数据库全部 {len(tables)} 张表 ===\n{', '.join(tables)}\n"
    
    def _get_table_schema(self, table_name: str) -> str:
        """
        按需获取指定表的完整结构（Reasoning Self-Correction）
        
        当自愈阶段检测到 UNKNOWN_COLUMN 或 UNKNOWN_TABLE 错误时，
        可以调用此方法动态获取正确的表结构。
        
        Args:
            table_name: 目标表名（大小写不敏感）
        
        Returns:
            该表的完整 Schema 文档，如果未找到则返回空字符串
        """
        if not hasattr(self.rag, 'documents') or not self.rag.documents:
            return ""
        
        table_name_lower = table_name.lower()
        for doc in self.rag.documents:
            if doc.startswith('【表名】'):
                first_line = doc.split('\n')[0]
                doc_table_name = first_line.replace('【表名】', '').strip().lower()
                if doc_table_name == table_name_lower:
                    return doc
        return ""

    def _get_required_tables_from_terms(self, query: str) -> set:
        """
        从术语词典中提取查询涉及的必需表（Hint Injection）
        
        当查询包含特定业务术语时，强制返回该术语关联的表，
        无论 RAG 向量搜索分数是多少。
        
        Args:
            query: 用户自然语言查询
        
        Returns:
            需要强制注入的表名集合（小写）
        """
        required_tables = set()
        
        # 从 PromptBuilder 获取术语词典
        if hasattr(self, 'prompt_builder') and self.prompt_builder:
            term_dict = getattr(self.prompt_builder.manager, 'term_dictionary', None)
            if term_dict and hasattr(term_dict, 'terms'):
                for term, info in term_dict.terms.items():
                    if term in query:
                        # 新格式：info 是 dict {'explanation': ..., 'required_tables': [...]}
                        if isinstance(info, dict) and 'required_tables' in info:
                            for table in info['required_tables']:
                                required_tables.add(table.lower())
                        # 旧格式兼容：从 explanation 文本中提取表名
                        elif isinstance(info, str):
                            # 尝试从解释中提取表名（简单匹配）
                            import re
                            table_matches = re.findall(r'(\w+)\s*表', info)
                            for table in table_matches:
                                required_tables.add(table.lower())
        
        if required_tables:
            self._write_log(f"Hint Injection: 术语匹配到 {len(required_tables)} 张必需表: {required_tables}")
        
        return required_tables

    def _get_cte_few_shot(self, query_complexity: str = 'complex') -> str:
        """
        获取 CTE 和窗口函数的 Few-Shot 示例
        
        仅在复杂查询时注入示例，帮助 LLM 学习正确的 CTE 语法格式。
        这可以显著降低"缺失 WITH 关键字"等常见语法错误。
        
        Args:
            query_complexity: 查询复杂度 ('simple'/'medium'/'complex')
            
        Returns:
            格式化的 Few-Shot 示例字符串，或空字符串
        """
        # 仅对复杂查询注入示例
        if query_complexity != 'complex':
            return ""
        
        # 从 config 或 prompt_builder 获取 cte_few_shot
        cte_examples = []
        
        # 方式1: 直接从 config 读取
        if self.config and 'cte_few_shot' in self.config:
            cte_examples = self.config['cte_few_shot']
        
        # 方式2: 从 prompt_builder.manager 读取
        elif hasattr(self, 'prompt_builder') and self.prompt_builder:
            manager = getattr(self.prompt_builder, 'manager', None)
            if manager and hasattr(manager, 'config'):
                cte_examples = manager.config.get('cte_few_shot', [])
        
        if not cte_examples:
            return ""
        
        # 格式化示例
        few_shot_text = "\n【复杂查询参考示例 - 请严格遵循 WITH 关键字语法】\n"
        for i, example in enumerate(cte_examples[:2], 1):  # 最多展示 2 个示例以节省 Token
            few_shot_text += f"\n示例 {i}: {example.get('description', '')}\n"
            few_shot_text += f"问题: {example.get('question', '')}\n"
            few_shot_text += f"SQL:\n{example.get('sql', '')}\n"
        
        self._write_log(f"Few-Shot: 注入 {min(len(cte_examples), 2)} 个 CTE 示例")
        return few_shot_text


    def _get_db_type_info(self) -> dict:
        """
        获取当前数据库类型信息，用于生成适配的 SQL Prompt
        
        Returns:
            dict: 包含 db_type, db_expert, date_hint, syntax_tips
        """
        # 从配置或连接字符串推断数据库类型
        db_type = self.config.get("db_type", "").lower()
        
        # 如果配置中没有明确指定，从连接字符串推断
        if not db_type and self.db_uris:
            uri = self.db_uris[0].lower()
            if "mysql" in uri or "pymysql" in uri:
                db_type = "mysql"
            elif "sqlite" in uri:
                db_type = "sqlite"
            elif "postgresql" in uri or "psycopg" in uri:
                db_type = "postgresql"
            elif "mssql" in uri or "sqlserver" in uri or "pyodbc" in uri:
                db_type = "sqlserver"
            elif "oracle" in uri:
                db_type = "oracle"
            else:
                db_type = "sql"  # 通用 SQL
        
        # 根据数据库类型返回相应的信息
        db_info = {
            "mysql": {
                "db_type": "MySQL",
                "db_expert": "精通 MySQL 的高级数据库工程师",
                "date_hint": "日期处理请使用 MySQL 函数，如 `YEAR(date_col)`, `MONTH(date_col)`, `DATE_FORMAT(date_col, '%Y-%m')`, `DATEDIFF(date1, date2)`",
                "syntax_tips": [
                    "使用 LIMIT 而非 TOP 进行分页",
                    "字符串连接使用 CONCAT() 函数",
                    "GROUP BY 子句中需要包含 SELECT 中所有非聚合列",
                    "使用 IFNULL() 处理空值",
                    "支持 CTE (WITH 子句) 和窗口函数"
                ]
            },
            "sqlite": {
                "db_type": "SQLite",
                "db_expert": "精通 SQLite 的高级数据库工程师",
                "date_hint": "日期处理请使用 `strftime` 函数，如 `strftime('%Y', date_col)`, `strftime('%m', date_col)`, `julianday(date1) - julianday(date2)`",
                "syntax_tips": [
                    "使用 LIMIT 进行分页",
                    "字符串连接使用 || 运算符",
                    "支持灵活的 GROUP BY",
                    "使用 IFNULL() 或 COALESCE() 处理空值"
                ]
            },
            "postgresql": {
                "db_type": "PostgreSQL",
                "db_expert": "精通 PostgreSQL 的高级数据库工程师",
                "date_hint": "日期处理请使用 `EXTRACT(YEAR FROM date_col)`, `DATE_TRUNC('month', date_col)`, `date1 - date2`",
                "syntax_tips": [
                    "使用 LIMIT 和 OFFSET 进行分页",
                    "字符串连接使用 || 或 CONCAT()",
                    "支持丰富的窗口函数和 CTE",
                    "使用 COALESCE() 处理空值"
                ]
            },
            "sqlserver": {
                "db_type": "SQL Server",
                "db_expert": "精通 SQL Server 的高级数据库工程师",
                "date_hint": "日期处理请使用 `YEAR(date_col)`, `MONTH(date_col)`, `DATEPART()`, `DATEDIFF()`",
                "syntax_tips": [
                    "使用 TOP N 或 OFFSET-FETCH 进行分页（如 TOP 10 或 ORDER BY col OFFSET 0 ROWS FETCH NEXT 10 ROWS ONLY）",
                    "字符串连接使用 + 或 CONCAT()",
                    "使用 ISNULL() 或 COALESCE() 处理空值",
                    "CTE语法: WITH cte_name AS (SELECT...), cte2 AS (SELECT...) SELECT...（必须以WITH开头，多个CTE用逗号分隔）",
                    "窗口函数: ROW_NUMBER() OVER (PARTITION BY col ORDER BY col) 用于分组排名",
                    "Schema前缀: 表名应包含Schema前缀如 Sales.SalesOrderHeader, Production.Product"
                ]
            },
            "oracle": {
                "db_type": "Oracle",
                "db_expert": "精通 Oracle 的高级数据库工程师",
                "date_hint": "日期处理请使用 `EXTRACT(YEAR FROM date_col)`, `TO_CHAR(date_col, 'YYYY-MM')`, `date1 - date2`",
                "syntax_tips": [
                    "使用 ROWNUM 或 FETCH FIRST 进行分页",
                    "字符串连接使用 || 运算符",
                    "使用 NVL() 处理空值"
                ]
            }
        }
        
        # 默认返回通用 SQL 信息
        default_info = {
            "db_type": "SQL",
            "db_expert": "精通 SQL 的高级数据库工程师",
            "date_hint": "请使用标准 SQL 日期函数",
            "syntax_tips": ["使用标准 SQL 语法"]
        }
        
        return db_info.get(db_type, default_info)

    def _build_traditional_prompt(self, query: str, context: str, best_interpretation: str, current_try: int) -> str:
        """构建传统的SQL生成Prompt（自动适配数据库类型）"""
        # 获取数据库类型信息
        db_info = self._get_db_type_info()
        
        # 构建语法提示
        syntax_tips_str = "\n".join([f"        - {tip}" for tip in db_info["syntax_tips"]])
        
        base_prompt = f"""
        你是一个{db_info["db_expert"]}。
        
        【数据库类型】: {db_info["db_type"]}
        
        【Schema 信息】:
        {context}
        
        【用户原始问题】: "{query}"
        【已确认的业务逻辑】: "{best_interpretation}"
        
        【任务】:
        编写可执行的 {db_info["db_type"]} SQL 语句。
        
        【严格约束】:
        1. 仅输出 SQL 代码。
        2. 不要使用 Markdown 格式 (不要写 ```sql)。
        3. {db_info["date_hint"]}。
        4. 不要解释代码。
        
        【{db_info["db_type"]} 语法要点】:
{syntax_tips_str}
        """
        
        # 如果有错误历史，获取重试上下文并增强prompt
        if current_try > 1:
            retry_context = self.error_context_manager.get_retry_context(max_errors=3)
            enhanced_prompt = self.prompt_enhancer.enhance_retry_prompt(base_prompt, retry_context)
            return enhanced_prompt
        else:
            return base_prompt

    def _is_safe_query(self, sql: str) -> bool:
        """
        检查SQL语句是否为安全的查询语句
        
        允许的语句类型：
        - SELECT 语句
        - WITH 子句（CTE）开头的查询
        - EXPLAIN 语句
        - SHOW 语句（MySQL）
        - DESCRIBE/DESC 语句
        """
        sql_lower = sql.lower().strip()
        
        # 移除注释和多余空白
        import re
        sql_lower = re.sub(r'--.*$', '', sql_lower, flags=re.MULTILINE)  # 移除行注释
        sql_lower = re.sub(r'/\*.*?\*/', '', sql_lower, flags=re.DOTALL)  # 移除块注释
        sql_lower = re.sub(r'\s+', ' ', sql_lower).strip()  # 标准化空白
        
        # 如果清理后为空，则不安全
        if not sql_lower:
            return False
        
        # 检查是否包含危险关键词（即使在CTE中也不允许）
        dangerous_keywords = [
            r'\bdrop\b', r'\bdelete\b', r'\bupdate\b', r'\binsert\b',
            r'\bcreate\b', r'\balter\b', r'\btruncate\b', r'\bgrant\b',
            r'\brevoke\b', r'\bexec\b', r'\bexecute\b'
        ]
        
        for keyword in dangerous_keywords:
            if re.search(keyword, sql_lower):
                return False
        
        # 允许的安全查询模式
        safe_patterns = [
            r'^select\b',           # SELECT 语句
            r'^with\b.*select\b',   # CTE (WITH ... SELECT)
            r'^explain\b',          # EXPLAIN 语句
            r'^show\b',             # SHOW 语句 (MySQL)
            r'^describe\b',         # DESCRIBE 语句
            r'^desc\b',             # DESC 语句
            r'^\(\s*select\b',      # 括号包围的 SELECT
        ]
        
        # 检查是否匹配任何安全模式
        for pattern in safe_patterns:
            if re.match(pattern, sql_lower):
                return True
        
        return False

    def _extract_sql_from_response(self, response: str) -> str:
        """
        从 LLM 响应中智能提取 SQL 语句
        
        支持的格式：
        1. 纯 SQL 语句
        2. ```sql ... ``` 代码块
        3. 完整 CoT 格式（含 1.查询理解, 2.涉及表格, 3.SQL查询 等结构化输出）
        4. 数字编号格式 (如 "3. **SQL查询**:" 后跟代码块)
        
        集成 SQLPreValidator 进行 CTE 修复和语法预验证
        
        Returns:
            提取的 SQL 语句
        """
        import re
        
        if not response or not response.strip():
            return ""
        
        # 尝试导入 SQLPreValidator
        sql_validator = None
        try:
            from error_context_system import SQLPreValidator
            sql_validator = SQLPreValidator()
        except ImportError:
            pass
        
        def _post_process_sql(sql: str) -> str:
            """后处理: 使用 SQLPreValidator 修复和验证 SQL"""
            if not sql:
                return sql
            if sql_validator:
                fixed = sql_validator.fix_and_format(sql)
                if fixed:
                    return fixed
            return sql
        
        # 策略 0: 检测 CoT 格式 - 如果响应以数字编号开头 (如 "1. **查询理解**")
        # 这表明是完整的 CoT 输出，需要从中提取 SQL 代码块
        cot_pattern = r'^\s*1\.\s*\*?\*?查询理解'
        if re.match(cot_pattern, response, re.MULTILINE):
            self._write_log("📝 SQL提取器: 检测到CoT格式输出")
            
            # 从 CoT 中提取 ```sql ... ``` 代码块
            sql_block_pattern = r'```sql\s*(.*?)\s*```'
            matches = re.findall(sql_block_pattern, response, re.DOTALL | re.IGNORECASE)
            if matches:
                # 返回最后一个 SQL 代码块（通常是最终版本）
                extracted = matches[-1].strip()
                # 移除可能的注释行
                lines = [line for line in extracted.split('\n') if not line.strip().startswith('--')]
                extracted = '\n'.join(lines).strip()
                self._write_log(f"📝 SQL提取器: 从CoT中成功提取SQL")
                return _post_process_sql(extracted)
            
            # 尝试提取无标记代码块
            code_block_pattern = r'```\s*(.*?)\s*```'
            matches = re.findall(code_block_pattern, response, re.DOTALL)
            for match in matches:
                if self._looks_like_sql(match):
                    self._write_log(f"📝 SQL提取器: 从CoT无标记代码块提取")
                    return _post_process_sql(match.strip())
            
            # CoT 格式但未找到代码块，返回空（避免将整个 CoT 当作 SQL）
            self._write_log("⚠️ SQL提取器: CoT格式但未找到SQL代码块")
            return ""
        
        # 策略 1：尝试提取 ```sql ... ``` 代码块
        sql_block_pattern = r'```sql\s*(.*?)\s*```'
        matches = re.findall(sql_block_pattern, response, re.DOTALL | re.IGNORECASE)
        if matches:
            # 返回最后一个 SQL 代码块（通常是最终版本）
            extracted = matches[-1].strip()
            self._write_log(f"📝 SQL提取器: 从 ```sql 代码块提取")
            return _post_process_sql(extracted)
        
        # 策略 2：尝试提取 ``` ... ``` 无语言标记的代码块
        code_block_pattern = r'```\s*(.*?)\s*```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        for match in matches:
            # 检查是否看起来像 SQL
            if self._looks_like_sql(match):
                self._write_log(f"📝 SQL提取器: 从无标记代码块提取")
                return _post_process_sql(match.strip())
        
        # 策略 3: 如果响应包含 CoT 标记但没有代码块，跳过
        cot_markers = ['**查询理解**', '**涉及表格**', '**SQL查询**', '**查询说明**', '**注意事项**']
        has_cot_markers = any(marker in response for marker in cot_markers)
        if has_cot_markers:
            self._write_log("⚠️ SQL提取器: 包含CoT标记但无代码块，跳过")
            return ""
        
        # 策略 4：查找 SELECT/WITH 开头的语句（处理纯 SQL 或混合文本）
        # 使用更精确的模式匹配 SQL 语句结构
        sql_patterns = [
            # WITH ... SELECT (CTE) - 匹配到分号或文档结尾
            r'(WITH\s+\w+.*?SELECT\s+[\s\S]+?)(?:;|\n\s*\n|\n\d+\.|\Z)',
            # 普通 SELECT - 匹配到分号或文档结尾
            r'(SELECT\s+[\s\S]+?(?:FROM\s+[\s\S]+?))(?:;|\n\s*\n|\n\d+\.|\Z)',
        ]
        
        for pattern in sql_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                # 清理可能混入的 Markdown
                candidate = re.sub(r'\*\*[^*]+\*\*', '', candidate)  # 移除 **粗体**
                candidate = candidate.strip()
                if self._looks_like_sql(candidate):
                    self._write_log(f"📝 SQL提取器: 从文本中模式匹配提取")
                    return _post_process_sql(candidate)
        
        # 策略 5：如果响应本身就是纯 SQL
        if self._looks_like_sql(response):
            self._write_log(f"📝 SQL提取器: 使用原始响应（纯SQL）")
            return _post_process_sql(response.strip())
        
        # 策略 6：返回空（避免将非 SQL 内容当作 SQL）
        self._write_log(f"⚠️ SQL提取器: 无法提取有效SQL")
        return ""

    def _looks_like_sql(self, text: str) -> bool:
        """判断文本是否像 SQL 语句"""
        if not text:
            return False
        text_lower = text.lower().strip()
        # 必须包含 SELECT 和 FROM
        if 'select' not in text_lower:
            return False
        if 'from' not in text_lower and 'with' not in text_lower:
            return False
        # 不应该包含明显的非 SQL 内容
        non_sql_markers = ['**查询理解**', '**涉及表格**', '**查询说明**', '**注意事项**']
        for marker in non_sql_markers:
            if marker in text:
                return False
        return True

    def analyze_intent(self, query: str, schema_context: str) -> str:
        """
        分析用户意图：是查询数据 (SQL) 还是闲聊 (CHAT)。
        """
        try:
            prompt = f"""
            你是一个精准的意图分类器。
            
            【数据库上下文】: 
            {schema_context[:800]}... (已截断)
            
            【用户输入】: "{query}"
            
            【任务】:
            判断用户是否想要查询数据库中的数据、统计信息或业务指标。
            
            【输出规则】:
            1. 如果涉及数据查询、统计、分析 -> 输出 [SQL]
            2. 如果是打招呼、写代码、翻译、闲聊 -> 输出 [CHAT]
            3. 仅输出标签，不要任何解释。
            """
            
            resp = self.client.chat.completions.create(
                model=self.model_name, 
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1 # 低温度确保分类稳定
            )
            content = resp.choices[0].message.content.strip()
            
            if "[SQL]" in content: return "SQL"
            if "[CHAT]" in content: return "CHAT"
            return "CHAT" # 默认回退
            
        except Exception as e:
            self._write_log(f"意图分析失败: {e}")
            return "CHAT"

    def execute_sql(self, sql: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        执行 SQL 并返回 Pandas DataFrame。
        包含严格的安全检查和异常分类，并收集错误信息用于重试优化。
        """
        try:
            # 1. 预处理：清理 Markdown 标记
            clean_sql = sql.replace("```sql", "").replace("```", "").strip()
            clean_sql = clean_sql.rstrip(';')
            
            # 2. 安全检查：只允许查询相关语句（在数据库连接检查之前）
            if not self._is_safe_query(clean_sql):
                warn_msg = "⚠️ 安全拦截: 检测到非查询语句，已终止执行。"
                self._write_log(f"{warn_msg} SQL: {clean_sql}")
                
                # 收集安全错误信息
                error_info = self.error_context_manager.error_collector.capture_sql_error(
                    sql=clean_sql,
                    error_message="非查询语句被安全拦截",
                    context={"security_check": "failed", "sql_type": "non_query"}
                )
                error_info.category = ErrorCategory.SYNTAX
                error_info.severity = ErrorSeverity.HIGH
                self.error_context_manager.add_error(error_info)
                
                return None, warn_msg
            
            # 3. 数据库连接检查
            if not self.engine:
                error_msg = "数据库连接未初始化。"
                # 收集错误信息
                error_info = self.error_context_manager.error_collector.capture_execution_error(
                    command="database_connection",
                    output=error_msg,
                    exit_code=-1,
                    context={"operation": "sql_execution", "sql": clean_sql[:100]}
                )
                self.error_context_manager.add_error(error_info)
                return None, error_msg
                
            # 4. 执行查询
            start_t = time.perf_counter()
            with self.engine.connect() as conn:
                # 使用 SQLAlchemy text() 确保安全
                df = pd.read_sql(text(clean_sql), conn)
            
            exec_time = (time.perf_counter() - start_t) * 1000
            self._write_log(f"SQL 执行成功 (耗时 {exec_time:.2f}ms). 返回行数: {len(df)}")
            
            return df, None

        except ProgrammingError as e:
            # 通常是 SQL 语法错误或表名/列名不存在
            err_msg = f"SQL 语法错误: {e.orig}"
            self._write_log(err_msg)
            
            # 收集SQL语法错误信息
            error_info = self.error_context_manager.error_collector.capture_sql_error(
                sql=clean_sql,
                error_message=str(e.orig),
                context={
                    "error_type": "ProgrammingError",
                    "execution_time": time.perf_counter() - start_t if 'start_t' in locals() else 0
                }
            )
            error_info.category = ErrorCategory.SYNTAX
            error_info.severity = ErrorSeverity.HIGH
            self.error_context_manager.add_error(error_info)
            
            return None, err_msg
            
        except OperationalError as e:
            # 数据库连接中断或锁死
            err_msg = f"数据库操作错误 (连接/权限): {e.orig}"
            self._write_log(err_msg)
            
            # 收集数据库操作错误信息
            error_info = self.error_context_manager.error_collector.capture_sql_error(
                sql=clean_sql,
                error_message=str(e.orig),
                context={
                    "error_type": "OperationalError",
                    "database_uri": self.db_uris[0] if self.db_uris else "unknown"
                }
            )
            error_info.category = ErrorCategory.DATABASE
            error_info.severity = ErrorSeverity.HIGH
            self.error_context_manager.add_error(error_info)
            
            return None, err_msg
            
        except Exception as e:
            # 其他未知错误
            err_msg = f"执行发生未知错误: {str(e)}"
            self._write_log(err_msg)
            
            # 收集未知错误信息
            error_info = self.error_context_manager.error_collector.capture_exception(
                exception=e,
                context={
                    "sql": clean_sql,
                    "operation": "sql_execution",
                    "database_uri": self.db_uris[0] if self.db_uris else "unknown"
                }
            )
            self.error_context_manager.add_error(error_info)
            
            return None, err_msg

    def generate_and_execute_stream(
        self,
        query: str,
        history_context: List[Dict],
        cache_query_key: Optional[str] = None,
    ) -> Generator[Dict, None, None]:
        """
        【核心主流程】
        流式生成器，逐步输出：
        1. RAG 检索状态
        2. 意图分析
        3. 歧义分析 (Thinking)
        4. SQL 生成 (Coding)
        5. 执行结果 (Result)
        """
        self._write_log(f"========== 新对话任务启动: {query} ==========")

        # 查询级缓存：命中则直接返回，避免进入后续 LLM 调用链
        # cache_query_key 允许把“缓存键”与“实际推理query”解耦，
        # 以兼容上下文增强提示词和稳定缓存命中。
        cache_key = None
        if CACHE_AVAILABLE and universal_optimizer:
            try:
                db_id = self.db_uris[0] if self.db_uris else ""
                key_query = cache_query_key if cache_query_key is not None else query
                cache_key = universal_optimizer.generate_cache_key(key_query, "", self.model_name, db_id)
                cached_payload = universal_optimizer.get_cached_result(cache_key)
                if cached_payload:
                    cached_rows = cached_payload.get('rows', [])
                    cached_sql = cached_payload.get('sql', '')
                    cached_df = pd.DataFrame(cached_rows)

                    self._write_log(f"♻️ 查询缓存命中: rows={len(cached_df)}")
                    yield {"type": "step", "icon": "♻️", "msg": "命中查询缓存，跳过检索与生成", "status": "complete"}
                    yield {
                        "type": "cumulative_token_usage",
                        "usage": {
                            'prompt_tokens': 0,
                            'completion_tokens': 0,
                            'total_tokens': 0,
                            'call_count': 0,
                            'breakdown': []
                        }
                    }
                    yield {"type": "result", "df": cached_df, "sql": cached_sql, "from_cache": True}
                    return
            except Exception as e:
                self._write_log(f"⚠️ 查询缓存读取失败，继续常规流程: {e}")
        
        # 🆕 累加 Token 使用统计（跨所有 LLM 调用）
        # 统计模型：
        # - G1: T = T_generator (单次生成)
        # - G2: T = T_selector + T_generator (精排 + 生成)
        # - G3: T = T_selector + T_generator + Σ T_retry_i (精排 + 生成 + N次自愈)
        cumulative_token_usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'call_count': 0,  # LLM 调用次数
            'breakdown': []    # 每次调用的详细记录
        }
        
        # ---------------------------------------------------------
        # 阶段 1: 二阶段混合检索 (粗排 → LLM精排 → Schema获取)
        # 
        # 🔧 重构说明 (2026-02-08):
        # 之前存在两条并行的检索路径，导致 LLM Prompt 使用的是简单检索结果
        # 现在统一为: 粗排(Top-15) → LLM精排(3-5表) → 构建Prompt
        # ---------------------------------------------------------
        yield {"type": "step", "icon": "⚡", "msg": "正在调用 OpenVINO 二阶段检索...", "status": "running"}
        
        context = None
        retrieval_result = None
        rag_latency = 0
        query_complexity = 'medium'  # 默认复杂度
        selected_tables = None
        
        try:
            import time as time_module
            start_time = time_module.perf_counter()
            
            # 评估查询复杂度
            query_complexity = self._estimate_query_complexity(query)
            self._write_log(f"查询复杂度评估: {query_complexity}")
            
            # 🆕 统一使用二阶段检索（包含 LLM 精排）
            retrieval_result = self.retrieve_context_two_stage(query, enable_pruning=True)
            
            if retrieval_result:
                # 从二阶段检索结果中提取信息
                database_index = retrieval_result.get('database_index', [])
                core_tables = retrieval_result.get('core_tables', [])
                core_table_details = retrieval_result.get('core_table_details', [])
                matched_terms = retrieval_result.get('matched_terms', [])
                matched_examples = retrieval_result.get('matched_examples', [])
                sample_data = retrieval_result.get('sample_data', {})
                metrics = retrieval_result.get('metrics', {})
                
                # 🆕 捕获精排器的 Token 消耗，累加到总计
                selector_token_usage = metrics.get('selector_token_usage', {})
                if selector_token_usage:
                    cumulative_token_usage['prompt_tokens'] += selector_token_usage.get('prompt_tokens', 0)
                    cumulative_token_usage['completion_tokens'] += selector_token_usage.get('completion_tokens', 0)
                    cumulative_token_usage['total_tokens'] += selector_token_usage.get('total_tokens', 0)
                    cumulative_token_usage['call_count'] += 1
                    cumulative_token_usage['breakdown'].append({
                        'call_type': 'selector',
                        **selector_token_usage
                    })
                    self._write_log(f"精排器 Token: prompt={selector_token_usage.get('prompt_tokens', 0)}, completion={selector_token_usage.get('completion_tokens', 0)}")
                
                # 构建 context（按设计文档格式）
                context_parts = []
                
                # [1] Database Index 全局导航
                if database_index:
                    context_parts.append(f"=== 数据库表索引 (共 {len(database_index)} 张表) ===")
                    context_parts.append(", ".join(database_index[:20]))  # 限制展示数量
                    if len(database_index) > 20:
                        context_parts.append(f"... 及其他 {len(database_index) - 20} 张表")
                
                # [2] 核心表详细 Schema (精排后的 3-5 张表)
                if core_table_details:
                    context_parts.append(f"\n=== 核心表详情 (LLM精排: {len(core_tables)} 张) ===")
                    for detail in core_table_details:
                        table_doc = detail.get('document', '')
                        if table_doc:
                            context_parts.append(table_doc)
                        else:
                            # 如果没有预生成的文档，手动构建
                            table_name = detail.get('table_name', '')
                            columns = detail.get('columns', [])
                            context_parts.append(f"\n【表名】{table_name}")
                            context_parts.append(f"【描述】{detail.get('description', '')}")
                            if columns:
                                col_info = ", ".join([f"{c['name']}({c['type']})" for c in columns[:10]])
                                context_parts.append(f"【列】{col_info}")
                
                # [3] 样本数据（帮助 LLM 理解数据格式）
                if sample_data:
                    context_parts.append("\n=== 样本数据 ===")
                    for table_name, samples in sample_data.items():
                        if samples:
                            context_parts.append(f"表 {table_name} 示例: {samples[0]}")
                
                # [4] 业务术语（KECA 术语词典匹配）
                if matched_terms:
                    context_parts.append("\n=== 业务术语 ===")
                    for term in matched_terms:
                        term_name = term.get('term', '')
                        explanation = term.get('explanation', '')
                        context_parts.append(f"• {term_name}: {explanation}")
                
                # [5] Few-Shot 示例（KECA 示例匹配）
                if matched_examples:
                    context_parts.append("\n=== 参考示例 ===")
                    for ex in matched_examples[:2]:  # 限制 2 个示例
                        ex_q = ex.get('question', '')
                        ex_sql = ex.get('sql', '')
                        context_parts.append(f"问题: {ex_q}\nSQL: {ex_sql}")
                
                context = "\n".join(context_parts)
                
                # 计算延迟
                rag_latency = (time_module.perf_counter() - start_time) * 1000
                
                yield {
                    "type": "step",
                    "icon": "✅",
                    "msg": f"二阶段检索完成: {len(core_tables)} 核心表 (精排), {len(matched_terms)} 术语 (耗时 {rag_latency:.0f}ms)",
                    "status": "complete",
                    "rag_latency": rag_latency
                }
                self._write_log(f"二阶段检索完成: 精排核心表={core_tables}, 耗时={rag_latency:.2f}ms")
            else:
                # 兜底: 如果二阶段检索失败，使用简单检索
                self._write_log("⚠️ 二阶段检索返回空，回退到简单检索")
                rag_context, rag_latency, rag_mem = self.rag.retrieve(query, top_k=10)
                table_map = self._get_table_map()
                if table_map and rag_context:
                    context = f"{table_map}\n=== 相关表详情 ===\n{rag_context}"
                else:
                    context = rag_context
                    
                yield {
                    "type": "step",
                    "icon": "⚠️",
                    "msg": f"回退简单检索 (耗时 {rag_latency:.0f}ms)",
                    "status": "complete",
                    "rag_latency": rag_latency
                }
                
        except Exception as e:
            err_msg = f"RAG 检索模块故障: {str(e)}"
            self._write_log(err_msg)
            import traceback
            self._write_log(traceback.format_exc())
            yield {"type": "error", "msg": err_msg}
            return

        # ---------------------------------------------------------
        # 阶段 2: 意图识别
        # ---------------------------------------------------------
        yield {"type": "step", "icon": "🧠", "msg": "正在分析用户意图...", "status": "running"}
        intent = self.analyze_intent(query, context)
        
        if intent == "CHAT":
            self._write_log("意图识别结果: CHAT (非数据库查询)")
            yield {"type": "final_chat"}
            return

        yield {"type": "step", "icon": "🔍", "msg": "意图确认: 数据查询请求", "status": "complete"}

        # ---------------------------------------------------------
        # 阶段 2.5: KECA 知识增强（统一流程，无复杂度分支）
        # 所有查询均经过完整流程：粗排→精排→KECA增强→Prompt构建
        # ---------------------------------------------------------
        yield {"type": "step", "icon": "🧠", "msg": "KECA 知识增强...", "status": "running"}
        
        try:
            # context 已在阶段 1 通过 retrieve_context_two_stage() 构建完成，
            # 包含：database_index + core_table_details + sample_data + matched_terms + matched_examples
            # 此阶段仅做日志记录和状态更新
            
            if context:
                # 统计 context 中的知识增强内容
                term_count = context.count('业务术语') if '业务术语' in context else 0
                example_count = context.count('参考示例') if '参考示例' in context else 0
                
                yield {"type": "step", "icon": "✅", "msg": f"KECA 增强完成: 已注入术语词典和参考示例", "status": "complete"}
                self._write_log(f"KECA知识增强完成: context长度={len(context)}")
            else:
                yield {"type": "step", "icon": "⚠️", "msg": "无 KECA 上下文，使用基础 Schema", "status": "complete"}
                self._write_log("KECA知识增强: 无上下文")
                
        except Exception as e:
            self._write_log(f"KECA增强阶段出错: {e}")
            yield {"type": "error_log", "content": f"KECA增强失败: {str(e)}"}
            # 继续执行，不中断流程

        # ---------------------------------------------------------
        # 阶段 3: 基于用户配置的可能性枚举
        # ---------------------------------------------------------
        possibilities = []
        selected_possibility = None
        
        if self.max_candidates > 1:
            yield {"type": "step", "icon": "🎯", "msg": f"正在生成 {self.max_candidates} 种可能的查询理解...", "status": "running"}
            yield {"type": "thought_start"}
            
            try:
                # 生成多种可能的理解方式
                possibilities = self.possibility_generator.generate_possibilities(
                    query=query, 
                    context=context, 
                    max_count=self.max_candidates
                )
                
                # 流式输出思考过程
                for possibility in possibilities:
                    thought_content = f"理解方式 {possibility.rank}: {possibility.natural_description or possibility.description}\n"
                    thought_content += f"置信度: {possibility.confidence:.1%}\n"
                    if possibility.key_interpretations:
                        interpretations_text = " | ".join([
                            f"{term}: {interp['desc']}" 
                            for term, interp in possibility.key_interpretations.items()
                        ])
                        thought_content += f"技术解释: {interpretations_text}\n"
                    thought_content += "\n"
                    yield {"type": "thought_chunk", "content": thought_content}
                
                # 选择最佳理解
                selected_possibility = possibilities[0] if possibilities else None
                best_interpretation = selected_possibility.description if selected_possibility else query
                
                yield {"type": "step", "icon": "✅", "msg": f"已生成 {len(possibilities)} 种理解方式", "status": "complete"}
                self._write_log(f"可能性枚举完成. 最佳理解: {best_interpretation}")
                
            except Exception as e:
                self._write_log(f"可能性枚举阶段出错: {e}")
                yield {"type": "error_log", "content": f"可能性分析失败: {str(e)}"}
                best_interpretation = query
        else:
            # 用户只要1种可能性，直接使用原始查询
            best_interpretation = query
            yield {"type": "step", "icon": "🔍", "msg": "使用标准理解方式", "status": "complete"}

        # ---------------------------------------------------------
        # 阶段 4: 智能SQL生成与执行
        # ---------------------------------------------------------
        # 如果有多种可能性，按置信度顺序尝试执行
        if possibilities and len(possibilities) > 1:
            yield {"type": "step", "icon": "🚀", "msg": "按置信度顺序尝试执行查询...", "status": "running"}
            
            for i, possibility in enumerate(possibilities):
                try:
                    # 为当前可能性生成SQL
                    sql = self.generate_sql_for_possibility(possibility, context, query)
                    df, err = self.execute_sql(sql)
                    
                    if df is not None and not df.empty:
                        # 成功执行，返回结果和备选方案
                        alternatives = [p for p in possibilities if p != possibility]
                        yield {"type": "step", "icon": "🎉", "msg": f"理解方式 {possibility.rank} 执行成功！获取 {len(df)} 条记录", "status": "complete"}
                        yield {
                            "type": "result", 
                            "df": df, 
                            "sql": sql,
                            "from_cache": False,
                            "selected_possibility": possibility,
                            "alternatives": alternatives
                        }

                        # 写入缓存（查询级）
                        if cache_key and CACHE_AVAILABLE and universal_optimizer:
                            try:
                                universal_optimizer.set_cached_result(
                                    cache_key,
                                    {
                                        'rows': df.to_dict('records'),
                                        'sql': sql,
                                        'cached_at': time.time()
                                    }
                                )
                            except Exception as e:
                                self._write_log(f"⚠️ 查询缓存写入失败: {e}")
                        return
                    elif i == 0:
                        # 最佳理解也失败了，记录错误继续尝试
                        yield {"type": "error_log", "content": f"理解方式 {possibility.rank} 执行失败: {err or '结果为空'}"}
                        
                except Exception as e:
                    if i == 0:
                        yield {"type": "error_log", "content": f"理解方式 {possibility.rank} 生成失败: {str(e)}"}
                    continue
            
            # 所有可能性都失败，回退到传统重试机制
            yield {"type": "step", "icon": "🔄", "msg": "所有理解方式都失败，回退到传统重试机制...", "status": "complete"}

        # ---------------------------------------------------------
        # 阶段 5: 传统SQL生成与执行 (ReAct Loop) - 集成错误上下文
        # ---------------------------------------------------------
        last_error = None
        
        # 在开始重试前清空错误历史（针对当前查询）
        self.error_context_manager.clear_history()
        
        for i in range(max(1, self.max_retries)):  # 确保至少执行 1 次 SQL 生成
            current_try = i + 1
            status_msg = f"正在构建 SQL 查询 (第 {current_try} 次尝试)..."
            yield {"type": "step", "icon": "💻", "msg": status_msg, "status": "running"}
            
            # 构建基础 SQL 生成提示词
            if self.prompt_builder:
                # 🧠 使用增强的Prompt模板系统
                
                # 安全获取 Prompt 模式（Streamlit 导入失败不会导致整体降级）
                prompt_mode_str = 'flexible'  # 默认值
                try:
                    import streamlit as st
                    if hasattr(st, 'session_state'):
                        prompt_mode_str = st.session_state.get('prompt_mode', 'flexible')
                except Exception:
                    pass  # 非 Streamlit 环境，使用默认值
                
                try:
                    from prompt_template_system import PromptMode
                    prompt_mode = PromptMode.PROFESSIONAL if prompt_mode_str == 'professional' else PromptMode.FLEXIBLE
                    
                    # 构建重试上下文（安全处理）
                    retry_context = None
                    if current_try > 1:
                        try:
                            retry_context = self.error_context_manager.get_retry_context(max_errors=3)
                        except Exception as ctx_err:
                            self._write_log(f"⚠️ 获取重试上下文失败: {ctx_err}")
                    
                    # 使用增强的Prompt构建器
                    enhanced_prompt = self.prompt_builder.build_sql_generation_prompt(
                        user_query=query,
                        schema_info=context,
                        rag_context="",  # RAG上下文已经包含在context中
                        selected_tables=selected_tables if 'selected_tables' in locals() else None,
                        query_possibilities=possibilities if 'possibilities' in locals() else None,
                        retry_context=retry_context.to_dict() if retry_context else None,
                        mode=prompt_mode
                    )
                    
                    sys_prompt = enhanced_prompt
                    self._write_log(f"✅ 使用增强Prompt模板系统 (模式: {prompt_mode_str})")
                    
                except Exception as e:
                    self._write_log(f"⚠️ Prompt模板系统失败，回退到传统方式: {e}")
                    import traceback
                    self._write_log(f"详细错误: {traceback.format_exc()}")
                    # 回退到传统Prompt构建
                    sys_prompt = self._build_traditional_prompt(query, context, best_interpretation, current_try)
            else:
                # 传统Prompt构建方式
                sys_prompt = self._build_traditional_prompt(query, context, best_interpretation, current_try)

            yield {"type": "code_start", "label": f"Generated SQL (v{current_try})"}
            
            full_sql_buffer = ""
            
            try:
                # ⭐ 根据重试次数动态选择模型 (自愈模式可启用 Reasoner)
                current_model = self._get_model_for_attempt(current_try)
                
                stream = self.client.chat.completions.create(
                    model=current_model, 
                    messages=[{"role":"user","content":sys_prompt}], 
                    stream=True,
                    temperature=self.temperature,
                    # 🆕 启用流式 Token 统计 (DeepSeek/OpenAI 兼容)
                    stream_options={"include_usage": True}
                )
                
                # 🆕 Token 使用统计（DeepSeek 在最后一个 chunk 返回 usage）
                token_usage = None
                
                for chunk in stream:
                    # 🔍 Debug: 检查每个 chunk 的结构
                    has_usage_attr = hasattr(chunk, 'usage')
                    usage_value = getattr(chunk, 'usage', None) if has_usage_attr else None
                    has_choices = bool(chunk.choices) if hasattr(chunk, 'choices') else False
                    
                    # 捕获 Token 使用信息（在最后一个 chunk 中）
                    if has_usage_attr and usage_value is not None:
                        self._write_log(f"📊 [Token] 检测到 usage chunk: {usage_value}")
                        token_usage = {
                            'prompt_tokens': getattr(usage_value, 'prompt_tokens', 0),
                            'completion_tokens': getattr(usage_value, 'completion_tokens', 0),
                            'total_tokens': getattr(usage_value, 'total_tokens', 0)
                        }
                        self._write_log(f"📊 [Token] 解析结果: {token_usage}")
                    
                    if has_choices and chunk.choices[0].delta:
                        delta = chunk.choices[0].delta
                        if delta.content:
                            full_sql_buffer += delta.content
                            yield {"type": "code_chunk", "content": delta.content}
                
                self._write_log(f"SQL 生成 (v{current_try}): {full_sql_buffer}")
                
                # 🆕 累加 Token 使用统计到 cumulative_token_usage
                if token_usage:
                    cumulative_token_usage['prompt_tokens'] += token_usage['prompt_tokens']
                    cumulative_token_usage['completion_tokens'] += token_usage['completion_tokens']
                    cumulative_token_usage['total_tokens'] += token_usage['total_tokens']
                    cumulative_token_usage['call_count'] += 1
                    cumulative_token_usage['breakdown'].append({
                        'call_type': 'generator',
                        'attempt': current_try,
                        **token_usage
                    })
                    
                    self._write_log(f"Token 使用 (本次): prompt={token_usage['prompt_tokens']}, completion={token_usage['completion_tokens']}, total={token_usage['total_tokens']}")
                    self._write_log(f"Token 累计: prompt={cumulative_token_usage['prompt_tokens']}, completion={cumulative_token_usage['completion_tokens']}, total={cumulative_token_usage['total_tokens']}")
                    
                    # 仍然 yield 每次调用的 token_usage（保持兼容性）
                    yield {"type": "token_usage", "usage": token_usage}

                # --- 执行 SQL ---
                yield {"type": "step", "icon": "⚡", "msg": "正在提交至数据库引擎...", "status": "running"}
                
                # 🔧 使用智能 SQL 提取器处理 LLM 响应
                extracted_sql = self._extract_sql_from_response(full_sql_buffer)
                
                df, err = self.execute_sql(extracted_sql)
                
                # 情况 A: SQL 报错
                if err:
                    last_error = err
                    yield {"type": "error_log", "content": f"执行错误: {err}"}
                    
                    # 🔧 动态自愈机制：扩展错误检测 + 差异化重试 (P1 + P3)
                    # ⭐ 只有当 max_retries > 0 时才启用自愈机制（G1/G2禁用，G3启用）
                    if self.max_retries > 0:
                        healing_applied, healed_context = self._apply_self_healing(err, context, retry_attempt=current_try)
                        if healing_applied:
                            # 更新 context 供下次重试使用
                            context = healed_context
                            yield {"type": "step", "icon": "🔧", "msg": f"自愈机制已触发 (尝试 {current_try})，正在重试...", "status": "running"}
                    
                    # 错误信息已经在execute_sql中收集，这里不需要重复收集
                    continue  # 重试
                
                # 情况 B: 结果为空
                if df.empty:
                    # 如果不是最后一次尝试，则继续重试优化
                    if current_try < self.max_retries:
                        empty_result_msg = "SQL 语法正确但返回结果为空 (0 rows)。请检查 WHERE 条件（如日期格式、大小写）是否过于严格。"
                        last_error = empty_result_msg
                        
                        # 收集空结果错误信息
                        error_info = self.error_context_manager.error_collector.capture_sql_error(
                            sql=full_sql_buffer,
                            error_message="查询结果为空",
                            context={
                                "result_count": 0,
                                "retry_attempt": current_try,
                                "query_type": "empty_result"
                            }
                        )
                        error_info.category = ErrorCategory.LOGIC
                        error_info.severity = ErrorSeverity.MEDIUM
                        self.error_context_manager.add_error(error_info)
                        
                        yield {"type": "step", "icon": "⚠️", "msg": "查询结果为空，正在进行逻辑自愈...", "status": "complete"}
                        continue
                    else:
                        # 次数用尽，虽然为空但也是一种结果
                        yield {"type": "step", "icon": "🏁", "msg": "查询完成 (无数据匹配)", "status": "complete"}
                        # 🆕 yield 累计 Token 统计（所有 LLM 调用的总和）
                        yield {"type": "cumulative_token_usage", "usage": cumulative_token_usage}
                        yield {"type": "result", "df": df, "sql": full_sql_buffer, "from_cache": False}
                        return
                
                # 情况 C: 成功获取数据
                yield {"type": "step", "icon": "🎉", "msg": f"查询成功！获取 {len(df)} 条记录", "status": "complete"}
                # 🆕 yield 累计 Token 统计（所有 LLM 调用的总和）
                yield {"type": "cumulative_token_usage", "usage": cumulative_token_usage}
                yield {"type": "result", "df": df, "sql": full_sql_buffer, "from_cache": False}

                # 写入缓存（查询级）
                if cache_key and CACHE_AVAILABLE and universal_optimizer:
                    try:
                        universal_optimizer.set_cached_result(
                            cache_key,
                            {
                                'rows': df.to_dict('records'),
                                'sql': full_sql_buffer,
                                'cached_at': time.time()
                            }
                        )
                    except Exception as e:
                        self._write_log(f"⚠️ 查询缓存写入失败: {e}")
                return

            except Exception as e:
                # 收集SQL生成过程中的错误
                error_info = self.error_context_manager.error_collector.capture_exception(
                    exception=e,
                    context={
                        "operation": "sql_generation",
                        "retry_attempt": current_try,
                        "partial_sql": full_sql_buffer[:200] if full_sql_buffer else ""
                    }
                )
                self.error_context_manager.add_error(error_info)
                
                yield {"type": "error", "msg": f"生成过程发生致命错误: {str(e)}"}
                return

        # 循环结束仍未成功 - 提供详细的失败报告
        error_summary = self.error_context_manager.get_error_summary()
        failure_report = f"已达到最大重试次数 ({self.max_retries})，无法生成有效查询。\n"
        failure_report += f"总错误数: {error_summary['total_errors']}\n"
        
        if error_summary.get('categories'):
            failure_report += f"主要错误类型: {', '.join(error_summary['categories'].keys())}\n"
        
        # 获取最终的修复建议
        final_retry_context = self.error_context_manager.get_retry_context()
        if final_retry_context.suggestions:
            failure_report += f"建议: {'; '.join(final_retry_context.suggestions[:3])}"
        
        yield {"type": "error", "msg": failure_report}


    def generate_sql_for_possibility(self, possibility: QueryPossibility, context: str, original_query: str) -> str:
        """
        为特定的查询可能性生成SQL语句（自动适配数据库类型）
        
        :param possibility: 查询可能性对象
        :param context: Schema上下文信息
        :param original_query: 用户原始查询
        :return: 生成的SQL语句
        """
        try:
            # 获取数据库类型信息
            db_info = self._get_db_type_info()
            syntax_tips_str = "\n".join([f"            - {tip}" for tip in db_info["syntax_tips"]])
            
            # 构建针对特定理解方式的提示词
            sys_prompt = f"""
            你是一个{db_info["db_expert"]}。
            
            【数据库类型】: {db_info["db_type"]}
            
            【Schema 信息】:
            {context}
            
            【用户原始问题】: "{original_query}"
            【确定的理解方式】: "{possibility.description}"
            【置信度】: {possibility.confidence:.1%}
            
            【关键解释】:
            """
            
            # 添加关键解释信息
            if possibility.key_interpretations:
                for term, interpretation in possibility.key_interpretations.items():
                    sys_prompt += f"\n- {term}: {interpretation['desc']}"
                    if 'sql_condition' in interpretation:
                        sys_prompt += f" (SQL条件: {interpretation['sql_condition']})"
                    if 'sql_expression' in interpretation:
                        sys_prompt += f" (SQL表达式: {interpretation['sql_expression']})"
                    if 'sql_pattern' in interpretation:
                        sys_prompt += f" (SQL模式: {interpretation['sql_pattern']})"
            
            sys_prompt += f"""
            
            【任务】:
            根据上述确定的理解方式，编写精确的 {db_info["db_type"]} SQL 语句。
            
            【严格约束】:
            1. 仅输出 SQL 代码，不要任何解释。
            2. 不要使用 Markdown 格式 (不要写 ```sql)。
            3. {db_info["date_hint"]}。
            4. 严格按照关键解释中的SQL条件、表达式和模式来构建查询。
            5. 确保 {db_info["db_type"]} SQL语法正确且可执行。
            
            【{db_info["db_type"]} 语法要点】:
{syntax_tips_str}
            """
            
            # 调用LLM生成SQL
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": sys_prompt}],
                temperature=0.1  # 低温度确保生成稳定
            )
            
            generated_sql = resp.choices[0].message.content.strip()
            
            # 清理生成的SQL
            clean_sql = generated_sql.replace("```sql", "").replace("```", "").strip()
            clean_sql = clean_sql.rstrip(';')
            
            self._write_log(f"为理解方式 {possibility.rank} 生成SQL: {clean_sql}")
            
            return clean_sql
            
        except Exception as e:
            self._write_log(f"为理解方式 {possibility.rank} 生成SQL失败: {e}")
            
            # 收集SQL生成错误
            error_info = self.error_context_manager.error_collector.capture_exception(
                exception=e,
                context={
                    "operation": "possibility_sql_generation",
                    "possibility_rank": possibility.rank,
                    "possibility_description": possibility.description,
                    "original_query": original_query
                }
            )
            self.error_context_manager.add_error(error_info)
            
            raise e

    def generate_insight_stream(self, query: str, df: pd.DataFrame) -> Generator[str, None, None]:
        """
        基于数据生成商业洞察。
        """
        if df is None or df.empty:
            yield "⚠️ **未查询到有效数据**，无法生成商业洞察。"
            return

        try:
            # 截取前 10 行数据以节省 Token
            data_preview = df.head(10).to_markdown(index=False)
            
            prompt = f"""
            你是一位资深商业数据分析师。
            
            【用户问题】: "{query}"
            【查询到的数据 (前10行)】:
            {data_preview}
            
            【任务】:
            根据数据回答用户问题，并给出一句简短的商业洞察或建议。
            
            【要求】:
            1. 基于事实，严谨客观。
            2. 语言简练，控制在 80 字以内。
            3. 不要重复数据的具体数值，而是总结趋势或异常点。
            """
            
            stream = self.client.chat.completions.create(
                model=self.model_name, # 这里使用 R1 或 V3 均可
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            
            for chunk in stream:
                delta = chunk.choices[0].delta
                # 兼容 R1 的 reasoning_content (虽然洞察阶段通常不需要展示思考)
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                     yield delta.reasoning_content
                if delta.content:
                    yield delta.content
                    
        except Exception as e:
            self._write_log(f"洞察生成失败: {e}")
            yield f"生成洞察时发生错误: {str(e)}"

    def chat_stream(self, query: str, history_context: List[Dict]) -> Generator[str, None, None]:
        """
        处理非 SQL 的闲聊请求。
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": query}],
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"连接错误: {str(e)}"
    
    def reset_error_context(self):
        """重置错误上下文，用于新的查询会话"""
        self.error_context_manager.clear_history()
        self._write_log("错误上下文已重置")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计信息，用于调试和监控"""
        return self.error_context_manager.get_error_summary()
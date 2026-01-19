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
from table_selector import IntelligentTableSelector

# 🧠 集成Prompt模板系统
try:
    from prompt_integration import EnhancedPromptBuilder
    PROMPT_TEMPLATE_AVAILABLE = True
except ImportError:
    PROMPT_TEMPLATE_AVAILABLE = False
    print("⚠️ Prompt模板系统不可用，使用传统Prompt构建")

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
        config: Optional[Dict[str, Any]] = None
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
        self.config = config or {}
        
        # 3. 🧠 初始化Prompt模板系统
        if PROMPT_TEMPLATE_AVAILABLE:
            try:
                self.prompt_builder = EnhancedPromptBuilder(self.config)
                self._write_log("✅ Prompt模板系统初始化成功")
            except Exception as e:
                self.prompt_builder = None
                self._write_log(f"⚠️ Prompt模板系统初始化失败: {e}")
        else:
            self.prompt_builder = None
        
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
        
        # 5. 新增：智能表选择器（传入RAG引擎用于向量计算）
        # 注意：这里先初始化为None，在RAG引擎创建后再初始化
        self.table_selector = None
        
        # 6. 新增：错误上下文重试机制
        self.error_context_manager = ErrorContextManager(max_history=10)
        self.prompt_enhancer = PromptEnhancer(max_context_length=1000)
        
        # 7. 日志与文件系统准备
        self._setup_logging()
        
        # 8. 数据库引擎初始化
        self.engine = None
        self._init_db_connection()
        
        # 9. 初始化表选择器（在RAG引擎传入后）
        self._init_table_selector()
    
    def _init_table_selector(self):
        """初始化表选择器，使用RAG引擎进行语义匹配"""
        try:
            # 从RAG引擎的知识库路径中提取schema文件
            schema_paths = []
            if hasattr(self.rag, 'kb_paths') and self.rag.kb_paths:
                schema_paths = [path for path in self.rag.kb_paths if path.endswith('.json')]
            
            # 初始化表选择器
            self.table_selector = IntelligentTableSelector(
                rag_engine=self.rag,
                schema_paths=schema_paths
            )
            
            self._write_log(f"表选择器初始化成功，加载了 {len(schema_paths)} 个schema文件")
            
        except Exception as e:
            self._write_log(f"表选择器初始化失败: {e}")
            # 创建一个基础的表选择器作为备用
            self.table_selector = IntelligentTableSelector()

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

    def _build_traditional_prompt(self, query: str, context: str, best_interpretation: str, current_try: int) -> str:
        """构建传统的SQL生成Prompt"""
        base_prompt = f"""
        你是一个精通 SQLite 的高级数据库工程师。
        
        【Schema 信息】:
        {context}
        
        【用户原始问题】: "{query}"
        【已确认的业务逻辑】: "{best_interpretation}"
        
        【任务】:
        编写可执行的 SQL 语句。
        
        【严格约束】:
        1. 仅输出 SQL 代码。
        2. 不要使用 Markdown 格式 (不要写 ```sql)。
        3. 日期处理请使用 `strftime` 函数，例如 `strftime('%Y', order_date) = '2016'`。
        4. 不要解释代码。
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

    def generate_and_execute_stream(self, query: str, history_context: List[Dict]) -> Generator[Dict, None, None]:
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
        
        # ---------------------------------------------------------
        # 阶段 1: OpenVINO RAG 检索
        # ---------------------------------------------------------
        yield {"type": "step", "icon": "⚡", "msg": "正在调用 OpenVINO 进行向量检索...", "status": "running"}
        
        try:
            # 调用 rag_engine 的 retrieve 方法 (返回: 文本, 延迟, 内存)
            context, rag_latency, rag_mem = self.rag.retrieve(query)
            
            perf_info = f"耗时 {rag_latency:.2f}ms | 内存 +{rag_mem:.2f}MB"
            self._write_log(f"RAG 检索完成. {perf_info}")
            
            yield {
                "type": "step", 
                "icon": "✅", 
                "msg": f"OpenVINO 检索完成 ({perf_info})", 
                "status": "complete",
                "rag_latency": rag_latency 
            }
        except Exception as e:
            err_msg = f"RAG 检索模块故障: {str(e)}"
            self._write_log(err_msg)
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
        # 阶段 2.5: 智能表选择
        # ---------------------------------------------------------
        yield {"type": "step", "icon": "🗄️", "msg": "正在智能分析相关数据表...", "status": "running"}
        
        try:
            # 使用智能表选择器分析查询
            if self.table_selector:
                # 第一步：初步表选择
                yield {"type": "step", "icon": "🔍", "msg": "第一步：基于语义相似度初步筛选表...", "status": "running"}
                
                selected_tables, table_analysis = self.table_selector.select_tables(query, top_k=8)  # 先选择更多表
                
                if selected_tables:
                    # 显示初步筛选结果
                    initial_tables_info = f"初步筛选出 {len(selected_tables)} 个候选表：\n"
                    for i, table_rel in enumerate(selected_tables[:5], 1):
                        initial_tables_info += f"{i}. {table_rel.table_name} (相关性: {table_rel.relevance_score:.1f})\n"
                    if len(selected_tables) > 5:
                        initial_tables_info += f"... 还有 {len(selected_tables) - 5} 个表\n"
                    
                    yield {"type": "table_analysis", "content": initial_tables_info}
                    
                    # 第二步：Agent智能二次筛选
                    yield {"type": "step", "icon": "🤖", "msg": "第二步：Agent基于语义和表结构进行智能筛选...", "status": "running"}
                    
                    # 调用Agent进行二次筛选
                    final_tables, reasoning = self._agent_table_refinement(query, selected_tables, context)
                    
                    # 显示Agent筛选推理过程
                    yield {"type": "agent_reasoning", "content": reasoning}
                    
                    # 第三步：表关联分析
                    if len(final_tables) > 1:
                        yield {"type": "step", "icon": "🔗", "msg": "第三步：分析表关联关系...", "status": "running"}
                        
                        join_analysis = self._analyze_table_relationships(final_tables)
                        yield {"type": "join_analysis", "content": join_analysis}
                    
                    # 输出最终表选择结果
                    yield {
                        "type": "table_selection",
                        "selected_tables": final_tables,
                        "analysis": table_analysis,
                        "table_context": self.table_selector.get_table_context(final_tables)
                    }
                    
                    # 更新context以包含选中的表信息
                    table_info = f"\n\n=== 智能表选择结果 ===\n"
                    table_info += f"Agent筛选推理: {reasoning[:200]}...\n" if len(reasoning) > 200 else f"Agent筛选推理: {reasoning}\n"
                    table_info += f"最终选择表数量: {len(final_tables)}\n"
                    
                    # 显示语义匹配信息
                    if table_analysis.get('use_semantic_matching'):
                        table_info += "使用OpenVINO语义匹配: ✅\n"
                    
                    table_info += "\n"
                    
                    for i, table_rel in enumerate(final_tables[:3], 1):  # 只显示前3个最相关的表
                        table_info += f"{i}. 表名: {table_rel.table_name} (相关性: {table_rel.relevance_score:.1f})\n"
                        table_info += f"   推理: {table_rel.reasoning}\n"
                        
                        # 显示语义相似度
                        if hasattr(table_rel, 'semantic_similarity') and table_rel.semantic_similarity > 0:
                            table_info += f"   语义相似度: {table_rel.semantic_similarity:.2f}\n"
                        
                        if table_rel.keyword_matches:
                            table_info += f"   匹配关键词: {', '.join(table_rel.keyword_matches[:3])}\n"
                        table_info += "\n"
                    
                    context = context + table_info
                    
                    yield {"type": "step", "icon": "✅", "msg": f"表选择完成，最终选择 {len(final_tables)} 个表", "status": "complete"}
                    self._write_log(f"表选择完成. 最终选择表: {[t.table_name for t in final_tables]}")
                else:
                    yield {"type": "step", "icon": "⚠️", "msg": "未找到明确相关的表，使用全部表信息", "status": "complete"}
            else:
                yield {"type": "step", "icon": "⚠️", "msg": "表选择器未初始化，跳过表选择", "status": "complete"}
                
        except Exception as e:
            self._write_log(f"表选择阶段出错: {e}")
            yield {"type": "error_log", "content": f"表选择失败: {str(e)}，将使用全部表信息"}
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
                            "selected_possibility": possibility,
                            "alternatives": alternatives
                        }
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
        
        for i in range(self.max_retries):
            current_try = i + 1
            status_msg = f"正在构建 SQL 查询 (第 {current_try} 次尝试)..."
            yield {"type": "step", "icon": "💻", "msg": status_msg, "status": "running"}
            
            # 构建基础 SQL 生成提示词
            if self.prompt_builder:
                # 🧠 使用增强的Prompt模板系统
                try:
                    # 获取当前的Prompt模式
                    import streamlit as st
                    prompt_mode_str = st.session_state.get('prompt_mode', 'flexible') if 'streamlit' in globals() else 'flexible'
                    
                    from prompt_template_system import PromptMode
                    prompt_mode = PromptMode.PROFESSIONAL if prompt_mode_str == 'professional' else PromptMode.FLEXIBLE
                    
                    # 构建重试上下文
                    retry_context = None
                    if current_try > 1:
                        retry_context = self.error_context_manager.get_retry_context(max_errors=3)
                    
                    # 使用增强的Prompt构建器
                    enhanced_prompt = self.prompt_builder.build_sql_generation_prompt(
                        user_query=query,
                        schema_info=context,
                        rag_context="",  # RAG上下文已经包含在context中
                        selected_tables=selected_tables if 'selected_tables' in locals() else None,
                        query_possibilities=possibilities if 'possibilities' in locals() else None,
                        retry_context=retry_context.__dict__ if retry_context else None,
                        mode=prompt_mode
                    )
                    
                    sys_prompt = enhanced_prompt
                    self._write_log(f"✅ 使用增强Prompt模板系统 (模式: {prompt_mode_str})")
                    
                except Exception as e:
                    self._write_log(f"⚠️ Prompt模板系统失败，回退到传统方式: {e}")
                    # 回退到传统Prompt构建
                    sys_prompt = self._build_traditional_prompt(query, context, best_interpretation, current_try)
            else:
                # 传统Prompt构建方式
                sys_prompt = self._build_traditional_prompt(query, context, best_interpretation, current_try)

            yield {"type": "code_start", "label": f"Generated SQL (v{current_try})"}
            
            full_sql_buffer = ""
            
            try:
                stream = self.client.chat.completions.create(
                    model=self.model_name, 
                    messages=[{"role":"user","content":sys_prompt}], 
                    stream=True
                )
                
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        full_sql_buffer += delta.content
                        yield {"type": "code_chunk", "content": delta.content}
                
                self._write_log(f"SQL 生成 (v{current_try}): {full_sql_buffer}")

                # --- 执行 SQL ---
                yield {"type": "step", "icon": "⚡", "msg": "正在提交至数据库引擎...", "status": "running"}
                df, err = self.execute_sql(full_sql_buffer)
                
                # 情况 A: SQL 报错
                if err:
                    last_error = err
                    yield {"type": "error_log", "content": f"执行错误: {err}"}
                    
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
                        yield {"type": "result", "df": df, "sql": full_sql_buffer}
                        return
                
                # 情况 C: 成功获取数据
                yield {"type": "step", "icon": "🎉", "msg": f"查询成功！获取 {len(df)} 条记录", "status": "complete"}
                yield {"type": "result", "df": df, "sql": full_sql_buffer}
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

    def _agent_table_refinement(self, query: str, selected_tables: List, context: str) -> Tuple[List, str]:
        """
        使用Agent进行智能二次筛选，基于语义和表结构分析
        
        :param query: 用户查询
        :param selected_tables: 初步筛选的表列表
        :param context: Schema上下文信息
        :return: (最终选择的表列表, 推理过程说明)
        """
        try:
            if not selected_tables:
                return [], "未找到相关表"
            
            # 如果只有1-2个表，直接返回
            if len(selected_tables) <= 2:
                reasoning = f"初步筛选结果良好，直接选择 {len(selected_tables)} 个最相关的表"
                return selected_tables, reasoning
            
            # 构建Agent分析提示词
            tables_info = ""
            for i, table_rel in enumerate(selected_tables, 1):
                tables_info += f"{i}. 表名: {table_rel.table_name}\n"
                tables_info += f"   描述: {table_rel.table_description}\n"
                tables_info += f"   相关性得分: {table_rel.relevance_score:.1f}\n"
                tables_info += f"   推理: {table_rel.reasoning}\n"
                
                # 显示相关字段
                if hasattr(table_rel, 'matched_columns') and table_rel.matched_columns:
                    tables_info += f"   相关字段: "
                    col_names = [col.get('col', '') for col in table_rel.matched_columns[:3]]
                    tables_info += ", ".join(col_names) + "\n"
                
                tables_info += "\n"
            
            prompt = f"""
            你是一个数据库专家，需要为用户查询选择最合适的数据表。
            
            【用户查询】: "{query}"
            
            【候选数据表】:
            {tables_info}
            
            【任务】:
            从上述候选表中选择最适合回答用户查询的表（建议选择2-4个表）。
            
            【分析要求】:
            1. 考虑表的相关性得分和语义匹配度
            2. 考虑查询所需的数据类型和业务逻辑
            3. 避免选择冗余或不相关的表
            4. 如果需要关联查询，选择有关联关系的表
            
            【输出格式】:
            选择的表: [表名1, 表名2, ...]
            推理过程: [详细说明选择这些表的原因，包括每个表的作用和为什么排除其他表]
            
            请严格按照上述格式输出，不要包含其他内容。
            """
            
            # 调用LLM进行分析
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1  # 低温度确保分析稳定
            )
            
            analysis_result = resp.choices[0].message.content.strip()
            
            # 解析LLM的回复
            selected_table_names = []
            reasoning = ""
            
            lines = analysis_result.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('选择的表:') or line.startswith('Selected tables:'):
                    # 提取表名
                    table_part = line.split(':', 1)[1].strip()
                    # 移除方括号并分割
                    table_part = table_part.strip('[]')
                    if table_part:
                        selected_table_names = [name.strip().strip(',') for name in table_part.split(',')]
                elif line.startswith('推理过程:') or line.startswith('Reasoning:'):
                    reasoning = line.split(':', 1)[1].strip()
                elif reasoning and line:  # 继续推理过程的内容
                    reasoning += " " + line
            
            # 根据LLM选择的表名筛选原始表对象
            final_tables = []
            for table_rel in selected_tables:
                if table_rel.table_name in selected_table_names:
                    final_tables.append(table_rel)
            
            # 如果LLM没有正确选择，回退到前3个最相关的表
            if not final_tables:
                final_tables = selected_tables[:3]
                reasoning = f"Agent分析失败，回退到前 {len(final_tables)} 个最相关的表"
            
            # 确保推理过程不为空
            if not reasoning:
                reasoning = f"基于相关性分析，选择了 {len(final_tables)} 个最相关的表"
            
            self._write_log(f"Agent表筛选完成: 从 {len(selected_tables)} 个候选表中选择了 {len(final_tables)} 个表")
            
            return final_tables, reasoning
            
        except Exception as e:
            self._write_log(f"Agent表筛选失败: {e}")
            # 出错时返回前3个最相关的表
            fallback_tables = selected_tables[:3]
            fallback_reasoning = f"Agent分析出错，使用前 {len(fallback_tables)} 个最相关的表作为备选"
            return fallback_tables, fallback_reasoning
    
    def _analyze_table_relationships(self, final_tables: List) -> str:
        """
        分析选中表之间的关联关系，识别可能需要的JOIN操作
        
        :param final_tables: 最终选择的表列表
        :return: 关联关系分析结果
        """
        try:
            if len(final_tables) <= 1:
                return "单表查询，无需关联分析"
            
            # 构建表结构信息
            tables_structure = ""
            table_columns = {}
            
            for table_rel in final_tables:
                table_name = table_rel.table_name
                tables_structure += f"表: {table_name}\n"
                
                # 收集字段信息
                columns = []
                if hasattr(table_rel, 'matched_columns') and table_rel.matched_columns:
                    for col in table_rel.matched_columns:
                        col_name = col.get('col', '')
                        col_type = col.get('type', '')
                        columns.append(f"{col_name} ({col_type})")
                
                # 如果没有匹配的字段信息，尝试从原始表数据获取
                if not columns:
                    # 这里可以扩展从self.table_selector.tables中获取完整字段信息
                    columns = ["字段信息未完全加载"]
                
                table_columns[table_name] = columns
                tables_structure += f"  字段: {', '.join(columns[:5])}\n"  # 只显示前5个字段
                if len(columns) > 5:
                    tables_structure += f"  ... 还有 {len(columns) - 5} 个字段\n"
                tables_structure += "\n"
            
            # 使用LLM分析表关联关系
            prompt = f"""
            你是一个数据库关联分析专家，需要分析多个表之间的潜在关联关系。
            
            【表结构信息】:
            {tables_structure}
            
            【任务】:
            分析这些表之间可能的关联关系，识别：
            1. 哪些表可能有主外键关系
            2. 常见的关联字段（如ID、名称等）
            3. 建议的JOIN方式
            4. 关联的业务逻辑
            
            【输出要求】:
            - 简洁明了，重点突出
            - 如果发现明确的关联关系，说明JOIN条件
            - 如果关联关系不明确，说明可能的连接方式
            - 控制在150字以内
            
            请直接输出分析结果，不要包含格式标记。
            """
            
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            
            join_analysis = resp.choices[0].message.content.strip()
            
            # 添加表数量信息
            analysis_result = f"涉及 {len(final_tables)} 个表的关联分析:\n\n{join_analysis}"
            
            self._write_log(f"表关联分析完成: {len(final_tables)} 个表")
            
            return analysis_result
            
        except Exception as e:
            self._write_log(f"表关联分析失败: {e}")
            # 提供基础的关联分析
            table_names = [table_rel.table_name for table_rel in final_tables]
            fallback_analysis = f"涉及 {len(final_tables)} 个表: {', '.join(table_names)}。"
            fallback_analysis += "建议检查表之间是否有共同的ID字段或名称字段进行关联。"
            fallback_analysis += "如果是业务相关的表，通常通过主键-外键关系或共同的业务标识符进行JOIN。"
            
            return fallback_analysis

    def generate_sql_for_possibility(self, possibility: QueryPossibility, context: str, original_query: str) -> str:
        """
        为特定的查询可能性生成SQL语句
        
        :param possibility: 查询可能性对象
        :param context: Schema上下文信息
        :param original_query: 用户原始查询
        :return: 生成的SQL语句
        """
        try:
            # 构建针对特定理解方式的提示词
            sys_prompt = f"""
            你是一个精通 SQLite 的高级数据库工程师。
            
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
            根据上述确定的理解方式，编写精确的 SQL 语句。
            
            【严格约束】:
            1. 仅输出 SQL 代码，不要任何解释。
            2. 不要使用 Markdown 格式 (不要写 ```sql)。
            3. 日期处理请使用 `strftime` 函数，例如 `strftime('%Y', order_date) = '2023'`。
            4. 严格按照关键解释中的SQL条件、表达式和模式来构建查询。
            5. 确保SQL语法正确且可执行。
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
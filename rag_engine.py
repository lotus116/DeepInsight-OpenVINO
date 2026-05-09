import numpy as np
from optimum.intel import OVModelForFeatureExtraction
from transformers import AutoTokenizer
import torch
import os
import json
import time  # 新增：用于计时
import psutil  # 新增：用于内存监控
import re
from typing import List, Dict, Optional, Tuple, Any
from sqlalchemy import create_engine, inspect, text

# --- 1. 依赖库容错导入 ---
try:
    import fitz  # PyMuPDF, 用于读取 PDF
except ImportError:
    fitz = None
    print("⚠️ 提示: 未检测到 pymupdf 库，PDF 解析功能将不可用。")

try:
    import docx  # python-docx, 用于读取 Word
except ImportError:
    docx = None
    print("⚠️ 提示: 未检测到 python-docx 库，Word 解析功能将不可用。")


class IntelRAG:
    def __init__(self, model_path, db_uris=None, kb_paths=None):
        """
        RAG 引擎核心类
        :param model_path: OpenVINO 导出的 Embedding 模型文件夹路径
        :param db_uris: 数据库连接字符串列表 (支持多库)
        :param kb_paths: 知识库文件路径列表 (PDF/TXT/JSON/Word)
        """
        # 处理参数兼容性
        if db_uris is None:
            db_uris = []
        if kb_paths is None:
            kb_paths = []
            
        print(f"⚡ [RAG] 引擎初始化...")
        print(f"   📂 模型路径: {model_path}")
        print(f"   🗄️ 数据库源: {len(db_uris)} 个")
        print(f"   📚 知识文件: {len(kb_paths)} 个")
        
        # 1. 加载模型
        self.embedding_dim = 512  # 默认维度，将从模型配置中自动更新
        
        if not os.path.exists(model_path):
            print(f"❌ 严重错误: 模型路径不存在 {model_path}")
            self.model = None
            self.tokenizer = None
        else:
            try:
                # 使用 OpenVINO 加速推理
                self.model = OVModelForFeatureExtraction.from_pretrained(
                    model_path, 
                    device="CPU", 
                    ov_config={"PERFORMANCE_HINT": "LATENCY"}
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                # 自动获取 embedding 维度 (从模型配置中读取 hidden_size)
                if hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):
                    self.embedding_dim = self.model.config.hidden_size
                    print(f"✅ OpenVINO 模型加载成功 (向量维度: {self.embedding_dim})")
                else:
                    print(f"✅ OpenVINO 模型加载成功 (使用默认向量维度: {self.embedding_dim})")
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
                self.model = None

        self.db_uris = db_uris
        self.kb_paths = kb_paths
        
        # 内存向量库
        self.documents = []   # 存文本
        self.embeddings = None # 存向量 (NumPy Matrix)
        
        # 2. 构建知识库
        self._build_knowledge_base()

    def _get_embedding(self, text):
        """将文本转换为向量"""
        if self.model is None or not text: 
            return np.zeros(self.embedding_dim)  # 使用自动获取的维度
            
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 取 CLS 向量或首位向量
                return outputs.last_hidden_state[:, 0].squeeze().numpy()
        except Exception as e:
            print(f"⚠️ 向量化失败: {e}")
            return np.zeros(self.embedding_dim)  # 使用自动获取的维度

    def _read_file(self, file_path):
        """通用文件解析器"""
        if not os.path.exists(file_path):
            return ""
            
        ext = os.path.splitext(file_path)[1].lower()
        content = []
        
        try:
            # === PDF 解析 ===
            if ext == '.pdf':
                if fitz:
                    with fitz.open(file_path) as doc:
                        for page in doc: 
                            content.append(page.get_text())
                else:
                    return "Error: 缺少 pymupdf 库"

            # === Word 解析 ===
            elif ext == '.docx':
                if docx:
                    doc = docx.Document(file_path)
                    content = [p.text for p in doc.paragraphs if p.text.strip()]
                else:
                    return "Error: 缺少 python-docx 库"

            # === JSON 解析 (增强版 - 支持 schema 语义分块 + prompt_config 格式) ===
            elif ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 检测是否为 prompt_config 格式 (有 query_patterns 或 example_queries)
                if isinstance(data, dict) and ('query_patterns' in data or 'example_queries' in data):
                    # 处理 query_patterns (SQL查询模式)
                    for pattern in data.get('query_patterns', []):
                        doc = f"【查询模式】{pattern.get('intent', '')}\n"
                        doc += f"关键词: {', '.join(pattern.get('keywords', []))}\n"
                        doc += f"SQL模式: {pattern.get('pattern', '')}\n"
                        doc += f"说明: {pattern.get('explanation', '')}\n"
                        if pattern.get('common_mistakes'):
                            doc += f"常见错误: {', '.join(pattern.get('common_mistakes', []))}\n"
                        content.append(doc)
                    
                    # 处理 example_queries (示例查询)
                    for example in data.get('example_queries', []):
                        doc = f"【查询示例】{example.get('query', '')}\n"
                        doc += f"类别: {example.get('category', '')}\n"
                        if example.get('sql_pattern'):
                            doc += f"SQL: {example.get('sql_pattern')}\n"
                        if example.get('description'):
                            doc += f"说明: {example.get('description')}\n"
                        content.append(doc)
                    
                    # 处理 key_concepts (业务概念)
                    for concept, definition in data.get('key_concepts', {}).items():
                        doc = f"【业务概念】{concept}\n"
                        doc += f"定义: {definition}\n"
                        content.append(doc)
                    
                    # 处理 common_metrics (常用指标)
                    for metric, formula in data.get('common_metrics', {}).items():
                        doc = f"【业务指标】{metric}\n"
                        doc += f"计算公式: {formula}\n"
                        content.append(doc)
                    
                    # 处理 term_dictionary (术语词典)
                    for term, explanation in data.get('term_dictionary', {}).items():
                        doc = f"【术语】{term}\n"
                        doc += f"解释: {explanation}\n"
                        content.append(doc)
                    
                    # 处理 business_context (业务上下文)
                    biz_ctx = data.get('business_context', {})
                    if biz_ctx:
                        if biz_ctx.get('business_rules'):
                            doc = f"【业务规则】Northwind\n"
                            doc += f"规则: {biz_ctx.get('business_rules')}\n"
                            content.append(doc)
                
                # 检测是否为旧版查询模式知识库格式 (patterns 字段)
                elif isinstance(data, dict) and 'patterns' in data:
                    for pattern in data.get('patterns', []):
                        doc = f"【查询模式】{pattern.get('intent', '')}\n"
                        doc += f"关键词: {', '.join(pattern.get('keywords', []))}\n"
                        doc += f"SQL模式: {pattern.get('pattern', '')}\n"
                        doc += f"说明: {pattern.get('explanation', '')}\n"
                        if pattern.get('examples'):
                            doc += f"示例: {pattern['examples'][0]}\n"
                        if pattern.get('common_mistakes'):
                            doc += f"常见错误: {', '.join(pattern.get('common_mistakes', []))}\n"
                        content.append(doc)
                    
                    # 处理业务规则
                    business_rules = data.get('business_rules', {})
                    for db_name, rules in business_rules.items():
                        domain = rules.get('domain', '')
                        for concept, definition in rules.get('key_concepts', {}).items():
                            doc = f"【业务规则】[{db_name}] {concept}\n"
                            doc += f"定义: {definition}\n"
                            doc += f"领域: {domain}\n"
                            content.append(doc)
                        for metric, formula in rules.get('common_metrics', {}).items():
                            doc = f"【业务指标】[{db_name}] {metric}\n"
                            doc += f"计算公式: {formula}\n"
                            doc += f"领域: {domain}\n"
                            content.append(doc)
                
                # 检测是否为增强版 schema 格式 (有 table_name 和 columns)
                elif isinstance(data, list) and len(data) > 0:
                    first_item = data[0]
                    is_enhanced_schema = 'table_name' in first_item and 'columns' in first_item
                    
                    if is_enhanced_schema:
                        # 增强版 Schema: 按表分块，每个表一个完整文档
                        for table in data:
                            doc = self._format_table_as_document(table)
                            content.append(doc)
                    else:
                        # 普通 JSON 列表 (旧格式)
                        for item in data:
                            content.append(f"表名: {item.get('table_name')}, 描述: {item.get('description')}")
                            for col in item.get('columns', []):
                                col_name = col.get('name', col.get('col', ''))
                                col_desc = f"字段 {col_name}: {col.get('description', '')}"
                                if 'formula' in col: col_desc += f", 计算公式: {col['formula']}"
                                content.append(col_desc)
                else:
                    content.append(json.dumps(data, ensure_ascii=False))

            # === 纯文本/Markdown ===
            elif ext in ['.txt', '.md', '.csv', '.jsonl']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content.append(f.read())
            
            return "\n".join(content)
            
        except Exception as e:
            print(f"⚠️ 读取文件 {os.path.basename(file_path)} 失败: {e}")
            return ""
    
    def _format_table_as_document(self, table: dict) -> str:
        """
        将表信息格式化为完整的语义文档 (用于 RAG 检索)
        
        包含: 表名、描述、主键、外键关系、业务概念、示例查询、列信息
        """
        lines = []
        
        # 表基本信息
        table_name = table.get('table_name', '')
        description = table.get('table_description', table.get('description', ''))
        lines.append(f"【表名】{table_name}")
        lines.append(f"【描述】{description}")
        
        # 主键
        pk = table.get('primary_key', '')
        if pk:
            if isinstance(pk, list):
                pk = ", ".join(pk)
            lines.append(f"【主键】{pk}")
        
        # 外键关系 (关键信息，帮助 JOIN 推断)
        foreign_keys = table.get('foreign_keys', [])
        if foreign_keys:
            fk_lines = []
            for fk in foreign_keys:
                col = fk.get('column', '')
                ref = fk.get('references', '')
                desc = fk.get('description', '')
                fk_lines.append(f"  - {col} → {ref} ({desc})")
            lines.append(f"【外键关系】\n" + "\n".join(fk_lines))
        
        # 业务概念 (语义匹配关键词)
        concepts = table.get('business_concepts', [])
        if concepts:
            lines.append(f"【业务概念】{', '.join(concepts)}")
        
        # 示例查询场景
        sample_queries = table.get('sample_queries', [])
        if sample_queries:
            lines.append(f"【典型查询场景】{'; '.join(sample_queries)}")
        
        # 列详情
        columns = table.get('columns', [])
        if columns:
            col_lines = []
            for col in columns:
                col_name = col.get('col', col.get('name', ''))
                col_type = col.get('type', '')
                col_desc = col.get('description', '')
                col_lines.append(f"  - {col_name} ({col_type}): {col_desc}")
            lines.append(f"【字段详情】\n" + "\n".join(col_lines))
        
        return "\n".join(lines)

    def _build_knowledge_base(self):
        """核心：扫描数据库 + 读取文件 -> 向量化"""
        self.documents = []

        # --- 步骤 A: 扫描数据库结构 (Schema) ---
        if self.db_uris:
            print(f"🔍 正在扫描 {len(self.db_uris)} 个数据库结构...")
            for uri in self.db_uris:
                if not uri: continue
                try:
                    engine = create_engine(uri)
                    inspector = inspect(engine)
                    table_names = inspector.get_table_names()
                    
                    for t_name in table_names:
                        # 获取字段信息
                        columns = inspector.get_columns(t_name)
                        col_details = []
                        for c in columns:
                            # 格式: 字段名(类型)
                            col_info = f"{c['name']}({c['type']})"
                            if c.get('comment'): col_info += f" 注释:{c['comment']}"
                            col_details.append(col_info)
                        
                        # 组合成一条文档
                        doc = f"数据库表名: {t_name}\n包含字段: {', '.join(col_details)}"
                        self.documents.append(doc)
                except Exception as e:
                    print(f"❌ 数据库连接/扫描失败 ({uri}): {e}")

        # --- 步骤 B: 读取知识库文件 ---
        if self.kb_paths:
            print(f"📂 正在读取 {len(self.kb_paths)} 个文件...")
            for path in self.kb_paths:
                text = self._read_file(path)
                # 检查读取是否失败 (只检查以 "Error:" 开头的错误信息，避免误匹配列名如 ErrorLogID)
                if not text or text.startswith("Error:"): continue
                
                # 智能分块策略
                # 1. Schema 表格式 (以【表名】开头)，按表分块
                # 2. 知识库语义格式 (以【查询模式】【查询示例】等开头)，按语义标记分块
                # 3. 其他格式，使用传统 500 字符固定分块
                
                # 所有语义标记列表
                semantic_markers = ['【表名】', '【查询模式】', '【查询示例】', '【业务概念】', 
                                   '【业务指标】', '【术语】', '【业务规则】']
                
                # 检测是否以任何语义标记开头
                is_semantic_format = any(text.startswith(marker) for marker in semantic_markers)
                
                if is_semantic_format:
                    # 语义分块: 按语义标记边界分割
                    import re
                    # 构建正则表达式匹配任意语义标记
                    pattern = '(' + '|'.join(re.escape(m) for m in semantic_markers) + ')'
                    parts = re.split(pattern, text)
                    
                    # 重组: 每个标记 + 其后内容构成一个文档
                    current_marker = None
                    doc_count = 0
                    for part in parts:
                        if part in semantic_markers:
                            current_marker = part
                        elif current_marker and part.strip():
                            self.documents.append(f"{current_marker}{part.strip()}")
                            doc_count += 1
                            current_marker = None  # 重置，准备下一个文档
                    
                    # 统计并打印
                    marker_type = "Schema" if text.startswith('【表名】') else "知识库"
                    print(f"   📝 {marker_type}语义分块: {doc_count} 条文档")
                else:
                    # 传统分块: 按 500 字符切片
                    chunk_size = 500
                    for i in range(0, len(text), chunk_size):
                        chunk = text[i:i+chunk_size]
                        self.documents.append(f"来源文件[{os.path.basename(path)}]:\n{chunk}")

        # 保底处理
        if not self.documents:
            self.documents = ["暂无有效知识库信息。"]
            print("⚠️ 警告: 知识库为空，RAG 将无法提供上下文。")
        else:
            # 统计不同类型的文档数量
            pattern_count = sum(1 for d in self.documents if d.startswith('【查询模式】'))
            example_count = sum(1 for d in self.documents if d.startswith('【查询示例】'))
            concept_count = sum(1 for d in self.documents if d.startswith('【业务概念】'))
            rule_count = sum(1 for d in self.documents if d.startswith('【业务规则】'))
            metric_count = sum(1 for d in self.documents if d.startswith('【业务指标】'))
            term_count = sum(1 for d in self.documents if d.startswith('【术语】'))
            schema_count = sum(1 for d in self.documents if d.startswith('【表名】'))
            
            knowledge_total = pattern_count + example_count + concept_count + rule_count + metric_count + term_count
            if knowledge_total > 0:
                print(f"   📝 知识库: {pattern_count} 模式, {example_count} 示例, {term_count} 术语, {metric_count} 指标")
            if schema_count > 0:
                print(f"   📊 Schema: {schema_count} 个表")

        # --- 步骤 C: 向量化 (Embedding) ---
        print(f"🚀 [OpenVINO] 正在生成向量索引 (共 {len(self.documents)} 条)...")
        if self.model:
            try:
                embeddings_list = [self._get_embedding(doc) for doc in self.documents]
                self.embeddings = np.array(embeddings_list)
                print("✅ 向量化完成！")
            except Exception as e:
                print(f"❌ 向量化过程中断: {e}")
                self.embeddings = None

    def retrieve(self, query, top_k=5):
        """
        检索函数 (优化版：返回内容 + 性能指标)
        :param query: 用户问题
        :param top_k: 返回最相似的 k 条记录
        :return: (context_str, latency_ms, memory_delta_mb)
        """
        if self.embeddings is None or len(self.documents) == 0:
            return "", 0.0, 0.0
            
        # --- 性能监控开始 ---
        start_time = time.perf_counter()
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024) # MB
        
        # 1. 问题向量化
        query_emb = self._get_embedding(query)
        
        # 2. 余弦相似度计算 (Vector Search)
        # dot product / (norm_a * norm_b)
        scores = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-10
        )
        
        # 3. 排序并取 Top-K
        # argsort 返回的是从小到大的索引，所以要 [::-1] 反转
        top_k = min(top_k, len(self.documents))
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # 4. 拼接结果
        results = []
        for idx in top_indices:
            # 可以在这里打印分数调试
            results.append(self.documents[idx])
        
        context_str = "\n\n".join(results)

        # --- 性能监控结束 ---
        end_time = time.perf_counter()
        mem_after = process.memory_info().rss / (1024 * 1024)
        
        latency_ms = (end_time - start_time) * 1000
        mem_delta_mb = max(0, mem_after - mem_before) # 防止负数
            
        return context_str, latency_ms, mem_delta_mb

    # ========================================================================
    # 二阶段混合检索系统 (Two-Stage Hybrid Retrieval)
    # ========================================================================
    # 
    # 架构设计:
    # - 阶段1 (粗排/Rough): 使用 OpenVINO 加速的向量检索，快速筛选 Top-K 候选
    # - 阶段2 (精排/Pruning): 使用 LLM 进行语义理解，从候选中精选核心表
    # 
    # 三维统一检索:
    # - 维度A: Schema 精排 (粗排 -> 精排 -> 详情提取)
    # - 维度B: Few-Shot 示例匹配 (语义相似度 Top-2)
    # - 维度C: Terms 术语匹配 (关键词匹配 + 业务逻辑提取)
    # ========================================================================

    def _rough_search(self, query: str, top_k: int = 12) -> List[Dict]:
        """
        粗排阶段: 使用向量相似度快速检索 Top-K 候选表
        
        算法设计:
        1. 将用户查询向量化 (OpenVINO 加速)
        2. 计算与所有表文档的余弦相似度
        3. 返回相似度最高的 K 个表及其元信息
        
        性能约束:
        - Embedding 计算锁定在 OpenVINO 本地后端
        - 时间复杂度 O(N) 其中 N 为表数量
        
        Args:
            query: 用户自然语言查询
            top_k: 返回的候选表数量 (默认 12，基于数据分析优化)
            
        Returns:
            候选表列表，每项包含 {table_name, description, score, document}
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []
        
        start_time = time.perf_counter()
        
        # 1. 查询向量化
        query_emb = self._get_embedding(query)
        query_norm = np.linalg.norm(query_emb)
        
        if query_norm < 1e-10:
            return []
        
        # 2. 筛选表类型文档（以【表名】开头的文档）
        table_docs = []
        table_indices = []
        for idx, doc in enumerate(self.documents):
            if doc.startswith('【表名】'):
                table_docs.append(doc)
                table_indices.append(idx)
        
        if not table_docs:
            print("⚠️ [粗排] 未找到表类型文档")
            return []
        
        # 3. 计算余弦相似度（仅针对表文档）
        table_embeddings = self.embeddings[table_indices]
        scores = np.dot(table_embeddings, query_emb) / (
            np.linalg.norm(table_embeddings, axis=1) * query_norm + 1e-10
        )
        
        # 4. 排序并取 Top-K
        top_k = min(top_k, len(table_docs))
        sorted_indices = np.argsort(scores)[::-1][:top_k]
        
        # 5. 构建结果
        candidates = []
        for rank, idx in enumerate(sorted_indices):
            doc = table_docs[idx]
            score = float(scores[idx])
            
            # 解析表名和描述
            lines = doc.split('\n')
            table_name = lines[0].replace('【表名】', '').strip() if lines else ""
            description = ""
            for line in lines:
                if line.startswith('【描述】'):
                    description = line.replace('【描述】', '').strip()
                    break
            
            candidates.append({
                'table_name': table_name,
                'description': description,
                'score': score,
                'rank': rank + 1,
                'document': doc
            })
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        print(f"🔍 [粗排] 完成: {len(candidates)} 个候选表 (耗时 {latency_ms:.2f}ms)")
        
        return candidates

    def _llm_pruning(self, query: str, candidates: List[Dict], 
                     llm_client: Any = None, model_name: str = None) -> Tuple[List[str], Dict]:
        """
        精排阶段: 使用 LLM 从候选表中筛选生成 SQL 必不可少的核心表
        
        算法设计:
        1. 将候选表信息压缩为简洁的列表格式
        2. 构建精排专用 Prompt，强调"最小必要集合"
        3. 调用 LLM 进行语义推理
        4. 解析返回的核心表名列表
        
        参数:
            query: 用户自然语言查询
            candidates: 粗排阶段返回的候选表列表
            llm_client: OpenAI 兼容的 LLM 客户端（可选，用于外部注入）
            model_name: LLM 模型名称
            
        Returns:
            Tuple[核心表名列表, Token 使用统计 Dict]
            Token 统计: {'prompt_tokens': int, 'completion_tokens': int, 'total_tokens': int}
        """
        # 🆕 Token 使用统计
        token_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        
        if not candidates:
            return [], token_usage
        
        start_time = time.perf_counter()
        
        # 1. 格式化候选表信息（仅传递表名和描述，节省 Token）
        candidate_text = "\n".join([
            f"{i+1}. {c['table_name']}: {c['description'][:100]}..." 
            if len(c['description']) > 100 else f"{i+1}. {c['table_name']}: {c['description']}"
            for i, c in enumerate(candidates)
        ])
        
        # 2. 构建精排 Prompt (基于数据驱动优化：目标 Top-5)
        pruning_prompt = f"""你是一个专业的数据库架构分析师。

【用户查询】: {query}

【候选数据表】:
{candidate_text}

【任务】: 从上述候选表中，精选出生成该查询 SQL 所必需的核心表（3-6个）。

【选择原则】:
1. 优先选择存储查询所需数据的主表
2. 包含 JOIN 所需的直接关联表
3. 包含聚合/分组所需的维度表
4. 精准选择，宁缺勿滥（系统会自动补全外键依赖表）

【输出格式】:
- 仅输出表名，每行一个
- 不要解释，不要编号
- 按重要性从高到低排列

【必需的表】:"""

        # 3. 调用 LLM（如果未提供客户端，返回基于分数的兜底结果）
        if llm_client is None:
            # 兜底策略：返回分数最高的前 8 个表
            print("⚠️ [精排] LLM 客户端未提供，使用分数兜底")
            return [c['table_name'] for c in candidates[:8]], token_usage
        
        try:
            response = llm_client.chat.completions.create(
                model=model_name or "deepseek-chat",
                messages=[{"role": "user", "content": pruning_prompt}],
                temperature=0.1,  # 低温度确保输出稳定
                max_tokens=200
            )
            
            # 🆕 捕获 Token 使用统计
            if hasattr(response, 'usage') and response.usage:
                token_usage = {
                    'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0),
                    'completion_tokens': getattr(response.usage, 'completion_tokens', 0),
                    'total_tokens': getattr(response.usage, 'total_tokens', 0)
                }
                print(f"📊 [精排] Token 消耗: prompt={token_usage['prompt_tokens']}, completion={token_usage['completion_tokens']}, total={token_usage['total_tokens']}")
            
            result_text = response.choices[0].message.content.strip()
            
            # 4. 解析表名（支持多种格式）
            core_tables = []
            for line in result_text.split('\n'):
                line = line.strip()
                # 移除可能的编号前缀 (如 "1. ", "- ")
                line = re.sub(r'^[\d\.\-\*\s]+', '', line).strip()
                if line and line in [c['table_name'] for c in candidates]:
                    core_tables.append(line)
            
            # 限制数量 (3-6 个) - 最优配置
            core_tables = core_tables[:6]
            if len(core_tables) < 3:
                # 补充分数最高的表
                for c in candidates:
                    if c['table_name'] not in core_tables:
                        core_tables.append(c['table_name'])
                    if len(core_tables) >= 3:
                        break
            
            # 🆕 依赖补全：自动补充外键引用表 (最多 +3 个)
            print(f"📊 [精排] 初选 {len(core_tables)} 个核心表")
            core_tables = self._complete_table_dependencies(core_tables)
            print(f"📊 [补全后] {len(core_tables)} 个表")
            
            # 最终上限控制
            if len(core_tables) > 8:
                print(f"⚠️ [精排] 表数过多({len(core_tables)})，截断为 8 个")
                core_tables = core_tables[:8]
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            print(f"✅ [精排] 完成: {len(core_tables)} 个核心表 (耗时 {latency_ms:.2f}ms)")
            print(f"   核心表: {', '.join(core_tables)}")
            
            return core_tables, token_usage
            
        except Exception as e:
            print(f"❌ [精排] LLM 调用失败: {e}，使用分数兜底")
            return [c['table_name'] for c in candidates[:8]], token_usage

    def _complete_table_dependencies(self, selected_tables: List[str]) -> List[str]:
        """
        依赖补全：自动补充外键引用的关联表
        
        基于实际 Gold SQL 分析：
        - Medium P90 = 3 表, Hard P90 = 4 表, Extra-Hard P90 = 4 表
        - 最多补充 3 个依赖表以覆盖极端情况
        
        Args:
            selected_tables: LLM 精排后的核心表列表
            
        Returns:
            补全后的表列表（原表 + 外键依赖表）
        """
        if not self.documents or not selected_tables:
            return selected_tables
        
        completed = list(selected_tables)
        added_count = 0
        max_additions = 3  # 最多补充 3 个依赖表
        
        # 收集所有已选表的外键引用
        for table_name in list(selected_tables):  # 使用 list() 避免遍历时修改
            if added_count >= max_additions:
                break
            
            # 在文档中查找该表的外键信息
            for doc in self.documents:
                if not doc.startswith('【表名】'):
                    continue
                
                # 检查是否是目标表
                first_line = doc.split('\n')[0]
                doc_table_name = first_line.replace('【表名】', '').strip()
                
                if doc_table_name.lower() != table_name.lower():
                    continue
                
                # 提取外键关系
                # 格式: "  - col → ref_table.ref_col (description)"
                fk_pattern = r'(\w+)\s*→\s*(\w+)\.'
                for line in doc.split('\n'):
                    if added_count >= max_additions:
                        break
                    
                    matches = re.findall(fk_pattern, line)
                    for _, ref_table in matches:
                        ref_table_lower = ref_table.lower()
                        # 检查是否已在列表中
                        if ref_table_lower not in [t.lower() for t in completed]:
                            # 验证引用表确实存在于知识库中
                            if self._table_exists_in_documents(ref_table):
                                completed.append(ref_table)
                                added_count += 1
                                print(f"📎 [依赖补全] {table_name} → {ref_table}")
                                if added_count >= max_additions:
                                    break
                break  # 已处理该表，跳出内层循环
        
        if added_count > 0:
            print(f"✅ [依赖补全] 共补充 {added_count} 个关联表")
        
        return completed
    
    def _table_exists_in_documents(self, table_name: str) -> bool:
        """
        检查指定表是否存在于知识库文档中
        """
        table_name_lower = table_name.lower()
        for doc in self.documents:
            if doc.startswith('【表名】'):
                first_line = doc.split('\n')[0]
                doc_table = first_line.replace('【表名】', '').strip()
                if doc_table.lower() == table_name_lower:
                    return True
        return False

    def _extract_table_details(self, table_names: List[str]) -> List[Dict]:
        """
        提取核心表的详细信息（列名、类型、描述）
        
        Args:
            table_names: 核心表名列表
            
        Returns:
            表详情列表，每项包含 {table_name, description, columns, foreign_keys, document}
        """
        details = []
        table_name_lower_map = {name.lower(): name for name in table_names}
        
        for doc in self.documents:
            if not doc.startswith('【表名】'):
                continue
            
            # 解析表名
            first_line = doc.split('\n')[0]
            doc_table_name = first_line.replace('【表名】', '').strip()
            
            if doc_table_name.lower() not in table_name_lower_map:
                continue
            
            # 解析文档内容
            detail = {
                'table_name': doc_table_name,
                'description': '',
                'columns': [],
                'foreign_keys': [],
                'document': doc
            }
            
            current_section = None
            for line in doc.split('\n'):
                if line.startswith('【描述】'):
                    detail['description'] = line.replace('【描述】', '').strip()
                elif line.startswith('【字段详情】'):
                    current_section = 'columns'
                elif line.startswith('【外键关系】'):
                    current_section = 'fk'
                elif line.strip().startswith('- ') and current_section == 'columns':
                    # 解析列信息: "- col_name (type): description"
                    col_line = line.strip()[2:]  # 移除 "- "
                    match = re.match(r'(\w+)\s*\(([^)]+)\):\s*(.*)', col_line)
                    if match:
                        detail['columns'].append({
                            'name': match.group(1),
                            'type': match.group(2),
                            'description': match.group(3)
                        })
                elif line.strip().startswith('- ') and current_section == 'fk':
                    detail['foreign_keys'].append(line.strip()[2:])
            
            details.append(detail)
        
        return details

    def _get_sample_data(self, table_name: str, db_engine: Any = None, limit: int = 3) -> List[Dict]:
        """
        获取表的样本数据（用于增强 LLM 对数据格式的理解）
        
        注意: 此方法需要数据库连接，如果连接不可用则返回空列表
        
        Args:
            table_name: 表名
            db_engine: SQLAlchemy Engine 对象
            limit: 样本行数 (默认 3)
            
        Returns:
            样本数据列表，每项为一行记录的字典
        """
        if db_engine is None:
            return []
        
        try:
            with db_engine.connect() as conn:
                result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT {limit}"))
                columns = result.keys()
                rows = result.fetchall()
                
                sample_data = []
                for row in rows:
                    sample_data.append(dict(zip(columns, row)))
                
                return sample_data
        except Exception as e:
            print(f"⚠️ [样本数据] 获取 {table_name} 失败: {e}")
            return []

    def _match_few_shot_examples(self, query: str, config: Dict = None, top_k: int = 2) -> List[Dict]:
        """
        维度B: 语义相似度匹配 Few-Shot 示例
        
        从配置的 example_queries 中，找到与当前查询最相似的示例
        
        Args:
            query: 用户查询
            config: 包含 example_queries 的配置字典
            top_k: 返回的示例数量 (默认 2)
            
        Returns:
            匹配的示例列表
        """
        if not config or 'example_queries' not in config:
            return []
        
        examples = config.get('example_queries', [])
        if not examples:
            return []
        
        # 计算查询与每个示例的相似度
        query_emb = self._get_embedding(query)
        query_norm = np.linalg.norm(query_emb)
        
        if query_norm < 1e-10:
            return examples[:top_k]
        
        scored_examples = []
        for ex in examples:
            ex_query = ex.get('query', '')
            ex_emb = self._get_embedding(ex_query)
            ex_norm = np.linalg.norm(ex_emb)
            
            if ex_norm > 1e-10:
                score = np.dot(query_emb, ex_emb) / (query_norm * ex_norm)
                scored_examples.append((score, ex))
        
        # 排序并返回 Top-K
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        
        matched = [ex for _, ex in scored_examples[:top_k]]
        if matched:
            print(f"💡 [Few-Shot] 匹配到 {len(matched)} 个示例")
        
        return matched

    def _match_terms(self, query: str, config: Dict = None) -> List[Dict]:
        """
        维度C: 术语匹配与业务逻辑提取 (优化版: 支持 keywords 触发器)
        
        匹配策略:
        1. 精确术语匹配: 术语名直接出现在查询中
        2. 关键词触发器: 查询包含 keywords 列表中的任意词
        
        Args:
            query: 用户查询
            config: 包含 term_dictionary 的配置字典
            
        Returns:
            匹配的术语列表，每项包含 {term, explanation, required_tables, sql_pattern}
        """
        if not config or 'term_dictionary' not in config:
            return []
        
        term_dict = config.get('term_dictionary', {})
        matched_terms = []
        query_lower = query.lower()  # 统一小写匹配
        
        for term, info in term_dict.items():
            matched = False
            match_reason = ""
            
            # ============ 层级1: 精确术语匹配 ============
            if term.lower() in query_lower:
                matched = True
                match_reason = f"术语匹配: {term}"
            
            # ============ 层级2: keywords 触发器匹配 ============
            elif isinstance(info, dict) and 'keywords' in info:
                for keyword in info['keywords']:
                    if keyword.lower() in query_lower:
                        matched = True
                        match_reason = f"关键词触发: '{keyword}' → {term}"
                        break  # 匹配到一个即可
            
            # ============ 添加到结果 ============
            if matched:
                if isinstance(info, dict):
                    matched_terms.append({
                        'term': term,
                        'explanation': info.get('explanation', ''),
                        'sql_pattern': info.get('sql_pattern', ''),
                        'required_tables': info.get('required_tables', []),
                        'match_reason': match_reason
                    })
                elif isinstance(info, str):
                    matched_terms.append({
                        'term': term,
                        'explanation': info,
                        'required_tables': [],
                        'match_reason': match_reason
                    })
        
        if matched_terms:
            print(f"📚 [Terms] 匹配到 {len(matched_terms)} 个术语:")
            for t in matched_terms:
                print(f"   - {t['match_reason']}")
        
        return matched_terms

    def _get_database_index(self) -> List[str]:
        """
        生成全库表名简表（Database Index）
        
        作为 Prompt 头部的"全局导航"，让 LLM 知道所有可用的表
        Token 消耗极低（约 50-100 tokens）
        
        Returns:
            所有表名列表
        """
        tables = []
        for doc in self.documents:
            if doc.startswith('【表名】'):
                first_line = doc.split('\n')[0]
                table_name = first_line.replace('【表名】', '').strip()
                if table_name:
                    tables.append(table_name)
        
        return tables

    def retrieve_context(self, query: str, config: Dict = None,
                        llm_client: Any = None, model_name: str = None,
                        db_engine: Any = None,
                        enable_pruning: bool = True,
                        rough_top_k: int = 12,
                        include_sample_data: bool = True) -> Dict:
        """
        三维统一检索入口 (Knowledge Scheduler)
        
        实现"知识调度员"机制，整合三个维度的检索结果:
        - 维度A (Schema精排): 粗排 -> 精排 -> 详情提取
        - 维度B (Few-Shot): 语义相似度匹配示例
        - 维度C (Terms): 术语匹配与业务逻辑提取
        - 附加: Database Index 全局导航
        
        Args:
            query: 用户自然语言查询
            config: KECA 配置字典（包含 example_queries, term_dictionary 等）
            llm_client: OpenAI 兼容的 LLM 客户端
            model_name: LLM 模型名称
            db_engine: SQLAlchemy Engine（用于获取样本数据）
            enable_pruning: 是否启用 LLM 精排（设为 False 则仅使用粗排）
            rough_top_k: 粗排返回的候选数量
            include_sample_data: 是否包含样本数据
            
        Returns:
            Dict: {
                'database_index': List[str],      # 全库表名列表
                'rough_candidates': List[Dict],   # 粗排候选表
                'core_tables': List[str],         # 精排核心表名
                'core_table_details': List[Dict], # 核心表详细信息
                'matched_examples': List[Dict],   # 匹配的 Few-Shot 示例
                'matched_terms': List[Dict],      # 匹配的业务术语
                'sample_data': Dict[str, List],   # 表名 -> 样本数据
                'metrics': Dict                   # 性能指标
            }
        """
        start_time = time.perf_counter()
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)
        
        result = {
            'database_index': [],
            'rough_candidates': [],
            'core_tables': [],
            'core_table_details': [],
            'matched_examples': [],
            'matched_terms': [],
            'sample_data': {},
            'metrics': {
                'rough_latency_ms': 0,
                'pruning_latency_ms': 0,
                'total_latency_ms': 0,
                'memory_delta_mb': 0
            }
        }
        
        # 1. Database Index (全局导航)
        result['database_index'] = self._get_database_index()
        print(f"📋 [Index] 数据库共 {len(result['database_index'])} 张表")
        
        # 2. 维度A - Schema 精排
        # 2.1 粗排
        rough_start = time.perf_counter()
        result['rough_candidates'] = self._rough_search(query, top_k=rough_top_k)
        result['metrics']['rough_latency_ms'] = (time.perf_counter() - rough_start) * 1000
        
        # 2.2 精排 (可选)
        if enable_pruning and result['rough_candidates']:
            pruning_start = time.perf_counter()
            # 🆕 解包精排返回值：(核心表列表, Token 使用统计)
            core_tables, selector_token_usage = self._llm_pruning(
                query, result['rough_candidates'], llm_client, model_name
            )
            result['core_tables'] = core_tables
            result['metrics']['pruning_latency_ms'] = (time.perf_counter() - pruning_start) * 1000
            # 🆕 保存精排器的 Token 消耗（G2 第一次 LLM 调用）
            result['metrics']['selector_token_usage'] = selector_token_usage
        else:
            # 不启用精排时，使用粗排 Top-5
            result['core_tables'] = [c['table_name'] for c in result['rough_candidates'][:5]]
            result['metrics']['selector_token_usage'] = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        
        # 2.3 提取核心表详情
        if result['core_tables']:
            result['core_table_details'] = self._extract_table_details(result['core_tables'])
        
        # 2.4 获取样本数据 (可选)
        if include_sample_data and db_engine and result['core_tables']:
            for table_name in result['core_tables']:
                sample = self._get_sample_data(table_name, db_engine, limit=3)
                if sample:
                    result['sample_data'][table_name] = sample
        
        # 3. 维度B - Few-Shot 匹配
        if config:
            result['matched_examples'] = self._match_few_shot_examples(query, config, top_k=2)
        
        # 4. 维度C - Terms 匹配
        if config:
            result['matched_terms'] = self._match_terms(query, config)
            
            # 检查术语是否要求额外的表（Hint Injection）
            for term_info in result['matched_terms']:
                for req_table in term_info.get('required_tables', []):
                    if req_table.lower() not in [t.lower() for t in result['core_tables']]:
                        result['core_tables'].append(req_table)
                        print(f"📌 [Hint Injection] 术语 '{term_info['term']}' 要求添加表: {req_table}")
        
        # 性能指标
        end_time = time.perf_counter()
        mem_after = process.memory_info().rss / (1024 * 1024)
        
        result['metrics']['total_latency_ms'] = (end_time - start_time) * 1000
        result['metrics']['memory_delta_mb'] = max(0, mem_after - mem_before)
        
        print(f"✅ [三维检索] 完成: {len(result['core_tables'])} 核心表, "
              f"{len(result['matched_examples'])} 示例, {len(result['matched_terms'])} 术语 "
              f"(总耗时 {result['metrics']['total_latency_ms']:.2f}ms)")
        
        return result
import numpy as np
from optimum.intel import OVModelForFeatureExtraction
from transformers import AutoTokenizer
import torch
import os
import json
import time  # 新增：用于计时
import psutil  # 新增：用于内存监控
from sqlalchemy import create_engine, inspect

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
                print("✅ OpenVINO 模型加载成功")
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
            return np.zeros(384) # 这里的维度取决于你用的模型，bge-small 是 384 或 512
            
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 取 CLS 向量或首位向量
                return outputs.last_hidden_state[:, 0].squeeze().numpy()
        except Exception as e:
            print(f"⚠️ 向量化失败: {e}")
            return np.zeros(384)

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

            # === JSON 解析 ===
            elif ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 将 JSON 结构转为字符串描述，方便检索
                    if isinstance(data, list): # 如果是 schema_desc.json 这种列表
                        for item in data:
                            content.append(f"表名: {item.get('table_name')}, 描述: {item.get('description')}")
                            for col in item.get('columns', []):
                                col_desc = f"字段 {col['name']}: {col.get('description')}"
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
                if not text or "Error" in text: continue
                
                # 文本切片 (Chunking)
                # 为了防止文本过长超过 LLM 窗口，这里按 500 字符切片
                chunk_size = 500
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i+chunk_size]
                    # 加上文件名作为上下文
                    self.documents.append(f"来源文件[{os.path.basename(path)}]:\n{chunk}")

        # 保底处理
        if not self.documents:
            self.documents = ["暂无有效知识库信息。"]
            print("⚠️ 警告: 知识库为空，RAG 将无法提供上下文。")

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
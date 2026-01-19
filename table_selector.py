"""
智能表选择算法 - 基于OpenVINO优化的语义匹配
动态读取用户上传的schema文件，使用向量相似度进行智能表选择
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import os
import time
import psutil


@dataclass
class TableRelevance:
    """表相关性评分结果"""
    table_name: str
    table_description: str
    relevance_score: float
    semantic_similarity: float  # 语义相似度得分
    keyword_matches: List[str]
    matched_columns: List[Dict]
    reasoning: str
    is_primary: bool = False  # 是否为主要表
    is_join_required: bool = False  # 是否需要关联


class IntelligentTableSelector:
    """基于OpenVINO的智能表选择器"""
    
    def __init__(self, rag_engine=None, schema_paths=None):
        """
        初始化表选择器
        :param rag_engine: 已初始化的IntelRAG实例，用于向量计算
        :param schema_paths: 用户上传的schema文件路径列表
        """
        self.rag_engine = rag_engine
        self.schema_paths = schema_paths or []
        self.tables = []
        self.table_embeddings = None
        self.column_embeddings = None
        
        # 动态加载schema
        self.load_dynamic_schema()
        
        # 如果有RAG引擎，预计算表和列的向量
        if self.rag_engine and self.rag_engine.model:
            self.precompute_embeddings()
    
    def load_dynamic_schema(self):
        """动态加载用户上传的schema文件"""
        self.tables = []
        
        if not self.schema_paths:
            print("⚠️ [TableSelector] 未配置schema文件路径")
            return
        
        for schema_path in self.schema_paths:
            if not os.path.exists(schema_path):
                print(f"⚠️ [TableSelector] Schema文件不存在: {schema_path}")
                continue
                
            try:
                with open(schema_path, 'r', encoding='utf-8') as f:
                    schema_data = json.load(f)
                    
                # 支持不同的schema格式
                if isinstance(schema_data, list):
                    # 格式1: 直接是表列表 (如northwind格式)
                    self.tables.extend(schema_data)
                elif isinstance(schema_data, dict):
                    # 格式2: 包装在对象中
                    if 'tables' in schema_data:
                        self.tables.extend(schema_data['tables'])
                    elif 'schema' in schema_data:
                        self.tables.extend(schema_data['schema'])
                    else:
                        # 格式3: 单个表定义
                        self.tables.append(schema_data)
                        
                print(f"✅ [TableSelector] 成功加载schema: {schema_path} ({len(schema_data)} 个表)")
                
            except Exception as e:
                print(f"❌ [TableSelector] 加载schema失败 {schema_path}: {e}")
        
        print(f"📊 [TableSelector] 总共加载 {len(self.tables)} 个数据表")
    
    def precompute_embeddings(self):
        """预计算所有表和列的向量表示"""
        if not self.tables or not self.rag_engine:
            return
            
        print("🚀 [TableSelector] 正在预计算表向量...")
        
        # 为每个表生成描述文本并向量化
        table_texts = []
        column_texts = []
        
        for table in self.tables:
            table_name = table.get("table_name", "")
            table_desc = table.get("table_description", "")
            columns = table.get("columns", [])
            
            # 表级别描述
            table_text = f"表名: {table_name}. 描述: {table_desc}"
            table_texts.append(table_text)
            
            # 列级别描述
            for col in columns:
                col_name = col.get("col", "")
                col_type = col.get("type", "")
                col_desc = col.get("description", "")
                
                col_text = f"表 {table_name} 的字段 {col_name} ({col_type}): {col_desc}"
                column_texts.append({
                    "text": col_text,
                    "table_name": table_name,
                    "column": col
                })
        
        # 批量向量化
        try:
            if table_texts:
                table_embeddings_list = [self.rag_engine._get_embedding(text) for text in table_texts]
                self.table_embeddings = np.array(table_embeddings_list)
                
            if column_texts:
                col_embeddings_list = [self.rag_engine._get_embedding(item["text"]) for item in column_texts]
                self.column_embeddings = {
                    "embeddings": np.array(col_embeddings_list),
                    "metadata": column_texts
                }
                
            print(f"✅ [TableSelector] 向量预计算完成: {len(table_texts)} 个表, {len(column_texts)} 个字段")
            
        except Exception as e:
            print(f"❌ [TableSelector] 向量预计算失败: {e}")
            self.table_embeddings = None
            self.column_embeddings = None
    
    def calculate_semantic_similarity(self, query: str, table_index: int) -> float:
        """计算查询与表的语义相似度"""
        if not self.rag_engine or self.table_embeddings is None:
            return 0.0
            
        try:
            # 查询向量化
            query_embedding = self.rag_engine._get_embedding(query)
            
            # 计算与指定表的余弦相似度
            table_embedding = self.table_embeddings[table_index]
            similarity = np.dot(query_embedding, table_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(table_embedding) + 1e-10
            )
            
            return float(similarity)
            
        except Exception as e:
            print(f"⚠️ [TableSelector] 语义相似度计算失败: {e}")
            return 0.0
    
    def find_relevant_columns(self, query: str, table_name: str, top_k: int = 5) -> List[Dict]:
        """找到与查询最相关的列"""
        if not self.column_embeddings:
            return []
            
        try:
            query_embedding = self.rag_engine._get_embedding(query)
            
            # 筛选出属于指定表的列
            table_columns = []
            table_indices = []
            
            for i, col_meta in enumerate(self.column_embeddings["metadata"]):
                if col_meta["table_name"] == table_name:
                    table_columns.append(col_meta["column"])
                    table_indices.append(i)
            
            if not table_indices:
                return []
            
            # 计算相似度
            col_embeddings = self.column_embeddings["embeddings"][table_indices]
            similarities = np.dot(col_embeddings, query_embedding) / (
                np.linalg.norm(col_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-10
            )
            
            # 排序并返回top-k
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            relevant_columns = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # 相似度阈值
                    col_info = table_columns[idx].copy()
                    col_info["similarity"] = float(similarities[idx])
                    relevant_columns.append(col_info)
            
            return relevant_columns
            
        except Exception as e:
            print(f"⚠️ [TableSelector] 列相关性计算失败: {e}")
            return []
    
    def analyze_query_intent(self, query: str) -> Dict:
        """分析查询意图（简化版，主要依赖语义匹配）"""
        query_lower = query.lower()
        
        intent = {
            "has_aggregation": any(word in query_lower for word in ["总", "平均", "最大", "最小", "统计", "计算", "sum", "avg", "max", "min", "count"]),
            "has_filtering": any(word in query_lower for word in ["where", "条件", "筛选", "过滤", "等于", "大于", "小于"]),
            "has_grouping": any(word in query_lower for word in ["按", "分组", "group by", "每个", "各个"]),
            "has_sorting": any(word in query_lower for word in ["排序", "排名", "最高", "最低", "order by", "top"]),
            "has_time": any(word in query_lower for word in ["年", "月", "日", "时间", "日期", "yesterday", "today", "2023", "2024"]),
            "has_geography": any(word in query_lower for word in ["城市", "地区", "国家", "省", "州", "city", "country", "region"])
        }
        
        return intent
    
    def calculate_table_relevance(self, table: Dict, query: str, table_index: int) -> TableRelevance:
        """计算表的综合相关性得分"""
        table_name = table.get("table_name", "")
        table_desc = table.get("table_description", "")
        
        # 1. 语义相似度 (权重: 60%)
        semantic_score = self.calculate_semantic_similarity(query, table_index)
        
        # 2. 关键词匹配 (权重: 25%)
        keyword_score = 0.0
        matched_keywords = []
        
        query_lower = query.lower()
        table_text = f"{table_name} {table_desc}".lower()
        
        # 简单关键词匹配 - 改进中文分词
        query_words = []
        # 英文按空格分词
        for word in query_lower.split():
            query_words.append(word)
        
        # 中文按字符分词（简单方法）
        chinese_chars = []
        for char in query_lower:
            if '\u4e00' <= char <= '\u9fff':  # 中文字符范围
                chinese_chars.append(char)
        
        # 组合中文词（2-3字组合）
        for i in range(len(chinese_chars)):
            if i + 1 < len(chinese_chars):
                query_words.append(chinese_chars[i] + chinese_chars[i+1])
            if i + 2 < len(chinese_chars):
                query_words.append(chinese_chars[i] + chinese_chars[i+1] + chinese_chars[i+2])
        
        for word in query_words:
            if len(word) > 1 and word in table_text:
                keyword_score += 0.1
                matched_keywords.append(word)
        
        # 3. 列相关性 (权重: 15%)
        relevant_columns = self.find_relevant_columns(query, table_name, top_k=3)
        column_score = min(len(relevant_columns) * 0.05, 0.15)
        
        # 综合得分
        if semantic_score > 0:
            # 有语义匹配时使用原始权重
            total_score = (semantic_score * 0.6 + 
                          min(keyword_score, 0.25) * 0.25 + 
                          column_score * 0.15) * 100
        else:
            # 无语义匹配时提高关键词和列匹配的权重
            total_score = (min(keyword_score, 0.6) * 0.7 + 
                          column_score * 0.3) * 100
        
        # 生成推理说明
        reasoning_parts = []
        if semantic_score > 0.3:
            reasoning_parts.append(f"语义匹配度高 ({semantic_score:.2f})")
        if matched_keywords:
            reasoning_parts.append(f"关键词匹配: {', '.join(matched_keywords[:3])}")
        if relevant_columns:
            col_names = [col.get('col', '') for col in relevant_columns[:2]]
            reasoning_parts.append(f"相关字段: {', '.join(col_names)}")
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "相关性较低"
        
        return TableRelevance(
            table_name=table_name,
            table_description=table_desc,
            relevance_score=total_score,
            semantic_similarity=semantic_score,
            keyword_matches=matched_keywords,
            matched_columns=relevant_columns,
            reasoning=reasoning
        )
    
    def select_tables(self, query: str, top_k: int = 5) -> Tuple[List[TableRelevance], Dict]:
        """智能选择相关表"""
        if not self.tables:
            return [], {"error": "未加载任何数据表schema"}
        
        start_time = time.perf_counter()
        
        # 分析查询意图
        intent = self.analyze_query_intent(query)
        
        # 计算每个表的相关性
        table_relevances = []
        for i, table in enumerate(self.tables):
            relevance = self.calculate_table_relevance(table, query, i)
            table_relevances.append(relevance)
        
        # 按相关性排序
        table_relevances.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # 选择top-k个表
        selected_tables = table_relevances[:top_k]
        
        # 过滤掉得分过低的表
        selected_tables = [t for t in selected_tables if t.relevance_score > 1.0]
        
        end_time = time.perf_counter()
        
        # 生成分析报告
        analysis_report = {
            "query": query,
            "intent": intent,
            "total_tables": len(self.tables),
            "selected_count": len(selected_tables),
            "processing_time_ms": (end_time - start_time) * 1000,
            "selection_reasoning": self._generate_selection_reasoning(selected_tables, intent),
            "use_semantic_matching": self.rag_engine is not None and self.table_embeddings is not None
        }
        
        return selected_tables, analysis_report
    
    def _generate_selection_reasoning(self, selected_tables: List[TableRelevance], intent: Dict) -> str:
        """生成表选择推理说明"""
        if not selected_tables:
            return "未找到相关性足够高的数据表"
        
        reasoning_parts = []
        
        # 最高相关性表
        top_table = selected_tables[0]
        reasoning_parts.append(f"最相关表: {top_table.table_name} (得分: {top_table.relevance_score:.1f})")
        
        # 语义匹配情况
        if hasattr(top_table, 'semantic_similarity') and top_table.semantic_similarity > 0.3:
            reasoning_parts.append(f"语义匹配度: {top_table.semantic_similarity:.2f}")
        
        # 查询意图
        intent_features = [k.replace('has_', '') for k, v in intent.items() if v]
        if intent_features:
            reasoning_parts.append(f"查询特征: {', '.join(intent_features)}")
        
        return "; ".join(reasoning_parts)
    
    def get_table_context(self, selected_tables: List[TableRelevance]) -> str:
        """生成选中表的上下文信息"""
        if not selected_tables:
            return "未选择任何表"
        
        context_parts = []
        
        for table_rel in selected_tables:
            table_info = f"表名: {table_rel.table_name}\n"
            table_info += f"描述: {table_rel.table_description}\n"
            table_info += f"相关性得分: {table_rel.relevance_score:.1f}\n"
            
            if hasattr(table_rel, 'semantic_similarity'):
                table_info += f"语义相似度: {table_rel.semantic_similarity:.2f}\n"
            
            if table_rel.matched_columns:
                table_info += "相关字段:\n"
                for col in table_rel.matched_columns[:5]:  # 只显示前5个相关字段
                    similarity_info = ""
                    if 'similarity' in col:
                        similarity_info = f" (相似度: {col['similarity']:.2f})"
                    table_info += f"  - {col.get('col', '')} ({col.get('type', '')}): {col.get('description', '')}{similarity_info}\n"
            
            context_parts.append(table_info)
        
        return "\n".join(context_parts)
    
    def update_schema(self, new_schema_paths: List[str]):
        """更新schema配置（用户重新上传文件时调用）"""
        self.schema_paths = new_schema_paths
        self.load_dynamic_schema()
        
        # 重新计算向量
        if self.rag_engine and self.rag_engine.model:
            self.precompute_embeddings()


# 测试函数
def test_table_selector_with_rag():
    """测试基于RAG的表选择器"""
    # 这里需要传入实际的RAG引擎实例
    print("⚠️ 测试需要在实际环境中运行，需要RAG引擎实例")


if __name__ == "__main__":
    test_table_selector_with_rag()
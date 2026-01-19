"""
历史上下文管理器
使用向量检索技术为当前对话提供相关的历史上下文信息
"""

import json
import os
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import hashlib

class HistoryContextManager:
    """历史上下文管理器"""
    
    def __init__(self, rag_engine=None):
        """
        初始化历史上下文管理器
        :param rag_engine: RAG引擎实例，用于向量化和检索
        """
        self.rag_engine = rag_engine
        self.context_cache_file = "data/history_context_cache.json"
        self.context_vectors_file = "data/history_context_vectors.npy"
        self.max_context_items = 1000  # 最大缓存的历史条目数
        self.max_retrieved_items = 5   # 最大检索返回数量
        self._ensure_cache_files()
    
    def _ensure_cache_files(self):
        """确保缓存文件存在"""
        os.makedirs("data", exist_ok=True)
        
        if not os.path.exists(self.context_cache_file):
            with open(self.context_cache_file, 'w', encoding='utf-8') as f:
                json.dump({"contexts": []}, f, indent=2)
    
    def _get_text_hash(self, text: str) -> str:
        """获取文本的哈希值，用于去重"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def add_conversation_context(self, user_query: str, assistant_response: str, 
                               session_id: str = "default"):
        """
        添加对话上下文到历史记录
        :param user_query: 用户查询
        :param assistant_response: 助手回复
        :param session_id: 会话ID
        """
        if not self.rag_engine or not user_query.strip():
            return
        
        try:
            # 创建上下文条目
            context_text = f"用户问题: {user_query}\n助手回答: {assistant_response[:500]}..."  # 限制长度
            context_hash = self._get_text_hash(context_text)
            
            # 加载现有缓存
            with open(self.context_cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # 检查是否已存在（去重）
            existing_hashes = {item.get('hash') for item in cache_data['contexts']}
            if context_hash in existing_hashes:
                return
            
            # 生成向量
            try:
                vector = self.rag_engine.get_embedding(context_text)
                if vector is None:
                    return
                
                # 确保向量是numpy数组
                if hasattr(vector, 'values'):
                    vector = vector.values
                vector = np.array(vector, dtype=np.float32)
                
            except Exception as e:
                print(f"向量化失败: {e}")
                return
            
            # 创建新的上下文条目
            context_item = {
                "hash": context_hash,
                "user_query": user_query,
                "assistant_response": assistant_response[:1000],  # 限制存储长度
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "vector_index": len(cache_data['contexts'])  # 向量在数组中的索引
            }
            
            # 添加到缓存
            cache_data['contexts'].append(context_item)
            
            # 限制缓存大小
            if len(cache_data['contexts']) > self.max_context_items:
                # 删除最旧的条目
                removed_count = len(cache_data['contexts']) - self.max_context_items
                cache_data['contexts'] = cache_data['contexts'][removed_count:]
                
                # 重新索引
                for i, item in enumerate(cache_data['contexts']):
                    item['vector_index'] = i
            
            # 保存缓存
            with open(self.context_cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            # 更新向量文件
            self._update_vectors_file(cache_data['contexts'], vector)
            
        except Exception as e:
            print(f"添加历史上下文失败: {e}")
    
    def _update_vectors_file(self, contexts: List[Dict], new_vector: np.ndarray):
        """更新向量文件"""
        try:
            # 确保new_vector是numpy数组
            if hasattr(new_vector, 'values'):
                new_vector = new_vector.values
            new_vector = np.array(new_vector, dtype=np.float32)
            
            # 加载现有向量
            if os.path.exists(self.context_vectors_file):
                existing_vectors = np.load(self.context_vectors_file, allow_pickle=False)
                # 添加新向量
                all_vectors = np.vstack([existing_vectors, new_vector.reshape(1, -1)])
            else:
                all_vectors = new_vector.reshape(1, -1)
            
            # 如果向量数量超过限制，删除最旧的
            if len(all_vectors) > self.max_context_items:
                removed_count = len(all_vectors) - self.max_context_items
                all_vectors = all_vectors[removed_count:]
            
            # 保存向量文件，确保数据类型一致
            all_vectors = all_vectors.astype(np.float32)
            np.save(self.context_vectors_file, all_vectors)
            
        except Exception as e:
            print(f"更新向量文件失败: {e}")
    
    def retrieve_relevant_context(self, current_query: str, 
                                max_results: int = None) -> List[Dict]:
        """
        检索与当前查询相关的历史上下文
        :param current_query: 当前查询
        :param max_results: 最大返回结果数
        :return: 相关的历史上下文列表
        """
        if not self.rag_engine or not current_query.strip():
            return []
        
        if max_results is None:
            max_results = self.max_retrieved_items
        
        try:
            # 加载缓存数据
            if not os.path.exists(self.context_cache_file):
                return []
            
            with open(self.context_cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            contexts = cache_data.get('contexts', [])
            if not contexts:
                return []
            
            # 加载向量数据
            if not os.path.exists(self.context_vectors_file):
                return []
            
            context_vectors = np.load(self.context_vectors_file, allow_pickle=False)
            
            # 生成当前查询的向量
            query_vector = self.rag_engine.get_embedding(current_query)
            if query_vector is None:
                return []
            
            # 确保query_vector是numpy数组
            if hasattr(query_vector, 'values'):
                query_vector = query_vector.values
            query_vector = np.array(query_vector, dtype=np.float32)
            
            # 计算相似度
            similarities = []
            for i, context in enumerate(contexts):
                if i < len(context_vectors):
                    # 计算余弦相似度
                    context_vec = context_vectors[i]
                    similarity = np.dot(query_vector, context_vec) / (
                        np.linalg.norm(query_vector) * np.linalg.norm(context_vec)
                    )
                    similarities.append((similarity, context))
                    # 添加调试信息
                    print(f"Debug: 上下文 {i} 相似度: {similarity:.4f}")
            
            # 按相似度排序并返回top-k
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            # 过滤相似度阈值（避免返回不相关的内容）
            min_similarity = -1.0  # 设为-1.0，接受所有结果（余弦相似度范围是-1到1）
            relevant_contexts = [
                context for similarity, context in similarities[:max_results]
                if similarity > min_similarity
            ]
            
            return relevant_contexts
            
        except Exception as e:
            print(f"检索历史上下文失败: {e}")
            return []
    
    def get_context_summary(self, contexts: List[Dict]) -> str:
        """
        将检索到的上下文转换为摘要文本
        :param contexts: 上下文列表
        :return: 上下文摘要文本
        """
        if not contexts:
            return ""
        
        summary_parts = []
        for i, context in enumerate(contexts, 1):
            user_query = context.get('user_query', '')[:100]  # 限制长度
            assistant_response = context.get('assistant_response', '')[:200]
            
            summary_parts.append(
                f"历史对话{i}: 用户问「{user_query}」，系统回答了关于{assistant_response[:50]}的内容"
            )
        
        return "\n".join(summary_parts)
    
    def clear_context_cache(self):
        """清空上下文缓存"""
        try:
            with open(self.context_cache_file, 'w', encoding='utf-8') as f:
                json.dump({"contexts": []}, f, indent=2)
            
            if os.path.exists(self.context_vectors_file):
                os.remove(self.context_vectors_file)
            
            print("✅ 历史上下文缓存已清空")
            
        except Exception as e:
            print(f"清空上下文缓存失败: {e}")
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        try:
            if not os.path.exists(self.context_cache_file):
                return {"total_contexts": 0, "cache_size": "0 KB"}
            
            with open(self.context_cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            total_contexts = len(cache_data.get('contexts', []))
            
            # 计算文件大小
            cache_size = os.path.getsize(self.context_cache_file)
            if os.path.exists(self.context_vectors_file):
                cache_size += os.path.getsize(self.context_vectors_file)
            
            cache_size_str = f"{cache_size / 1024:.1f} KB" if cache_size < 1024*1024 else f"{cache_size / (1024*1024):.1f} MB"
            
            return {
                "total_contexts": total_contexts,
                "cache_size": cache_size_str,
                "max_items": self.max_context_items
            }
            
        except Exception as e:
            print(f"获取缓存统计失败: {e}")
            return {"total_contexts": 0, "cache_size": "0 KB"}

# 全局实例
history_context_manager = None

def get_history_context_manager(rag_engine=None):
    """获取历史上下文管理器的全局实例"""
    global history_context_manager
    if history_context_manager is None:
        history_context_manager = HistoryContextManager(rag_engine)
    elif rag_engine is not None and history_context_manager.rag_engine is None:
        history_context_manager.rag_engine = rag_engine
    return history_context_manager
"""
记忆存储实现

提供高效的对话历史和上下文存储、检索功能。
"""

import sqlite3
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path

from .models import (
    Interaction, ContextItem, SessionContext, StorageStats,
    ContextType, StorageError, ContextMemoryError
)


class MemoryStore:
    """记忆存储类 - 负责数据的持久化存储和检索"""
    
    def __init__(self, db_path: str = "context_memory.db"):
        """
        初始化记忆存储
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self) -> None:
        """初始化数据库表结构"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 创建会话表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        last_updated TEXT NOT NULL,
                        interaction_count INTEGER DEFAULT 0,
                        current_topic TEXT,
                        active_files TEXT,  -- JSON格式存储文件列表
                        configuration TEXT,  -- JSON格式存储配置
                        metadata TEXT  -- JSON格式存储元数据
                    )
                """)
                
                # 创建交互表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS interactions (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        user_input TEXT NOT NULL,
                        agent_response TEXT NOT NULL,
                        context_used TEXT,  -- JSON格式存储使用的上下文ID列表
                        metadata TEXT,  -- JSON格式存储元数据
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                    )
                """)
                
                # 创建上下文索引表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS context_index (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        content_type TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        relevance_score REAL DEFAULT 0.0,
                        source_interaction_id TEXT,
                        keywords TEXT,  -- JSON格式存储关键词
                        metadata TEXT,  -- JSON格式存储元数据
                        FOREIGN KEY (session_id) REFERENCES sessions (session_id),
                        FOREIGN KEY (source_interaction_id) REFERENCES interactions (id)
                    )
                """)
                
                # 创建索引以提高查询性能
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions (last_updated)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_session ON interactions (session_id, timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_context_session ON context_index (session_id, timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_context_hash ON context_index (content_hash)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_context_type ON context_index (content_type)")
                
                conn.commit()
                
        except sqlite3.Error as e:
            raise StorageError(f"数据库初始化失败: {e}")
    
    def save_interaction(self, session_id: str, interaction: Interaction) -> None:
        """
        保存交互记录
        
        Args:
            session_id: 会话ID
            interaction: 交互记录
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 保存交互记录
                cursor.execute("""
                    INSERT OR REPLACE INTO interactions 
                    (id, session_id, timestamp, user_input, agent_response, context_used, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    interaction.id,
                    session_id,
                    interaction.timestamp.isoformat(),
                    interaction.user_input,
                    interaction.agent_response,
                    json.dumps(interaction.context_used),
                    json.dumps(interaction.metadata)
                ))
                
                # 更新会话的最后更新时间和交互计数
                cursor.execute("""
                    UPDATE sessions 
                    SET last_updated = ?, interaction_count = interaction_count + 1
                    WHERE session_id = ?
                """, (datetime.now().isoformat(), session_id))
                
                # 如果会话不存在，创建新会话
                if cursor.rowcount == 0:
                    self._create_session_if_not_exists(cursor, session_id)
                
                conn.commit()
                
        except sqlite3.Error as e:
            raise StorageError(f"保存交互记录失败: {e}")
    
    def get_session_history(self, session_id: str, limit: Optional[int] = None) -> List[Interaction]:
        """
        获取会话历史记录
        
        Args:
            session_id: 会话ID
            limit: 限制返回的记录数量
            
        Returns:
            交互记录列表
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT id, session_id, timestamp, user_input, agent_response, context_used, metadata
                    FROM interactions 
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                """
                
                params = [session_id]
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                interactions = []
                for row in rows:
                    interaction = Interaction(
                        id=row[0],
                        session_id=row[1],
                        timestamp=datetime.fromisoformat(row[2]),
                        user_input=row[3],
                        agent_response=row[4],
                        context_used=json.loads(row[5]) if row[5] else [],
                        metadata=json.loads(row[6]) if row[6] else {}
                    )
                    interactions.append(interaction)
                
                return interactions
                
        except sqlite3.Error as e:
            raise StorageError(f"获取会话历史失败: {e}")
    
    def search_relevant_context(self, query: str, session_id: str, limit: int = 10) -> List[ContextItem]:
        """
        搜索相关上下文
        
        Args:
            query: 搜索查询
            session_id: 会话ID
            limit: 限制返回的结果数量
            
        Returns:
            相关上下文项列表
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 简单的关键词匹配搜索（后续可以扩展为语义搜索）
                cursor.execute("""
                    SELECT id, content, content_type, timestamp, relevance_score, 
                           source_interaction_id, metadata
                    FROM context_index 
                    WHERE session_id = ? AND (
                        content LIKE ? OR 
                        keywords LIKE ?
                    )
                    ORDER BY relevance_score DESC, timestamp DESC
                    LIMIT ?
                """, (session_id, f"%{query}%", f"%{query}%", limit))
                
                rows = cursor.fetchall()
                
                context_items = []
                for row in rows:
                    context_item = ContextItem(
                        id=row[0],
                        content=row[1],
                        type=ContextType(row[2]),
                        timestamp=datetime.fromisoformat(row[3]),
                        relevance_score=row[4],
                        source_interaction_id=row[5] or "",
                        metadata=json.loads(row[6]) if row[6] else {}
                    )
                    context_items.append(context_item)
                
                return context_items
                
        except sqlite3.Error as e:
            raise StorageError(f"搜索相关上下文失败: {e}")
    
    def save_context_item(self, session_id: str, context_item: ContextItem) -> None:
        """
        保存上下文项
        
        Args:
            session_id: 会话ID
            context_item: 上下文项
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 计算内容哈希以避免重复
                content_hash = hashlib.md5(context_item.content.encode()).hexdigest()
                
                # 提取简单关键词（后续可以使用更复杂的NLP技术）
                keywords = self._extract_keywords(context_item.content)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO context_index 
                    (id, session_id, content, content_type, content_hash, timestamp, 
                     relevance_score, source_interaction_id, keywords, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    context_item.id,
                    session_id,
                    context_item.content,
                    context_item.type.value,
                    content_hash,
                    context_item.timestamp.isoformat(),
                    context_item.relevance_score,
                    context_item.source_interaction_id,
                    json.dumps(keywords),
                    json.dumps(context_item.metadata)
                ))
                
                conn.commit()
                
        except sqlite3.Error as e:
            raise StorageError(f"保存上下文项失败: {e}")
    
    def get_context_items_by_type(self, session_id: str, context_type: ContextType) -> List[ContextItem]:
        """
        按类型获取上下文项
        
        Args:
            session_id: 会话ID
            context_type: 上下文类型
            
        Returns:
            指定类型的上下文项列表
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, content, content_type, timestamp, relevance_score, 
                           source_interaction_id, metadata
                    FROM context_index 
                    WHERE session_id = ? AND content_type = ?
                    ORDER BY timestamp DESC
                """, (session_id, context_type.value))
                
                rows = cursor.fetchall()
                context_items = []
                
                for row in rows:
                    metadata = json.loads(row[6]) if row[6] else {}
                    
                    context_item = ContextItem(
                        id=row[0],
                        content=row[1],
                        type=ContextType(row[2]),
                        timestamp=datetime.fromisoformat(row[3]),
                        relevance_score=row[4],
                        source_interaction_id=row[5],
                        metadata=metadata
                    )
                    context_items.append(context_item)
                
                return context_items
                
        except sqlite3.Error as e:
            raise StorageError(f"获取上下文项失败: {e}")
    
    def save_session_context(self, session_context: SessionContext) -> None:
        """
        保存会话上下文
        
        Args:
            session_context: 会话上下文
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO sessions 
                    (session_id, created_at, last_updated, interaction_count, 
                     current_topic, active_files, configuration, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_context.session_id,
                    session_context.created_at.isoformat(),
                    session_context.last_updated.isoformat(),
                    session_context.interaction_count,
                    session_context.current_topic,
                    json.dumps(session_context.active_files),
                    json.dumps(session_context.configuration.__dict__),
                    json.dumps(session_context.metadata)
                ))
                
                conn.commit()
                
        except sqlite3.Error as e:
            raise StorageError(f"保存会话上下文失败: {e}")
    
    def get_session_context(self, session_id: str) -> Optional[SessionContext]:
        """
        获取会话上下文
        
        Args:
            session_id: 会话ID
            
        Returns:
            会话上下文或None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT session_id, created_at, last_updated, interaction_count,
                           current_topic, active_files, configuration, metadata
                    FROM sessions 
                    WHERE session_id = ?
                """, (session_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                from .models import ContextConfig
                config_data = json.loads(row[6]) if row[6] else {}
                config = ContextConfig(**config_data)
                
                session_context = SessionContext(
                    session_id=row[0],
                    created_at=datetime.fromisoformat(row[1]),
                    last_updated=datetime.fromisoformat(row[2]),
                    interaction_count=row[3],
                    current_topic=row[4],
                    active_files=json.loads(row[5]) if row[5] else [],
                    configuration=config,
                    metadata=json.loads(row[7]) if row[7] else {}
                )
                
                return session_context
                
        except sqlite3.Error as e:
            raise StorageError(f"获取会话上下文失败: {e}")
    
    def cleanup_expired_data(self, retention_days: int) -> int:
        """
        清理过期数据
        
        Args:
            retention_days: 保留天数
            
        Returns:
            清理的记录数量
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 清理过期的上下文索引
                cursor.execute("""
                    DELETE FROM context_index 
                    WHERE timestamp < ?
                """, (cutoff_date.isoformat(),))
                context_deleted = cursor.rowcount
                
                # 清理过期的交互记录
                cursor.execute("""
                    DELETE FROM interactions 
                    WHERE timestamp < ?
                """, (cutoff_date.isoformat(),))
                interactions_deleted = cursor.rowcount
                
                # 清理没有交互记录的会话
                cursor.execute("""
                    DELETE FROM sessions 
                    WHERE session_id NOT IN (
                        SELECT DISTINCT session_id FROM interactions
                    )
                """)
                sessions_deleted = cursor.rowcount
                
                conn.commit()
                
                total_deleted = context_deleted + interactions_deleted + sessions_deleted
                return total_deleted
                
        except sqlite3.Error as e:
            raise StorageError(f"清理过期数据失败: {e}")
    
    def get_storage_stats(self) -> StorageStats:
        """
        获取存储统计信息
        
        Returns:
            存储统计信息
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 获取各表的记录数
                cursor.execute("SELECT COUNT(*) FROM sessions")
                total_sessions = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM interactions")
                total_interactions = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM context_index")
                total_context_items = cursor.fetchone()[0]
                
                # 获取数据库文件大小
                db_file = Path(self.db_path)
                storage_size_mb = db_file.stat().st_size / (1024 * 1024) if db_file.exists() else 0.0
                
                stats = StorageStats(
                    total_sessions=total_sessions,
                    total_interactions=total_interactions,
                    total_context_items=total_context_items,
                    storage_size_mb=round(storage_size_mb, 2),
                    avg_retrieval_time_ms=0.0,  # 需要实际测量
                    cache_hit_rate=0.0,  # 暂未实现缓存
                    last_cleanup=None  # 需要记录最后清理时间
                )
                
                return stats
                
        except sqlite3.Error as e:
            raise StorageError(f"获取存储统计失败: {e}")
    
    def _create_session_if_not_exists(self, cursor, session_id: str) -> None:
        """如果会话不存在则创建"""
        now = datetime.now().isoformat()
        cursor.execute("""
            INSERT OR IGNORE INTO sessions 
            (session_id, created_at, last_updated, interaction_count)
            VALUES (?, ?, ?, 1)
        """, (session_id, now, now))
    
    def _extract_keywords(self, content: str) -> List[str]:
        """
        提取内容关键词（简单实现）
        
        Args:
            content: 内容文本
            
        Returns:
            关键词列表
        """
        # 简单的关键词提取：分词并过滤常见停用词
        import re
        
        # 移除标点符号并转为小写
        words = re.findall(r'\b\w+\b', content.lower())
        
        # 简单的停用词列表
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            '的', '了', '在', '是', '我', '你', '他', '她', '它', '我们', '你们', '他们',
            '这', '那', '这个', '那个', '什么', '怎么', '为什么', '哪里', '什么时候'
        }
        
        # 过滤停用词和短词
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        # 返回前10个关键词
        return keywords[:10]
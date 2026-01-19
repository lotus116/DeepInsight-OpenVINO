#!/usr/bin/env python3
"""
图表Key生成工具
用于解决 Streamlit Plotly 图表ID冲突问题
"""

import hashlib
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
import pandas as pd
import streamlit as st


@dataclass
class ChartKeyConfig:
    """图表标识符配置"""
    context: str
    chart_type: Optional[str] = None
    data_source: str = "unknown"
    position: str = "main"
    timestamp: Optional[str] = None
    
    def generate_key(self) -> str:
        """生成唯一的图表key"""
        parts = [self.context, self.position]
        
        if self.chart_type:
            parts.append(self.chart_type)
        
        if self.data_source != "unknown":
            parts.append(self.data_source[:8])  # 限制长度
        
        if self.timestamp:
            parts.append(self.timestamp)
        
        return "_".join(parts).replace(" ", "_").lower()


class ChartContext:
    """图表上下文管理器，跟踪图表的来源和位置"""
    
    def __init__(self, source: str, location: str):
        self.source = source  # 'history', 'new_query', 'sidebar'
        self.location = location  # 'main_chart', 'trend_chart', etc.
        
    def get_key_prefix(self) -> str:
        """获取key前缀"""
        return f"{self.source}_{self.location}"
    
    def render_chart(self, fig, chart_type: str = None, data_df: pd.DataFrame = None, 
                    extra_info: str = None, **kwargs):
        """在当前上下文中渲染图表"""
        try:
            # 生成数据哈希
            data_hash = None
            if data_df is not None:
                data_hash = get_data_hash(data_df)
            
            # 生成唯一key
            chart_key = generate_chart_key(
                context=self.source,
                chart_type=chart_type,
                data_hash=data_hash,
                position=self.location,
                extra_info=extra_info
            )
            
            # 渲染图表
            st.plotly_chart(fig, key=chart_key, **kwargs)
            return chart_key
            
        except Exception as e:
            # 降级策略：不使用key渲染
            print(f"警告：上下文图表渲染失败，使用无key渲染: {e}")
            st.plotly_chart(fig, **kwargs)
            return None


# 全局上下文管理器实例
class ChartContextManager:
    """全局图表上下文管理器"""
    
    def __init__(self):
        self._contexts = {}
        self._current_context = None
    
    def create_context(self, source: str, location: str) -> ChartContext:
        """创建新的图表上下文"""
        context_key = f"{source}_{location}"
        context = ChartContext(source, location)
        self._contexts[context_key] = context
        return context
    
    def get_context(self, source: str, location: str) -> ChartContext:
        """获取或创建图表上下文"""
        context_key = f"{source}_{location}"
        if context_key not in self._contexts:
            self._contexts[context_key] = ChartContext(source, location)
        return self._contexts[context_key]
    
    def set_current_context(self, source: str, location: str):
        """设置当前上下文"""
        self._current_context = self.get_context(source, location)
    
    def render_chart_in_context(self, fig, source: str, location: str, 
                               chart_type: str = None, data_df: pd.DataFrame = None,
                               extra_info: str = None, **kwargs):
        """在指定上下文中渲染图表"""
        context = self.get_context(source, location)
        return context.render_chart(fig, chart_type, data_df, extra_info, **kwargs)


# 全局实例
chart_manager = ChartContextManager()


def get_data_hash(df: pd.DataFrame) -> str:
    """生成数据的哈希值用于key生成"""
    try:
        if df is None or df.empty:
            return "empty_data"
        
        # 使用数据的形状和列名生成哈希
        shape_str = f"{df.shape[0]}x{df.shape[1]}"
        columns_str = "_".join(sorted(df.columns.astype(str)))
        
        # 取前几行数据的哈希（避免大数据集性能问题）
        sample_data = df.head(5).to_string()
        
        hash_input = f"{shape_str}_{columns_str}_{sample_data}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:8]
    
    except Exception as e:
        # 降级策略：使用时间戳
        return f"fallback_{int(time.time() * 1000) % 100000}"


def get_query_hash(query: str) -> str:
    """生成查询内容的哈希值"""
    try:
        if not query or not query.strip():
            return "empty_query"
        
        # 清理查询文本并生成哈希
        clean_query = query.strip().lower()[:100]  # 限制长度
        return hashlib.md5(clean_query.encode()).hexdigest()[:8]
    
    except Exception as e:
        return f"query_fallback_{int(time.time() * 1000) % 100000}"


def generate_chart_key(context: str, chart_type: str = None, 
                      data_hash: str = None, position: str = None,
                      extra_info: str = None) -> str:
    """
    为图表生成唯一的key标识符
    
    Args:
        context: 上下文标识 ('sidebar', 'history', 'new_query')
        chart_type: 图表类型 ('bar', 'line', 'pie', etc.)
        data_hash: 数据哈希值
        position: 位置标识 ('main', 'trend', 'secondary')
        extra_info: 额外信息（如消息索引、时间戳等）
    
    Returns:
        唯一的图表key字符串
    """
    try:
        config = ChartKeyConfig(
            context=context or "unknown",
            chart_type=chart_type,
            data_source=data_hash or "no_data",
            position=position or "main",
            timestamp=extra_info
        )
        
        base_key = config.generate_key()
        
        # 添加时间戳后缀以确保唯一性（如果需要）
        if not extra_info:
            timestamp_suffix = str(int(time.time() * 1000) % 10000)
            base_key += f"_{timestamp_suffix}"
        
        return base_key
    
    except Exception as e:
        # 最终降级策略
        fallback_key = f"chart_fallback_{int(time.time() * 1000) % 100000}"
        print(f"警告：图表key生成失败，使用降级key: {fallback_key}, 错误: {e}")
        return fallback_key


def create_chart_with_key(fig, context: str, chart_type: str = None,
                         data_df: pd.DataFrame = None, position: str = None,
                         extra_info: str = None, **kwargs) -> None:
    """
    带有自动key生成的图表渲染函数
    
    Args:
        fig: Plotly 图表对象
        context: 上下文标识
        chart_type: 图表类型
        data_df: 数据DataFrame（用于生成哈希）
        position: 位置标识
        extra_info: 额外信息
        **kwargs: 传递给 st.plotly_chart 的其他参数
    """
    try:
        # 生成数据哈希
        data_hash = None
        if data_df is not None:
            data_hash = get_data_hash(data_df)
        
        # 生成唯一key
        chart_key = generate_chart_key(
            context=context,
            chart_type=chart_type,
            data_hash=data_hash,
            position=position,
            extra_info=extra_info
        )
        
        # 渲染图表
        st.plotly_chart(fig, key=chart_key, **kwargs)
        
    except Exception as e:
        # 降级策略：不使用key渲染
        print(f"警告：带key的图表渲染失败，使用无key渲染: {e}")
        st.plotly_chart(fig, **kwargs)


# 便捷函数：为不同上下文生成key
def generate_sidebar_chart_key(chart_type: str = "trend", extra_info: str = None) -> str:
    """为侧边栏图表生成key"""
    return generate_chart_key("sidebar", chart_type, position="trend", extra_info=extra_info)


def generate_history_chart_key(msg_index: int, chart_type: str = None, 
                              data_df: pd.DataFrame = None) -> str:
    """为历史消息图表生成key"""
    data_hash = get_data_hash(data_df) if data_df is not None else None
    return generate_chart_key("history", chart_type, data_hash, "main", str(msg_index))


def generate_query_chart_key(query: str, chart_type: str = None, 
                           data_df: pd.DataFrame = None) -> str:
    """为新查询图表生成key"""
    query_hash = get_query_hash(query)
    data_hash = get_data_hash(data_df) if data_df is not None else None
    combined_hash = f"{query_hash}_{data_hash}" if data_hash else query_hash
    return generate_chart_key("query", chart_type, combined_hash, "main")


# 高级集成函数
def render_sidebar_chart(fig, chart_type: str = "performance_trend", 
                        time_range: str = None, **kwargs):
    """渲染侧边栏图表"""
    return chart_manager.render_chart_in_context(
        fig, "sidebar", "trend", chart_type, None, time_range, **kwargs
    )


def render_history_chart(fig, msg_index: int, chart_type: str = None,
                        data_df: pd.DataFrame = None, **kwargs):
    """渲染历史消息图表"""
    return chart_manager.render_chart_in_context(
        fig, "history", "main", chart_type, data_df, str(msg_index), **kwargs
    )


def render_query_chart(fig, query: str, chart_type: str = None,
                      data_df: pd.DataFrame = None, **kwargs):
    """渲染新查询图表"""
    query_hash = get_query_hash(query)
    return chart_manager.render_chart_in_context(
        fig, "query", "main", chart_type, data_df, query_hash, **kwargs
    )


# 调试和诊断函数
def diagnose_chart_keys(keys_used: list) -> Dict[str, Any]:
    """诊断图表key的使用情况，检测潜在冲突"""
    diagnosis = {
        "total_keys": len(keys_used),
        "unique_keys": len(set(keys_used)),
        "duplicates": [],
        "key_patterns": {}
    }
    
    # 检测重复key
    key_counts = {}
    for key in keys_used:
        key_counts[key] = key_counts.get(key, 0) + 1
    
    diagnosis["duplicates"] = [key for key, count in key_counts.items() if count > 1]
    
    # 分析key模式
    for key in set(keys_used):
        parts = key.split("_")
        if len(parts) > 0:
            pattern = parts[0]
            diagnosis["key_patterns"][pattern] = diagnosis["key_patterns"].get(pattern, 0) + 1
    
    return diagnosis


if __name__ == "__main__":
    # 测试key生成功能
    print("测试图表key生成功能...")
    
    # 测试不同上下文的key生成
    sidebar_key = generate_sidebar_chart_key("performance", "1h")
    print(f"侧边栏key: {sidebar_key}")
    
    history_key = generate_history_chart_key(0, "bar")
    print(f"历史消息key: {history_key}")
    
    query_key = generate_query_chart_key("显示销售数据", "line")
    print(f"查询key: {query_key}")
    
    # 测试上下文管理器
    print("\n测试上下文管理器...")
    sidebar_context = chart_manager.get_context("sidebar", "trend")
    print(f"侧边栏上下文前缀: {sidebar_context.get_key_prefix()}")
    
    history_context = chart_manager.get_context("history", "main")
    print(f"历史消息上下文前缀: {history_context.get_key_prefix()}")
    
    # 测试key唯一性
    keys = [
        generate_sidebar_chart_key("trend", "1h"),
        generate_sidebar_chart_key("trend", "3h"),
        generate_history_chart_key(0, "bar"),
        generate_history_chart_key(1, "bar"),
        generate_query_chart_key("查询1", "line"),
        generate_query_chart_key("查询2", "line"),
    ]
    
    diagnosis = diagnose_chart_keys(keys)
    print(f"\nKey诊断结果: {diagnosis}")
    
    print("✅ 图表key生成工具和上下文管理器创建完成！")
"""
Intel DeepInsight 高级数据筛选器
交互式数据筛选、排序和快速查询功能
"""
import pandas as pd
import streamlit as st
from typing import Dict, List, Any, Optional, Tuple
import json
import os
from datetime import datetime

class DataFilter:
    """高级数据筛选器"""
    
    def __init__(self):
        self.saved_filters_file = "data/saved_filters.json"
        self._ensure_filters_file()
    
    def _ensure_filters_file(self):
        """确保筛选器配置文件存在"""
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(self.saved_filters_file):
            with open(self.saved_filters_file, 'w', encoding='utf-8') as f:
                json.dump({"filters": []}, f, indent=2)
    
    def create_filter_interface(self, df: pd.DataFrame, key_prefix: str = "filter") -> Tuple[pd.DataFrame, Dict]:
        """创建交互式筛选界面"""
        if df.empty:
            return df, {}
        
        st.markdown("##### 🔍 数据筛选器")
        
        # 筛选配置存储
        filter_config = {}
        filtered_df = df.copy()
        
        # 创建筛选控件
        filter_cols = st.columns([2, 2, 1])
        
        with filter_cols[0]:
            # 列选择
            available_columns = list(df.columns)
            selected_columns = st.multiselect(
                "显示列", 
                available_columns, 
                default=available_columns[:5] if len(available_columns) > 5 else available_columns,
                key=f"{key_prefix}_columns"
            )
            
            if selected_columns:
                filtered_df = filtered_df[selected_columns]
                filter_config["selected_columns"] = selected_columns
        
        with filter_cols[1]:
            # 排序选择
            if not filtered_df.empty:
                sort_column = st.selectbox(
                    "排序列", 
                    ["不排序"] + list(filtered_df.columns),
                    key=f"{key_prefix}_sort_col"
                )
                
                if sort_column != "不排序":
                    sort_ascending = st.checkbox("升序", value=True, key=f"{key_prefix}_sort_asc")
                    filtered_df = filtered_df.sort_values(sort_column, ascending=sort_ascending)
                    filter_config["sort_column"] = sort_column
                    filter_config["sort_ascending"] = sort_ascending
        
        with filter_cols[2]:
            # 行数限制
            max_rows = st.number_input(
                "显示行数", 
                min_value=10, 
                max_value=len(df), 
                value=min(100, len(df)),
                step=10,
                key=f"{key_prefix}_max_rows"
            )
            
            if max_rows < len(filtered_df):
                filtered_df = filtered_df.head(max_rows)
                filter_config["max_rows"] = max_rows
        
        # 高级筛选
        with st.expander("🎯 高级筛选", expanded=False):
            self._create_advanced_filters(df, filtered_df, filter_config, key_prefix)
        
        return filtered_df, filter_config
    
    def _create_advanced_filters(self, original_df: pd.DataFrame, filtered_df: pd.DataFrame, 
                                filter_config: Dict, key_prefix: str):
        """创建高级筛选选项"""
        
        # 数值列筛选
        numeric_columns = original_df.select_dtypes(include=['number']).columns.tolist()
        if numeric_columns:
            st.markdown("**数值范围筛选:**")
            for col in numeric_columns[:3]:  # 限制显示前3个数值列
                if col in filtered_df.columns:
                    col_min = float(original_df[col].min())
                    col_max = float(original_df[col].max())
                    
                    if col_min != col_max:  # 避免范围为0的情况
                        range_values = st.slider(
                            f"{col} 范围",
                            min_value=col_min,
                            max_value=col_max,
                            value=(col_min, col_max),
                            key=f"{key_prefix}_range_{col}"
                        )
                        
                        if range_values != (col_min, col_max):
                            mask = (original_df[col] >= range_values[0]) & (original_df[col] <= range_values[1])
                            filtered_df = filtered_df[mask]
                            filter_config[f"range_{col}"] = range_values
        
        # 分类列筛选
        categorical_columns = original_df.select_dtypes(exclude=['number']).columns.tolist()
        if categorical_columns:
            st.markdown("**分类筛选:**")
            for col in categorical_columns[:3]:  # 限制显示前3个分类列
                if col in filtered_df.columns:
                    unique_values = original_df[col].unique().tolist()
                    if len(unique_values) <= 20:  # 只对选项不太多的列提供筛选
                        selected_values = st.multiselect(
                            f"{col} 选择",
                            unique_values,
                            default=unique_values,
                            key=f"{key_prefix}_cat_{col}"
                        )
                        
                        if len(selected_values) < len(unique_values):
                            filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
                            filter_config[f"category_{col}"] = selected_values
        
        # 文本搜索
        text_columns = [col for col in categorical_columns if original_df[col].dtype == 'object']
        if text_columns:
            st.markdown("**文本搜索:**")
            search_column = st.selectbox(
                "搜索列", 
                ["不搜索"] + text_columns,
                key=f"{key_prefix}_search_col"
            )
            
            if search_column != "不搜索":
                search_text = st.text_input(
                    "搜索内容", 
                    placeholder="输入搜索关键词...",
                    key=f"{key_prefix}_search_text"
                )
                
                if search_text:
                    mask = original_df[search_column].astype(str).str.contains(search_text, case=False, na=False)
                    filtered_df = filtered_df[mask]
                    filter_config["search"] = {"column": search_column, "text": search_text}
    
    def save_filter_config(self, config: Dict, name: str) -> bool:
        """保存筛选配置"""
        try:
            with open(self.saved_filters_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查是否已存在同名配置
            existing_names = [f["name"] for f in data["filters"]]
            if name in existing_names:
                # 更新现有配置
                for i, f in enumerate(data["filters"]):
                    if f["name"] == name:
                        data["filters"][i] = {
                            "name": name,
                            "config": config,
                            "created_at": datetime.now().isoformat(),
                            "updated_at": datetime.now().isoformat()
                        }
                        break
            else:
                # 添加新配置
                data["filters"].append({
                    "name": name,
                    "config": config,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                })
            
            with open(self.saved_filters_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"保存筛选配置失败: {e}")
            return False
    
    def load_saved_filters(self) -> List[Dict]:
        """加载已保存的筛选配置"""
        try:
            with open(self.saved_filters_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("filters", [])
        except Exception as e:
            print(f"加载筛选配置失败: {e}")
            return []
    
    def apply_saved_filter(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """应用已保存的筛选配置"""
        try:
            filtered_df = df.copy()
            
            # 应用列选择
            if "selected_columns" in config:
                available_cols = [col for col in config["selected_columns"] if col in df.columns]
                if available_cols:
                    filtered_df = filtered_df[available_cols]
            
            # 应用排序
            if "sort_column" in config and config["sort_column"] in filtered_df.columns:
                filtered_df = filtered_df.sort_values(
                    config["sort_column"], 
                    ascending=config.get("sort_ascending", True)
                )
            
            # 应用数值范围筛选
            for key, value in config.items():
                if key.startswith("range_"):
                    col_name = key.replace("range_", "")
                    if col_name in df.columns:
                        mask = (df[col_name] >= value[0]) & (df[col_name] <= value[1])
                        filtered_df = filtered_df[mask]
            
            # 应用分类筛选
            for key, value in config.items():
                if key.startswith("category_"):
                    col_name = key.replace("category_", "")
                    if col_name in df.columns:
                        filtered_df = filtered_df[filtered_df[col_name].isin(value)]
            
            # 应用文本搜索
            if "search" in config:
                search_config = config["search"]
                col_name = search_config["column"]
                search_text = search_config["text"]
                if col_name in df.columns:
                    mask = df[col_name].astype(str).str.contains(search_text, case=False, na=False)
                    filtered_df = filtered_df[mask]
            
            # 应用行数限制
            if "max_rows" in config:
                filtered_df = filtered_df.head(config["max_rows"])
            
            return filtered_df
        except Exception as e:
            print(f"应用筛选配置失败: {e}")
            return df
    
    def create_quick_filter_buttons(self, df: pd.DataFrame, key_prefix: str = "quick") -> Optional[Dict]:
        """创建快速筛选按钮"""
        if df.empty:
            return None
        
        st.markdown("##### ⚡ 快速筛选")
        
        quick_filters = []
        
        # 基于数据特征生成快速筛选
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        
        # 数值列快速筛选
        if numeric_cols:
            for col in numeric_cols[:2]:  # 前2个数值列
                if col in ['sales', 'profit', '销售额', '利润']:
                    quick_filters.extend([
                        {"name": f"高{col}", "type": "top_percent", "column": col, "percent": 20},
                        {"name": f"低{col}", "type": "bottom_percent", "column": col, "percent": 20}
                    ])
        
        # 分类列快速筛选
        if categorical_cols:
            for col in categorical_cols[:2]:  # 前2个分类列
                unique_vals = df[col].unique()
                if len(unique_vals) <= 10:  # 选项不太多的列
                    for val in unique_vals[:3]:  # 前3个值
                        quick_filters.append({
                            "name": f"{col}={val}", 
                            "type": "category_filter", 
                            "column": col, 
                            "value": val
                        })
        
        # 显示快速筛选按钮
        if quick_filters:
            cols = st.columns(min(4, len(quick_filters)))
            for i, filter_config in enumerate(quick_filters[:4]):  # 最多显示4个
                with cols[i % 4]:
                    if st.button(filter_config["name"], key=f"{key_prefix}_quick_{i}", use_container_width=True):
                        return filter_config
        
        return None
    
    def apply_quick_filter(self, df: pd.DataFrame, filter_config: Dict) -> pd.DataFrame:
        """应用快速筛选"""
        try:
            if filter_config["type"] == "top_percent":
                col = filter_config["column"]
                percent = filter_config["percent"]
                threshold = df[col].quantile(1 - percent/100)
                return df[df[col] >= threshold]
            
            elif filter_config["type"] == "bottom_percent":
                col = filter_config["column"]
                percent = filter_config["percent"]
                threshold = df[col].quantile(percent/100)
                return df[df[col] <= threshold]
            
            elif filter_config["type"] == "category_filter":
                col = filter_config["column"]
                value = filter_config["value"]
                return df[df[col] == value]
            
            return df
        except Exception as e:
            print(f"应用快速筛选失败: {e}")
            return df

# 全局实例
data_filter = DataFilter()
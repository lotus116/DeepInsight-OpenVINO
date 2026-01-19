"""
Intel DeepInsight 鲁棒可视化引擎
基于Plotly的智能图表生成和交互式可视化
重构版本 - 防错、自愈、简单优先
【已修复图表导出功能 - 支持多图表同时导出】
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import logging
from dataclasses import dataclass
from enum import Enum

# 配置日志
logger = logging.getLogger(__name__)

class DataType(Enum):
    """数据类型枚举"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    MIXED = "mixed"
    UNKNOWN = "unknown"

class ChartComplexity(Enum):
    """图表复杂度枚举"""
    SIMPLE = "simple"      # 1-2维数据
    MEDIUM = "medium"      # 3维数据
    COMPLEX = "complex"    # 4+维数据

@dataclass
class ColumnInfo:
    """列信息数据类"""
    name: str
    original_type: str
    detected_type: DataType
    semantic_role: str  # 'time', 'metric', 'category', 'identifier'
    is_time_related: bool
    sample_values: List
    null_count: int
    unique_count: int

@dataclass
class ChartMapping:
    """图表映射数据类"""
    chart_type: str
    x_axis: Optional[str]
    y_axis: Optional[str]
    color_by: Optional[str]
    title: str
    complexity: ChartComplexity
    confidence: float  # 0-1, 映射的置信度

class RobustVisualizationEngine:
    """鲁棒可视化引擎 - 防错、自愈、简单优先"""

    def __init__(self):
        self.color_palette = [
            '#0068B5', '#FF6B35', '#28A745', '#FFC107',
            '#DC3545', '#6F42C1', '#20C997', '#FD7E14'
        ]

        # 配置参数
        self.max_categories = 15  # 最大分类数量
        self.max_rows_full = 1000  # 全量处理的最大行数
        self.max_rows_sample = 10000  # 采样处理的最大行数
        self.min_confidence = 0.6  # 最小置信度阈值

        # 时间相关关键词
        self.time_keywords = {
            'en': ['year', 'month', 'day', 'date', 'time', 'timestamp', 'period'],
            'zh': ['年', '月', '日', '日期', '时间', '年份', '年度', '月份']
        }

        # 指标相关关键词
        self.metric_keywords = {
            'en': ['amount', 'count', 'sum', 'total', 'value', 'price', 'cost', 'profit', 'revenue', 'sales'],
            'zh': ['金额', '数量', '总计', '价值', '价格', '成本', '利润', '收入', '销售', '销量']
        }

        # 分类相关关键词
        self.category_keywords = {
            'en': ['region', 'category', 'type', 'group', 'class', 'segment', 'department'],
            'zh': ['地区', '类别', '类型', '分组', '分类', '部门', '区域', '城市']
        }

    def analyze_dataframe(self, df: pd.DataFrame) -> Tuple[List[ColumnInfo], ChartComplexity]:
        """
        深度分析DataFrame，返回列信息和复杂度
        这是整个系统的基础，确保数据类型识别准确
        """
        if df.empty:
            return [], ChartComplexity.SIMPLE

        columns_info = []

        for col in df.columns:
            try:
                col_info = self._analyze_single_column(df, col)
                columns_info.append(col_info)
            except Exception as e:
                logger.warning(f"分析列 {col} 时出错: {e}")
                # 创建安全的默认列信息
                columns_info.append(ColumnInfo(
                    name=col,
                    original_type=str(df[col].dtype),
                    detected_type=DataType.UNKNOWN,
                    semantic_role='unknown',
                    is_time_related=False,
                    sample_values=[],
                    null_count=df[col].isnull().sum(),
                    unique_count=0
                ))

        # 确定数据复杂度
        complexity = self._determine_complexity(columns_info, len(df))

        return columns_info, complexity

    def _analyze_single_column(self, df: pd.DataFrame, col: str) -> ColumnInfo:
        """分析单个列的详细信息"""
        series = df[col]
        original_type = str(series.dtype)

        # 获取样本值（非空）
        sample_values = series.dropna().head(10).tolist()
        null_count = series.isnull().sum()
        unique_count = series.nunique()

        # 检测真实数据类型
        detected_type = self._detect_real_data_type(series)

        # 确定语义角色
        semantic_role = self._determine_semantic_role(col, series, detected_type)

        # 检查是否为时间相关
        is_time_related = self._is_time_related(col, series)

        return ColumnInfo(
            name=col,
            original_type=original_type,
            detected_type=detected_type,
            semantic_role=semantic_role,
            is_time_related=is_time_related,
            sample_values=sample_values,
            null_count=null_count,
            unique_count=unique_count
        )

    def _detect_real_data_type(self, series: pd.Series) -> DataType:
        """检测列的真实数据类型，处理常见的类型错误"""
        if series.empty:
            return DataType.UNKNOWN

        # 移除空值进行分析
        non_null_series = series.dropna()
        if non_null_series.empty:
            return DataType.UNKNOWN

        # 如果已经是数值类型
        if pd.api.types.is_numeric_dtype(series):
            return DataType.NUMERIC

        # 如果已经是日期时间类型
        if pd.api.types.is_datetime64_any_dtype(series):
            return DataType.DATETIME

        # 对于object类型，进行深度检测
        if series.dtype == 'object':
            return self._detect_object_type(non_null_series)

        # 默认为分类类型
        return DataType.CATEGORICAL

    def _detect_object_type(self, series: pd.Series) -> DataType:
        """检测object类型列的真实类型"""
        sample_size = min(100, len(series))
        sample = series.head(sample_size)

        # 尝试转换为数值
        numeric_count = 0
        for value in sample:
            try:
                float(str(value))
                numeric_count += 1
            except (ValueError, TypeError):
                pass

        # 如果80%以上可以转换为数值，认为是数值类型
        if numeric_count / len(sample) >= 0.8:
            return DataType.NUMERIC

        # 尝试转换为日期时间
        datetime_count = 0
        for value in sample:
            try:
                pd.to_datetime(str(value))
                datetime_count += 1
            except (ValueError, TypeError):
                pass

        # 如果60%以上可以转换为日期时间，认为是日期时间类型
        if datetime_count / len(sample) >= 0.6:
            return DataType.DATETIME

        # 检查是否为混合类型
        if numeric_count > 0 and datetime_count > 0:
            return DataType.MIXED

        # 默认为分类类型
        return DataType.CATEGORICAL

    def _determine_semantic_role(self, col_name: str, series: pd.Series, data_type: DataType) -> str:
        """确定列的语义角色"""
        col_lower = col_name.lower()

        # 检查时间角色
        for lang_keywords in self.time_keywords.values():
            if any(keyword in col_lower for keyword in lang_keywords):
                return 'time'

        # 检查指标角色
        if data_type == DataType.NUMERIC:
            for lang_keywords in self.metric_keywords.values():
                if any(keyword in col_lower for keyword in lang_keywords):
                    return 'metric'

        # 检查分类角色
        for lang_keywords in self.category_keywords.values():
            if any(keyword in col_lower for keyword in lang_keywords):
                return 'category'

        # 基于数据类型的默认角色
        if data_type == DataType.NUMERIC:
            return 'metric'
        elif data_type == DataType.DATETIME:
            return 'time'
        else:
            return 'category'

    def _is_time_related(self, col_name: str, series: pd.Series) -> bool:
        """检查列是否与时间相关"""
        col_lower = col_name.lower()

        # 检查列名
        for lang_keywords in self.time_keywords.values():
            if any(keyword in col_lower for keyword in lang_keywords):
                return True

        # 检查数据内容（对于数值类型的年份）
        if pd.api.types.is_numeric_dtype(series):
            sample_values = series.dropna().head(20)
            if len(sample_values) > 0:
                # 检查是否为年份范围
                min_val, max_val = sample_values.min(), sample_values.max()
                if 1900 <= min_val <= 2100 and 1900 <= max_val <= 2100:
                    return True

        return False

    def _determine_complexity(self, columns_info: List[ColumnInfo], row_count: int) -> ChartComplexity:
        """确定数据复杂度"""
        # 统计不同类型的列数
        numeric_count = sum(1 for col in columns_info if col.detected_type == DataType.NUMERIC)
        categorical_count = sum(1 for col in columns_info if col.detected_type == DataType.CATEGORICAL)
        datetime_count = sum(1 for col in columns_info if col.detected_type == DataType.DATETIME)

        total_meaningful_cols = numeric_count + categorical_count + datetime_count

        # 基于列数和行数确定复杂度
        if total_meaningful_cols <= 2 or row_count <= 100:
            return ChartComplexity.SIMPLE
        elif total_meaningful_cols == 3 or row_count <= 1000:
            return ChartComplexity.MEDIUM
        else:
            return ChartComplexity.COMPLEX

    def create_robust_chart(self, df: pd.DataFrame, query_context: str = "") -> go.Figure:
        """
        创建鲁棒图表 - 主入口方法
        采用多层降级策略，确保始终能生成合理的图表
        """
        try:
            # 第一步：数据验证和预处理
            if df.empty:
                return self._create_empty_chart("数据为空")

            # 第二步：深度分析数据
            columns_info, complexity = self.analyze_dataframe(df)

            # 第三步：数据预处理和清洗
            processed_df = self._preprocess_data(df, columns_info, complexity)

            # 第四步：智能图表映射
            mapping = self._create_smart_mapping(processed_df, columns_info, complexity, query_context)

            # 第五步：生成图表
            if mapping.confidence >= self.min_confidence:
                return self._generate_chart_by_mapping(processed_df, mapping)
            else:
                # 置信度不足，降级到简单图表
                return self._create_fallback_chart(processed_df, columns_info)

        except Exception as e:
            logger.error(f"图表生成失败: {e}")
            return self._create_error_chart(f"图表生成遇到问题: {str(e)}")

    def _preprocess_data(self, df: pd.DataFrame, columns_info: List[ColumnInfo], complexity: ChartComplexity) -> pd.DataFrame:
        """数据预处理和清洗"""
        processed_df = df.copy()

        try:
            # 处理数据量过大的情况
            if len(processed_df) > self.max_rows_sample:
                processed_df = processed_df.sample(n=self.max_rows_sample, random_state=42)
                logger.info(f"数据量过大，已采样到 {self.max_rows_sample} 行")

            # 数据类型转换和修复
            for col_info in columns_info:
                col_name = col_info.name

                if col_info.detected_type == DataType.NUMERIC and col_info.original_type == 'object':
                    # 转换字符串数值
                    processed_df[col_name] = pd.to_numeric(processed_df[col_name], errors='coerce')

                elif col_info.detected_type == DataType.DATETIME and col_info.original_type == 'object':
                    # 转换字符串日期
                    processed_df[col_name] = pd.to_datetime(processed_df[col_name], errors='coerce')

            # 处理缺失值
            processed_df = self._handle_missing_values(processed_df, columns_info)

            # 处理异常值（仅对数值列）
            processed_df = self._handle_outliers(processed_df, columns_info)

            return processed_df

        except Exception as e:
            logger.warning(f"数据预处理失败: {e}，使用原始数据")
            return df

    def _handle_missing_values(self, df: pd.DataFrame, columns_info: List[ColumnInfo]) -> pd.DataFrame:
        """处理缺失值"""
        for col_info in columns_info:
            col_name = col_info.name

            if col_info.null_count > 0:
                if col_info.detected_type == DataType.NUMERIC:
                    # 数值列用中位数填充
                    df[col_name].fillna(df[col_name].median(), inplace=True)
                elif col_info.detected_type == DataType.CATEGORICAL:
                    # 分类列用众数填充
                    mode_value = df[col_name].mode()
                    if not mode_value.empty:
                        df[col_name].fillna(mode_value[0], inplace=True)
                    else:
                        df[col_name].fillna('未知', inplace=True)

        return df

    def _handle_outliers(self, df: pd.DataFrame, columns_info: List[ColumnInfo]) -> pd.DataFrame:
        """处理异常值（简单的IQR方法）"""
        for col_info in columns_info:
            if col_info.detected_type == DataType.NUMERIC and col_info.semantic_role == 'metric':
                col_name = col_info.name
                Q1 = df[col_name].quantile(0.25)
                Q3 = df[col_name].quantile(0.75)
                IQR = Q3 - Q1

                # 只处理极端异常值（3倍IQR）
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR

                # 将异常值替换为边界值
                df[col_name] = df[col_name].clip(lower=lower_bound, upper=upper_bound)

        return df

    def _create_smart_mapping(self, df: pd.DataFrame, columns_info: List[ColumnInfo],
                            complexity: ChartComplexity, query_context: str) -> ChartMapping:
        """创建智能图表映射"""

        # 分类列信息
        time_cols = [col for col in columns_info if col.semantic_role == 'time']
        metric_cols = [col for col in columns_info if col.semantic_role == 'metric']
        category_cols = [col for col in columns_info if col.semantic_role == 'category']

        # 根据复杂度选择策略
        if complexity == ChartComplexity.COMPLEX:
            return self._create_simple_mapping(df, time_cols, metric_cols, category_cols)
        elif complexity == ChartComplexity.MEDIUM:
            return self._create_medium_mapping(df, time_cols, metric_cols, category_cols, query_context)
        else:
            return self._create_optimal_mapping(df, time_cols, metric_cols, category_cols, query_context)

    def _create_simple_mapping(self, df: pd.DataFrame, time_cols: List[ColumnInfo],
                             metric_cols: List[ColumnInfo], category_cols: List[ColumnInfo]) -> ChartMapping:
        """创建简单映射（复杂数据的降级方案）"""

        # 优先选择最重要的列
        x_col = None
        y_col = None

        # 选择主要指标列
        if metric_cols:
            y_col = metric_cols[0].name

        # 选择主要维度列
        if time_cols:
            x_col = time_cols[0].name
        elif category_cols:
            # 选择唯一值最少的分类列（更适合可视化）
            category_cols_sorted = sorted(category_cols, key=lambda x: x.unique_count)
            x_col = category_cols_sorted[0].name

        # 确定图表类型
        if x_col and y_col:
            if time_cols and x_col in [col.name for col in time_cols]:
                chart_type = "line"
                title = f"{y_col} 趋势分析"
            else:
                chart_type = "bar"
                title = f"{y_col} 分布"
        else:
            chart_type = "table"
            title = "数据表格"

        return ChartMapping(
            chart_type=chart_type,
            x_axis=x_col,
            y_axis=y_col,
            color_by=None,
            title=title,
            complexity=ChartComplexity.SIMPLE,
            confidence=0.8
        )

    def _create_medium_mapping(self, df: pd.DataFrame, time_cols: List[ColumnInfo],
                             metric_cols: List[ColumnInfo], category_cols: List[ColumnInfo],
                             query_context: str) -> ChartMapping:
        """创建中等复杂度映射"""

        query_lower = query_context.lower()

        # 检查是否为多维趋势查询
        is_trend_query = any(word in query_lower for word in ['趋势', '变化', '年度', '月度', 'trend'])
        is_multi_dim = any(word in query_lower for word in ['地区', '类别', '分组', '每个', 'region', 'category'])

        if is_trend_query and is_multi_dim and time_cols and metric_cols and category_cols:
            # 多维趋势图
            title = self._generate_context_aware_title(time_cols[0].name, metric_cols[0].name, category_cols[0].name, query_context)
            return ChartMapping(
                chart_type="multi_line",
                x_axis=time_cols[0].name,
                y_axis=metric_cols[0].name,
                color_by=category_cols[0].name,
                title=title,
                complexity=ChartComplexity.MEDIUM,
                confidence=0.9
            )

        # 降级到简单映射
        return self._create_simple_mapping(df, time_cols, metric_cols, category_cols)

    def _generate_context_aware_title(self, x_col: str, y_col: str, color_by: str, query_context: str) -> str:
        """基于查询上下文生成智能标题"""
        if not query_context:
            return f"{y_col} 按 {color_by} 的 {x_col} 趋势"

        query_lower = query_context.lower()

        # 提取关键信息
        if '每个地区' in query_context or '各地区' in query_context:
            if '年度' in query_context or '年份' in query_context:
                return f"各地区年度{y_col}趋势对比"
            elif '月度' in query_context:
                return f"各地区月度{y_col}趋势对比"
            else:
                return f"各地区{y_col}趋势对比"

        elif '每个类别' in query_context or '各类别' in query_context:
            return f"各类别{y_col}趋势对比"

        elif '趋势' in query_context:
            if color_by:
                return f"{y_col}按{color_by}的趋势分析"
            else:
                return f"{y_col}趋势分析"

        # 默认标题
        return f"{y_col} 按 {color_by} 的 {x_col} 分析"

    def _create_optimal_mapping(self, df: pd.DataFrame, time_cols: List[ColumnInfo],
                              metric_cols: List[ColumnInfo], category_cols: List[ColumnInfo],
                              query_context: str) -> ChartMapping:
        """创建最优映射（简单数据的完整方案）"""

        # 对于简单数据，可以使用更复杂的逻辑
        return self._create_medium_mapping(df, time_cols, metric_cols, category_cols, query_context)

    def _generate_chart_by_mapping(self, df: pd.DataFrame, mapping: ChartMapping) -> go.Figure:
        """根据映射生成图表"""
        try:
            if mapping.chart_type == "line":
                return self._create_line_chart_by_mapping(df, mapping)
            elif mapping.chart_type == "multi_line":
                return self._create_multi_line_chart_by_mapping(df, mapping)
            elif mapping.chart_type == "bar":
                return self._create_bar_chart_by_mapping(df, mapping)
            elif mapping.chart_type == "pie":
                return self._create_pie_chart_by_mapping(df, mapping)
            elif mapping.chart_type == "scatter":
                return self._create_scatter_chart_by_mapping(df, mapping)
            elif mapping.chart_type == "table":
                return self._create_table_display(df)
            else:
                return self._create_fallback_chart(df, [])
        except Exception as e:
            logger.error(f"图表生成失败: {e}")
            return self._create_fallback_chart(df, [])

    def _create_line_chart_by_mapping(self, df: pd.DataFrame, mapping: ChartMapping) -> go.Figure:
        """根据映射创建折线图"""
        if not mapping.x_axis or not mapping.y_axis:
            return self._create_fallback_chart(df, [])

        try:
            # 数据聚合
            df_agg = df.groupby(mapping.x_axis)[mapping.y_axis].sum().reset_index()

            fig = px.line(df_agg, x=mapping.x_axis, y=mapping.y_axis,
                         title=mapping.title,
                         color_discrete_sequence=self.color_palette)

            fig.update_traces(
                mode='lines+markers',
                hovertemplate=f'<b>%{{x}}</b><br>{mapping.y_axis}: %{{y:,.0f}}<extra></extra>'
            )

            fig.update_layout(
                hovermode='x unified',
                showlegend=False,
                height=400,
                xaxis_title=mapping.x_axis,
                yaxis_title=mapping.y_axis
            )

            return fig
        except Exception as e:
            logger.error(f"折线图创建失败: {e}")
            return self._create_fallback_chart(df, [])

    def _create_multi_line_chart_by_mapping(self, df: pd.DataFrame, mapping: ChartMapping) -> go.Figure:
        """根据映射创建多系列折线图"""
        if not mapping.x_axis or not mapping.y_axis or not mapping.color_by:
            return self._create_line_chart_by_mapping(df, mapping)

        try:
            # 数据聚合
            df_agg = df.groupby([mapping.x_axis, mapping.color_by])[mapping.y_axis].sum().reset_index()

            fig = px.line(df_agg, x=mapping.x_axis, y=mapping.y_axis, color=mapping.color_by,
                         title=mapping.title,
                         color_discrete_sequence=self.color_palette)

            fig.update_traces(
                mode='lines+markers',
                hovertemplate='<b>%{fullData.name}</b><br>%{x}: %{y:,.0f}<extra></extra>'
            )

            fig.update_layout(
                hovermode='x unified',
                showlegend=True,
                height=400,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis_title=mapping.x_axis,
                yaxis_title=mapping.y_axis
            )

            return fig
        except Exception as e:
            logger.error(f"多系列折线图创建失败: {e}")
            return self._create_line_chart_by_mapping(df, mapping)

    def _create_bar_chart_by_mapping(self, df: pd.DataFrame, mapping: ChartMapping) -> go.Figure:
        """根据映射创建柱状图"""
        if not mapping.x_axis or not mapping.y_axis:
            return self._create_fallback_chart(df, [])

        try:
            # 数据聚合
            df_agg = df.groupby(mapping.x_axis)[mapping.y_axis].sum().reset_index()

            # 如果数据太多，只显示前15个
            if len(df_agg) > 15:
                df_plot = df_agg.nlargest(15, mapping.y_axis)
            else:
                df_plot = df_agg

            fig = px.bar(df_plot, x=mapping.x_axis, y=mapping.y_axis,
                        title=mapping.title,
                        color=mapping.y_axis,
                        color_continuous_scale='Blues')

            fig.update_traces(
                hovertemplate=f'<b>%{{x}}</b><br>{mapping.y_axis}: %{{y:,.0f}}<extra></extra>'
            )

            fig.update_layout(
                xaxis_tickangle=-45,
                height=400,
                xaxis_title=mapping.x_axis,
                yaxis_title=mapping.y_axis
            )

            return fig
        except Exception as e:
            logger.error(f"柱状图创建失败: {e}")
            return self._create_fallback_chart(df, [])

    def _create_pie_chart_by_mapping(self, df: pd.DataFrame, mapping: ChartMapping) -> go.Figure:
        """根据映射创建饼图"""
        if not mapping.x_axis or not mapping.y_axis:
            return self._create_fallback_chart(df, [])

        try:
            # 数据聚合
            df_agg = df.groupby(mapping.x_axis)[mapping.y_axis].sum().reset_index()

            # 如果类别太多，合并小的类别
            if len(df_agg) > 8:
                df_sorted = df_agg.nlargest(7, mapping.y_axis)
                others_sum = df_agg[~df_agg.index.isin(df_sorted.index)][mapping.y_axis].sum()
                if others_sum > 0:
                    others_row = pd.DataFrame({mapping.x_axis: ['其他'], mapping.y_axis: [others_sum]})
                    df_plot = pd.concat([df_sorted, others_row], ignore_index=True)
                else:
                    df_plot = df_sorted
            else:
                df_plot = df_agg

            fig = px.pie(df_plot, names=mapping.x_axis, values=mapping.y_axis,
                        title=mapping.title,
                        color_discrete_sequence=self.color_palette)

            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate=f'<b>%{{label}}</b><br>{mapping.y_axis}: %{{value:,.0f}}<br>占比: %{{percent}}<extra></extra>'
            )

            fig.update_layout(height=400)

            return fig
        except Exception as e:
            logger.error(f"饼图创建失败: {e}")
            return self._create_fallback_chart(df, [])

    def _create_scatter_chart_by_mapping(self, df: pd.DataFrame, mapping: ChartMapping) -> go.Figure:
        """根据映射创建散点图"""
        if not mapping.x_axis or not mapping.y_axis:
            return self._create_fallback_chart(df, [])

        try:
            if mapping.color_by:
                fig = px.scatter(df, x=mapping.x_axis, y=mapping.y_axis, color=mapping.color_by,
                               title=mapping.title,
                               color_discrete_sequence=self.color_palette)
            else:
                fig = px.scatter(df, x=mapping.x_axis, y=mapping.y_axis,
                               title=mapping.title,
                               color_discrete_sequence=self.color_palette)

            fig.update_traces(
                hovertemplate='<b>%{x:,.0f}, %{y:,.0f}</b><extra></extra>'
            )

            fig.update_layout(
                height=400,
                xaxis_title=mapping.x_axis,
                yaxis_title=mapping.y_axis
            )

            return fig
        except Exception as e:
            logger.error(f"散点图创建失败: {e}")
            return self._create_fallback_chart(df, [])

    def _create_fallback_chart(self, df: pd.DataFrame, columns_info: List[ColumnInfo]) -> go.Figure:
        """创建降级图表"""
        try:
            # 尝试创建最简单的柱状图
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                fig = go.Figure(data=[go.Bar(x=df.index, y=df[col],
                                           marker_color=self.color_palette[0])])
                fig.update_layout(
                    title=f"{col} 数据展示（简化版）",
                    height=400,
                    xaxis_title="索引",
                    yaxis_title=col
                )
                return fig
            else:
                return self._create_error_chart("数据不包含数值列，无法生成图表")
        except Exception as e:
            return self._create_error_chart(f"降级图表生成失败: {str(e)}")

    def _create_error_chart(self, error_message: str) -> go.Figure:
        """创建错误提示图表"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"⚠️ {error_message}<br><br>💡 建议：检查数据格式或使用表格模式查看",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=14, color="#dc3545"),
            bgcolor="rgba(248, 215, 218, 0.8)",
            bordercolor="#dc3545",
            borderwidth=1
        )
        fig.update_layout(
            height=300,
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    def _create_empty_chart(self, message: str = "暂无数据可视化") -> go.Figure:
        """创建空图表"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"📊 {message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="#6c757d")
        )
        fig.update_layout(
            height=300,
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    def create_interactive_chart(self, df: pd.DataFrame, chart_type: str = None,
                                query_context: str = "", highlight_anomalies: bool = True) -> go.Figure:
        """创建交互式图表"""
        if df.empty:
            return self._create_empty_chart()

        if chart_type is None:
            chart_type = self.detect_chart_type(df, query_context)

        try:
            if chart_type == "table":
                return self._create_table_display(df)
            elif chart_type == "line":
                return self._create_line_chart(df, query_context)
            elif chart_type == "multi_line":
                return self._create_multi_line_chart(df, query_context)
            elif chart_type == "bar":
                return self._create_bar_chart(df, query_context)
            elif chart_type == "pie":
                return self._create_pie_chart(df)
            elif chart_type == "scatter":
                return self._create_scatter_chart(df)
            elif chart_type == "map":
                return self._create_map_chart(df)
            else:
                return self._create_bar_chart(df, query_context)  # 默认柱状图
        except Exception as e:
            st.warning(f"图表生成遇到问题: {str(e)}，已切换到简单显示")
            return self._create_simple_chart(df)

    def _create_table_display(self, df: pd.DataFrame) -> go.Figure:
        """创建表格显示（实际返回一个提示图表）"""
        fig = go.Figure()
        fig.add_annotation(
            text="📊 数据表格模式<br>请查看下方的数据表格",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16, color="#666666")
        )
        fig.update_layout(
            height=200,
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    def _create_multi_line_chart(self, df: pd.DataFrame, query_context: str = "") -> go.Figure:
        """创建多系列折线图（用于多维趋势分析）"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        # 特殊处理：检查是否有数值型的年份列被错误分类
        for col in categorical_cols.copy():
            if col.lower() in ['year', '年份', '年度']:
                try:
                    # 尝试转换为数值
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if not df[col].isna().all():
                        numeric_cols.append(col)
                        categorical_cols.remove(col)
                except:
                    pass

        if len(numeric_cols) == 0 or len(categorical_cols) < 1:
            return self._create_line_chart(df, query_context)

        # 智能识别时间列、分组列和数值列
        time_col = None
        group_col = None
        value_col = None

        # 寻找数值型时间列（如年份）
        for col in numeric_cols:
            col_lower = col.lower()
            if any(word in col_lower for word in ['year', 'month', '年', '月', '年份', '年度']):
                time_col = col
                break

        # 如果没有数值型时间列，寻找分类型时间列
        if not time_col:
            for col in categorical_cols:
                col_lower = col.lower()
                if any(word in col_lower for word in ['year', 'month', 'date', 'time', '年', '月', '日期', '时间']):
                    time_col = col
                    break

        # 寻找数值列作为值（排除时间列）
        for col in numeric_cols:
            if col != time_col:
                col_lower = col.lower()
                if not any(word in col_lower for word in ['year', 'month', '年', '月', '年份', '年度']):
                    value_col = col
                    break

        # 如果没有找到合适的数值列，使用第一个非时间数值列
        if not value_col:
            for col in numeric_cols:
                if col != time_col:
                    value_col = col
                    break

        # 寻找分组列（非时间列）
        for col in categorical_cols:
            if col != time_col:
                col_lower = col.lower()
                if any(word in col_lower for word in ['region', 'category', 'type', 'group', '地区', '类别', '分组', '区域']):
                    group_col = col
                    break

        # 如果没有找到明确的分组列，使用第一个非时间分类列
        if not group_col:
            for col in categorical_cols:
                if col != time_col:
                    group_col = col
                    break

        # 如果没有时间列，使用第一个分类列作为x轴
        if not time_col:
            time_col = categorical_cols[0] if categorical_cols else None
            if len(categorical_cols) > 1:
                group_col = categorical_cols[1]

        # 检查必要的列是否存在
        if not time_col or not value_col:
            return self._create_line_chart(df, query_context)

        # 数据预处理：确保数据正确聚合
        try:
            if group_col:
                # 按时间和分组聚合数据
                df_agg = df.groupby([time_col, group_col])[value_col].sum().reset_index()

                # 创建多系列折线图
                fig = px.line(df_agg, x=time_col, y=value_col, color=group_col,
                             title=f"{value_col} 按 {group_col} 的 {time_col} 趋势",
                             color_discrete_sequence=self.color_palette)
            else:
                # 只有时间和数值，创建单系列折线图
                df_agg = df.groupby(time_col)[value_col].sum().reset_index()
                fig = px.line(df_agg, x=time_col, y=value_col,
                             title=f"{value_col} 按 {time_col} 的趋势",
                             color_discrete_sequence=self.color_palette)

            fig.update_traces(
                mode='lines+markers',
                hovertemplate='<b>%{fullData.name}</b><br>%{x}: %{y:,.0f}<extra></extra>'
            )

            fig.update_layout(
                hovermode='x unified',
                showlegend=True if group_col else False,
                height=400,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ) if group_col else None,
                xaxis_title=time_col,
                yaxis_title=value_col
            )

            return fig

        except Exception as e:
            # 如果聚合失败，降级到简单折线图
            return self._create_line_chart(df, query_context)

    def _create_line_chart(self, df: pd.DataFrame, query_context: str = "") -> go.Figure:
        """创建折线图 - 改进版"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        # 特殊处理：检查是否有数值型的年份列被错误分类
        for col in categorical_cols.copy():
            if col.lower() in ['year', '年份', '年度']:
                try:
                    # 尝试转换为数值
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if not df[col].isna().all():
                        numeric_cols.append(col)
                        categorical_cols.remove(col)
                except:
                    pass

        if len(numeric_cols) == 0:
            return self._create_bar_chart(df, query_context)

        # 智能选择x轴和y轴
        x_col = None
        y_col = None

        # 寻找数值型的时间列作为x轴（如年份）
        for col in numeric_cols:
            col_lower = col.lower()
            if any(word in col_lower for word in ['year', 'month', '年', '月', '年份', '年度']):
                x_col = col
                # 选择另一个数值列作为y轴
                for other_col in numeric_cols:
                    if other_col != x_col:
                        y_col = other_col
                        break
                break

        # 如果没有找到数值型时间列，寻找分类型时间列
        if not x_col:
            for col in categorical_cols:
                col_lower = col.lower()
                if any(word in col_lower for word in ['year', 'month', 'date', 'time', '年', '月', '日期', '时间']):
                    x_col = col
                    y_col = numeric_cols[0]
                    break

        # 如果还是没有时间列，使用第一个分类列作为x轴
        if not x_col and len(categorical_cols) > 0:
            x_col = categorical_cols[0]
            y_col = numeric_cols[0]

        # 最后的降级方案：使用索引
        if not x_col:
            x_col = "索引"
            y_col = numeric_cols[0]
            df_plot = df.copy()
            df_plot[x_col] = df_plot.index
        else:
            df_plot = df.copy()

        # 数据聚合：如果有重复的x值，进行聚合
        try:
            if x_col in df_plot.columns and y_col:
                df_agg = df_plot.groupby(x_col)[y_col].sum().reset_index()
            else:
                df_agg = df_plot

            fig = px.line(df_agg, x=x_col, y=y_col,
                         title=f"{y_col} 按 {x_col} 的趋势分析",
                         color_discrete_sequence=self.color_palette)

            fig.update_traces(
                mode='lines+markers',
                hovertemplate='<b>%{x}</b><br>%{y:,.0f}<extra></extra>'
            )

            fig.update_layout(
                hovermode='x unified',
                showlegend=False,
                height=400,
                xaxis_title=x_col,
                yaxis_title=y_col
            )

            return fig

        except Exception as e:
            return self._create_simple_chart(df)

    def _create_bar_chart(self, df: pd.DataFrame, query_context: str = "") -> go.Figure:
        """创建柱状图 - 改进版"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        # 特殊处理：检查是否有数值型的年份列被错误分类
        for col in categorical_cols.copy():
            if col.lower() in ['year', '年份', '年度']:
                try:
                    # 尝试转换为数值
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if not df[col].isna().all():
                        numeric_cols.append(col)
                        categorical_cols.remove(col)
                except:
                    pass

        if len(numeric_cols) == 0 or len(categorical_cols) == 0:
            return self._create_simple_chart(df)

        # 智能选择x轴和y轴
        x_col = None
        y_col = None

        # 寻找合适的数值列作为y轴（排除年份等时间列）
        for col in numeric_cols:
            col_lower = col.lower()
            if not any(word in col_lower for word in ['year', 'month', '年', '月', '年份', '年度']):
                y_col = col
                break

        # 如果没有找到合适的数值列，使用第一个数值列
        if not y_col:
            y_col = numeric_cols[0]

        # 寻找合适的分类列作为x轴（优先选择非时间列）
        for col in categorical_cols:
            col_lower = col.lower()
            if not any(word in col_lower for word in ['year', 'month', 'date', 'time', '年', '月', '日期', '时间']):
                x_col = col
                break

        # 如果没有找到合适的分类列，使用第一个分类列
        if not x_col and len(categorical_cols) > 0:
            x_col = categorical_cols[0]

        # 最后的降级方案
        if not x_col:
            return self._create_simple_chart(df)

        # 数据聚合：按x轴分组聚合y轴数据
        try:
            df_agg = df.groupby(x_col)[y_col].sum().reset_index()

            # 如果数据太多，只显示前15个
            if len(df_agg) > 15:
                df_plot = df_agg.nlargest(15, y_col)
            else:
                df_plot = df_agg

            fig = px.bar(df_plot, x=x_col, y=y_col,
                        title=f"{y_col} 按 {x_col} 分布",
                        color=y_col,
                        color_continuous_scale='Blues')

            fig.update_traces(
                hovertemplate='<b>%{x}</b><br>%{y:,.0f}<extra></extra>'
            )

            fig.update_layout(
                xaxis_tickangle=-45,
                height=400,
                showlegend=False,
                xaxis_title=x_col,
                yaxis_title=y_col
            )

            return fig

        except Exception as e:
            return self._create_simple_chart(df)

    def _create_pie_chart(self, df: pd.DataFrame, query_context: str = "") -> go.Figure:
        """创建饼图 - 改进版"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns

        if len(numeric_cols) == 0 or len(categorical_cols) == 0:
            return self._create_bar_chart(df, query_context)

        # 智能选择标签列和数值列
        labels_col = categorical_cols[0]
        values_col = numeric_cols[0]

        # 优先选择非时间列作为标签
        for col in categorical_cols:
            col_lower = col.lower()
            if not any(word in col_lower for word in ['year', 'month', 'date', 'time', '年', '月', '日期', '时间']):
                labels_col = col
                break

        try:
            # 数据聚合：按标签分组聚合数值
            df_agg = df.groupby(labels_col)[values_col].sum().reset_index()

            # 如果类别太多，合并小的类别
            if len(df_agg) > 8:
                df_sorted = df_agg.nlargest(7, values_col)
                others_sum = df_agg[~df_agg.index.isin(df_sorted.index)][values_col].sum()
                if others_sum > 0:
                    others_row = pd.DataFrame({labels_col: ['其他'], values_col: [others_sum]})
                    df_plot = pd.concat([df_sorted, others_row], ignore_index=True)
                else:
                    df_plot = df_sorted
            else:
                df_plot = df_agg

            fig = px.pie(df_plot, names=labels_col, values=values_col,
                        title=f"{values_col} 按 {labels_col} 分布占比",
                        color_discrete_sequence=self.color_palette)

            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate=f'<b>%{{label}}</b><br>{values_col}: %{{value:,.0f}}<br>占比: %{{percent}}<extra></extra>'
            )

            fig.update_layout(height=400)

            return fig

        except Exception as e:
            return self._create_simple_chart(df)

    def _create_scatter_chart(self, df: pd.DataFrame, query_context: str = "") -> go.Figure:
        """创建散点图 - 改进版"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns

        if len(numeric_cols) < 2:
            return self._create_bar_chart(df, query_context)

        # 智能选择x轴和y轴
        x_col = numeric_cols[0]
        y_col = numeric_cols[1]

        # 寻找时间相关列作为x轴（如果是数值型的年份）
        for col in numeric_cols:
            col_lower = col.lower()
            if any(word in col_lower for word in ['year', 'month', '年', '月']):
                x_col = col
                # 选择另一个数值列作为y轴
                for other_col in numeric_cols:
                    if other_col != x_col:
                        y_col = other_col
                        break
                break

        # 如果有分类列，可以用作颜色分组
        color_col = None
        if len(categorical_cols) > 0:
            # 优先选择地区、类别等分组列
            for col in categorical_cols:
                col_lower = col.lower()
                if any(word in col_lower for word in ['region', 'category', 'type', 'group', '地区', '类别', '分组', '区域']):
                    color_col = col
                    break

            # 如果没有找到，使用第一个分类列
            if not color_col:
                color_col = categorical_cols[0]

        try:
            if color_col:
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                               title=f"{y_col} vs {x_col} 关系分析（按{color_col}分组）",
                               color_discrete_sequence=self.color_palette)
            else:
                fig = px.scatter(df, x=x_col, y=y_col,
                               title=f"{y_col} vs {x_col} 关系分析",
                               color_discrete_sequence=self.color_palette)

            fig.update_traces(
                hovertemplate='<b>%{x:,.0f}, %{y:,.0f}</b><extra></extra>'
            )

            fig.update_layout(
                height=400,
                xaxis_title=x_col,
                yaxis_title=y_col
            )

            return fig

        except Exception as e:
            return self._create_simple_chart(df)

    def _create_map_chart(self, df: pd.DataFrame) -> go.Figure:
        """创建地图（简化版，实际需要地理数据）"""
        # 这里简化处理，实际应该根据地理数据创建地图
        return self._create_bar_chart(df)

    def _create_simple_chart(self, df: pd.DataFrame) -> go.Figure:
        """创建简单图表作为降级方案"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            fig = go.Figure(data=[go.Bar(x=df.index, y=df[col],
                                       marker_color=self.color_palette[0])])
            fig.update_layout(title=f"{col} 数据展示", height=400)
            return fig
        else:
            return self._create_empty_chart()

    def get_chart_options(self, df: pd.DataFrame, query_context: str = "") -> List[Dict]:
        """获取可用的图表选项 - 改进版"""
        if df.empty:
            return [{"type": "table", "name": "数据表格", "icon": "📊"}]

        options = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns

        # 检测时间相关列
        time_related_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(word in col_lower for word in ['year', 'month', 'date', 'time', '年', '月', '日期', '时间']):
                time_related_cols.append(col)

        # 基础表格选项
        options.append({"type": "table", "name": "数据表格", "icon": "📊"})

        # 根据数据特征和查询上下文智能推荐
        query_lower = query_context.lower()

        # 多维趋势图（优先推荐）
        if len(time_related_cols) > 0 and len(categorical_cols) > 1 and len(numeric_cols) > 0:
            if any(word in query_lower for word in ['趋势', '变化', '年度', '月度']) and \
               any(word in query_lower for word in ['地区', '城市', '类别', '分组', '每个']):
                options.append({"type": "multi_line", "name": "多维趋势图", "icon": "📈"})

        # 柱状图
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            options.append({"type": "bar", "name": "柱状图", "icon": "📊"})

        # 折线图
        if len(numeric_cols) > 0 and (len(time_related_cols) > 0 or len(categorical_cols) > 0):
            options.append({"type": "line", "name": "折线图", "icon": "📈"})

        # 饼图（适合小数据集）
        if len(categorical_cols) > 0 and len(numeric_cols) > 0 and len(df) <= 10:
            options.append({"type": "pie", "name": "饼图", "icon": "🥧"})

        # 散点图
        if len(numeric_cols) >= 2:
            options.append({"type": "scatter", "name": "散点图", "icon": "⚪"})

        return options

    def get_chart_export_data(self, df: pd.DataFrame, chart_type: str = None, query_context: str = "") -> List[Dict]:
        """
        获取用于导出的图表数据 - 【关键修复版】
        
        改进点：
        1. 移除互斥的 if/elif 逻辑，允许同时生成多种适合的图表。
        2. 增加数据聚合逻辑，防止导出数据过大导致图表混乱。
        3. 确保所有可能的图表类型（Bar, Pie, Line, Scatter）只要数据支持都会生成。
        """
        if df.empty:
            return []

        charts = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        try:
            # --- 1. 生成柱状图 (Bar Chart) ---
            # 条件：至少1个分类列 + 1个数值列
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                x_col = categorical_cols[0]
                y_col = numeric_cols[0]
                
                # 数据聚合：防止数据过多，取Top 15
                try:
                    df_agg = df.groupby(x_col)[y_col].sum().reset_index()
                    if len(df_agg) > 15:
                        df_plot = df_agg.nlargest(15, y_col)
                    else:
                        df_plot = df_agg
                    
                    charts.append({
                        "type": "bar",
                        "title": f"{y_col} 按 {x_col} 分布",
                        "data": {
                            "x": df_plot[x_col].tolist(),
                            "y": df_plot[y_col].tolist(),
                            "name": y_col
                        }
                    })
                except Exception as e:
                    logger.warning(f"导出数据-柱状图生成失败: {e}")

            # --- 2. 生成折线图 (Line Chart) ---
            # 条件：至少1个数值列 (最好有时间轴)
            if len(numeric_cols) > 0:
                x_col = None
                y_col = None

                # 优先寻找时间相关的列作为X轴
                time_keywords = ['year', 'month', 'date', 'time', '年', '月', '日']
                
                # 策略A: 数值列中的时间 (如Year)
                for col in numeric_cols:
                    if any(k in col.lower() for k in time_keywords):
                        x_col = col
                        # 找另一个数值列作为Y
                        for potential_y in numeric_cols:
                            if potential_y != x_col:
                                y_col = potential_y
                                break
                        break
                
                # 策略B: 分类列中的时间
                if not x_col:
                    for col in categorical_cols:
                        if any(k in col.lower() for k in time_keywords):
                            x_col = col
                            y_col = numeric_cols[0]
                            break
                
                # 策略C: 没有任何时间列，但需要折线图，使用索引或第一分类列
                if not x_col:
                    # 只有在确实有"趋势"需求或者数据适合时才强行生成
                    if len(categorical_cols) > 0:
                        x_col = categorical_cols[0]
                        y_col = numeric_cols[0]
                    else:
                        # 纯数值序列
                        x_col = "索引"
                        y_col = numeric_cols[0]

                if x_col and y_col:
                    try:
                        # 对于折线图，如果X轴不是索引，最好聚合一下，防止重复X值导致连线混乱
                        if x_col != "索引":
                            df_plot = df.groupby(x_col)[y_col].sum().reset_index()
                            # 简单的排序确保线条顺畅
                            df_plot = df_plot.sort_values(by=x_col)
                            x_vals = df_plot[x_col].tolist()
                            y_vals = df_plot[y_col].tolist()
                        else:
                            x_vals = list(range(len(df)))
                            y_vals = df[y_col].tolist()

                        charts.append({
                            "type": "line",
                            "title": f"{y_col} 趋势图",
                            "data": {
                                "x": x_vals,
                                "y": y_vals,
                                "name": y_col
                            }
                        })
                    except Exception as e:
                        logger.warning(f"导出数据-折线图生成失败: {e}")

            # --- 3. 生成饼图 (Pie Chart) ---
            # 条件：分类+数值，且数据行数不能太多(或聚合后不多)
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                x_col = categorical_cols[0]
                y_col = numeric_cols[0]
                
                try:
                    df_agg = df.groupby(x_col)[y_col].sum().reset_index()
                    
                    # 饼图特殊逻辑：Top 7 + Others
                    if len(df_agg) > 8:
                        df_sorted = df_agg.nlargest(7, y_col)
                        others_sum = df_agg[~df_agg.index.isin(df_sorted.index)][y_col].sum()
                        if others_sum > 0:
                            others_row = pd.DataFrame({x_col: ['其他'], y_col: [others_sum]})
                            df_plot = pd.concat([df_sorted, others_row], ignore_index=True)
                        else:
                            df_plot = df_sorted
                    else:
                        df_plot = df_agg

                    charts.append({
                        "type": "pie",
                        "title": f"{y_col} 占比分布",
                        "data": {
                            "labels": df_plot[x_col].tolist(),
                            "values": df_plot[y_col].tolist()
                        }
                    })
                except Exception as e:
                    logger.warning(f"导出数据-饼图生成失败: {e}")

            # --- 4. 生成散点图 (Scatter Chart) ---
            # 条件：至少2个数值列
            if len(numeric_cols) >= 2:
                try:
                    x_col = numeric_cols[0]
                    y_col = numeric_cols[1]
                    
                    # 散点图通常不需要聚合，但如果点太多(>500)可以采样
                    if len(df) > 500:
                        df_sample = df.sample(500)
                    else:
                        df_sample = df

                    charts.append({
                        "type": "scatter",
                        "title": f"{x_col} vs {y_col} 散点图",
                        "data": {
                            "x": df_sample[x_col].tolist(),
                            "y": df_sample[y_col].tolist(),
                            "name": f"{x_col} vs {y_col}"
                        }
                    })
                except Exception as e:
                    logger.warning(f"导出数据-散点图生成失败: {e}")

        except Exception as e:
            logger.error(f"生成图表导出数据失败: {e}")
            return []

        # 去重逻辑：虽然我们生成了不同Type，但如果Bar和Line数据完全一样，可能显得冗余
        # 但既然用户要求"所有图"，我们这里不做强去重，只做基本的空检查
        return [c for c in charts if c.get("data", {}).get("y")]

    def detect_chart_type(self, df: pd.DataFrame, query_context: str = "") -> str:
        """智能检测最适合的图表类型"""
        if df.empty:
            return "table"

        # 分析数据结构
        columns_info, complexity = self.analyze_dataframe(df)

        # 创建智能映射
        mapping = self._create_smart_mapping(df, columns_info, complexity, query_context)

        return mapping.chart_type

    def get_chart_options_cached(self, df_shape: Tuple, columns: List[str]) -> List[Dict]:
        """缓存版本的图表选项获取（用于性能优化）"""
        # 创建一个简化的DataFrame用于分析
        sample_df = pd.DataFrame({col: [0] * min(10, df_shape[0]) for col in columns})
        return self.get_chart_options(sample_df)

# 创建兼容的旧版本可视化引擎类
class VisualizationEngine:
    """兼容旧版本的可视化引擎包装器"""

    def __init__(self):
        self.robust_engine = RobustVisualizationEngine()
        self.color_palette = self.robust_engine.color_palette

    def create_interactive_chart(self, df: pd.DataFrame, chart_type: str = None,
                                query_context: str = "", highlight_anomalies: bool = True) -> go.Figure:
        """创建交互式图表 - 兼容旧版本接口"""
        if chart_type is None:
            # 使用新的鲁棒引擎
            return self.robust_engine.create_robust_chart(df, query_context)
        else:
            # 对于指定类型，也使用鲁棒引擎，但强制使用指定类型
            try:
                # 分析数据
                columns_info, complexity = self.robust_engine.analyze_dataframe(df)

                # 预处理数据
                processed_df = self.robust_engine._preprocess_data(df, columns_info, complexity)

                # 创建强制映射
                mapping = self._create_forced_mapping(processed_df, columns_info, chart_type, query_context)

                # 生成图表
                return self.robust_engine._generate_chart_by_mapping(processed_df, mapping)
            except Exception as e:
                logger.error(f"强制类型图表生成失败: {e}")
                return self.robust_engine._create_fallback_chart(df, columns_info if 'columns_info' in locals() else [])

    def _create_forced_mapping(self, df: pd.DataFrame, columns_info: List, chart_type: str, query_context: str):
        """为指定图表类型创建强制映射"""
        # 分类列信息
        time_cols = [col for col in columns_info if col.semantic_role == 'time']
        metric_cols = [col for col in columns_info if col.semantic_role == 'metric']
        category_cols = [col for col in columns_info if col.semantic_role == 'category']

        # 智能选择轴
        x_col = None
        y_col = None
        color_by = None

        if chart_type in ["line", "multi_line"]:
            # 折线图优先使用时间列作为X轴
            if time_cols:
                x_col = time_cols[0].name
            elif category_cols:
                x_col = category_cols[0].name

            if metric_cols:
                y_col = metric_cols[0].name

            if chart_type == "multi_line" and category_cols and len(category_cols) > 1:
                color_by = category_cols[1].name if x_col != category_cols[1].name else (category_cols[0].name if x_col != category_cols[0].name else None)

        elif chart_type == "bar":
            # 柱状图优先使用分类列作为X轴
            if category_cols:
                x_col = category_cols[0].name
            elif time_cols:
                x_col = time_cols[0].name

            if metric_cols:
                y_col = metric_cols[0].name

        elif chart_type == "pie":
            if category_cols:
                x_col = category_cols[0].name
            if metric_cols:
                y_col = metric_cols[0].name

        elif chart_type == "scatter":
            if len(metric_cols) >= 2:
                x_col = metric_cols[0].name
                y_col = metric_cols[1].name
            elif metric_cols and time_cols:
                x_col = time_cols[0].name
                y_col = metric_cols[0].name

            if category_cols:
                color_by = category_cols[0].name

        # 生成更智能的标题
        title = self._generate_smart_title(x_col, y_col, color_by, chart_type, query_context)

        return ChartMapping(
            chart_type=chart_type,
            x_axis=x_col,
            y_axis=y_col,
            color_by=color_by,
            title=title,
            complexity=ChartComplexity.SIMPLE,
            confidence=0.8
        )

    def _generate_smart_title(self, x_col: str, y_col: str, color_by: str, chart_type: str, query_context: str) -> str:
        """生成智能标题"""
        # 如果有查询上下文，尝试从中提取更好的标题
        if query_context:
            query_lower = query_context.lower()

            # 检查是否为趋势查询
            if any(word in query_lower for word in ['趋势', 'trend', '变化', 'change']):
                if color_by:
                    return f"{y_col} 按 {color_by} 的趋势分析"
                else:
                    return f"{y_col} 趋势分析"

            # 检查是否为分布查询
            if any(word in query_lower for word in ['分布', 'distribution', '占比', 'proportion']):
                if color_by:
                    return f"{y_col} 按 {color_by} 分布"
                else:
                    return f"{y_col} 分布"

            # 检查是否为对比查询
            if any(word in query_lower for word in ['对比', 'compare', '比较', 'comparison']):
                if color_by:
                    return f"{y_col} 按 {color_by} 对比分析"
                else:
                    return f"{y_col} 对比分析"

        # 默认标题生成逻辑
        if x_col and y_col:
            if color_by:
                if chart_type == "line":
                    return f"{y_col} 按 {color_by} 的 {x_col} 趋势"
                elif chart_type == "bar":
                    return f"{y_col} 按 {color_by} 分组的 {x_col} 分布"
                else:
                    return f"{y_col} 按 {color_by} 的 {x_col} 分析"
            else:
                if chart_type == "line":
                    return f"{y_col} 随 {x_col} 的趋势"
                elif chart_type == "bar":
                    return f"{y_col} 按 {x_col} 分布"
                elif chart_type == "pie":
                    return f"{y_col} 占比分布"
                else:
                    return f"{y_col} vs {x_col}"
        else:
            return f"数据可视化 ({chart_type})"

    def get_chart_options(self, df: pd.DataFrame, query_context: str = "") -> List[Dict]:
        """获取图表选项 - 兼容旧版本接口"""
        return self.robust_engine.get_chart_options(df, query_context)

    def get_chart_options_cached(self, df_shape: Tuple, columns: List[str]) -> List[Dict]:
        """缓存版本的图表选项获取"""
        return self.robust_engine.get_chart_options_cached(df_shape, columns)

    def get_chart_export_data(self, df: pd.DataFrame, chart_type: str = None, query_context: str = "") -> List[Dict]:
        """获取图表导出数据 - 兼容旧版本接口"""
        return self.robust_engine.get_chart_export_data(df, chart_type, query_context)

    def detect_chart_type(self, df: pd.DataFrame, query_context: str = "") -> str:
        """检测图表类型 - 兼容旧版本接口"""
        return self.robust_engine.detect_chart_type(df, query_context)

# 全局实例 - 使用兼容包装器确保向下兼容
viz_engine = VisualizationEngine()

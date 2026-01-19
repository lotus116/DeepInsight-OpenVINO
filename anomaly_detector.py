"""
Intel DeepInsight 智能异常检测系统
自动识别数据异常、趋势变化和业务风险点
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
import os
from dataclasses import dataclass

# 异常类型图标映射
ANOMALY_TYPE_ICONS = {
    "statistical_outlier": "📊",
    "extreme_value": "⚠️", 
    "zero_value_anomaly": "🔴",
    "negative_profit": "💸",
    "high_profit_margin": "📈",
    "price_anomaly": "💰",
    "trend_break": "📉",
    "declining_trend": "⬇️"
}

# 异常类型中文名称映射
ANOMALY_TYPE_NAMES = {
    "statistical_outlier": "数据异常",
    "extreme_value": "极值异常", 
    "zero_value_anomaly": "零值异常",
    "negative_profit": "业务异常",
    "high_profit_margin": "利润异常",
    "price_anomaly": "价格异常",
    "trend_break": "趋势异常",
    "declining_trend": "下降趋势"
}

@dataclass
class AnomalyPreview:
    """异常预览数据结构"""
    type: str
    icon: str
    type_name: str
    short_description: str  # 限制50字符
    sample_data: List[str]  # 1-2个数据样本
    impact_level: str  # "high", "medium", "low"
    confidence: float
    quick_reason: str  # 简要原因说明
    quick_action: Optional[str] = None  # 快速处理建议

class AnomalyDetector:
    """智能异常检测器"""
    
    def __init__(self):
        self.anomaly_history_file = "data/anomaly_history.json"
        self._ensure_history_file()
    
    def _ensure_history_file(self):
        """确保异常历史文件存在"""
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(self.anomaly_history_file):
            with open(self.anomaly_history_file, 'w', encoding='utf-8') as f:
                json.dump({"anomalies": []}, f, indent=2)
    
    def detect_statistical_anomalies(self, df: pd.DataFrame, query_context: str = "") -> List[Dict]:
        """检测统计异常"""
        anomalies = []
        
        if df.empty:
            return anomalies
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if df[col].notna().sum() < 3:  # 数据点太少，跳过
                    continue
                
                # 计算统计指标
                mean_val = df[col].mean()
                std_val = df[col].std()
                median_val = df[col].median()
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                
                # 检测离群值 (IQR方法)
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                
                if len(outliers) > 0:
                    outlier_values = outliers[col].tolist()
                    # 获取异常值的行索引和完整记录
                    outlier_indices = outliers.index.tolist()
                    outlier_records = []
                    for idx in outlier_indices[:3]:  # 只显示前3个完整记录
                        record = df.loc[idx].to_dict()
                        outlier_records.append({
                            "row_index": idx,
                            "anomaly_value": record[col],
                            "full_record": {k: v for k, v in record.items() if pd.notna(v)}
                        })
                    
                    anomalies.append({
                        "type": "statistical_outlier",
                        "column": col,
                        "severity": "medium",
                        "count": len(outliers),
                        "values": outlier_values[:5],  # 只显示前5个
                        "description": f"{col} 列发现 {len(outliers)} 个统计异常值",
                        "details": f"正常范围: {lower_bound:.2f} - {upper_bound:.2f}",
                        "suggestion": f"建议检查这些异常值是否为数据错误或特殊情况",
                        "criteria": {
                            "method": "IQR (四分位距) 方法",
                            "threshold": "1.5倍IQR",
                            "q1": q1,
                            "q3": q3,
                            "iqr": iqr,
                            "lower_bound": lower_bound,
                            "upper_bound": upper_bound
                        },
                        "evidence": {
                            "outlier_records": outlier_records,
                            "statistical_summary": {
                                "mean": mean_val,
                                "median": median_val,
                                "std": std_val,
                                "min_outlier": min(outlier_values),
                                "max_outlier": max(outlier_values)
                            }
                        }
                    })
                
                # 检测极值
                if std_val > 0:
                    z_scores = np.abs((df[col] - mean_val) / std_val)
                    extreme_values = df[z_scores > 3]  # Z-score > 3
                    
                    if len(extreme_values) > 0:
                        extreme_indices = extreme_values.index.tolist()
                        extreme_records = []
                        for idx in extreme_indices[:3]:
                            record = df.loc[idx].to_dict()
                            z_score = abs((record[col] - mean_val) / std_val)
                            extreme_records.append({
                                "row_index": idx,
                                "anomaly_value": record[col],
                                "z_score": z_score,
                                "full_record": {k: v for k, v in record.items() if pd.notna(v)}
                            })
                        
                        anomalies.append({
                            "type": "extreme_value",
                            "column": col,
                            "severity": "high",
                            "count": len(extreme_values),
                            "values": extreme_values[col].tolist()[:3],
                            "description": f"{col} 列发现 {len(extreme_values)} 个极端值",
                            "details": f"均值: {mean_val:.2f}, 标准差: {std_val:.2f}",
                            "suggestion": "极端值可能表示异常业务情况，建议深入分析",
                            "criteria": {
                                "method": "Z-Score 标准化方法",
                                "threshold": "Z-Score > 3",
                                "mean": mean_val,
                                "std": std_val,
                                "z_threshold": 3.0
                            },
                            "evidence": {
                                "extreme_records": extreme_records,
                                "statistical_summary": {
                                    "mean": mean_val,
                                    "std": std_val,
                                    "max_z_score": max([abs((val - mean_val) / std_val) for val in extreme_values[col]]),
                                    "extreme_range": f"{extreme_values[col].min():.2f} - {extreme_values[col].max():.2f}"
                                }
                            }
                        })
                
                # 检测零值异常
                zero_count = (df[col] == 0).sum()
                zero_ratio = zero_count / len(df)
                if zero_ratio > 0.3 and col in ['sales', 'profit', '销售额', '利润']:
                    zero_records = df[df[col] == 0]
                    zero_sample_records = []
                    for idx in zero_records.index[:3]:
                        record = df.loc[idx].to_dict()
                        zero_sample_records.append({
                            "row_index": idx,
                            "full_record": {k: v for k, v in record.items() if pd.notna(v)}
                        })
                    
                    anomalies.append({
                        "type": "zero_value_anomaly",
                        "column": col,
                        "severity": "medium",
                        "count": zero_count,
                        "ratio": zero_ratio,
                        "description": f"{col} 列有 {zero_ratio:.1%} 的数据为零",
                        "details": f"零值数量: {zero_count}/{len(df)}",
                        "suggestion": "大量零值可能表示业务问题或数据质量问题",
                        "criteria": {
                            "method": "零值比例检测",
                            "threshold": "30%",
                            "actual_ratio": zero_ratio,
                            "threshold_ratio": 0.3
                        },
                        "evidence": {
                            "zero_records": zero_sample_records,
                            "distribution": {
                                "total_records": len(df),
                                "zero_count": zero_count,
                                "non_zero_count": len(df) - zero_count,
                                "zero_percentage": zero_ratio * 100
                            }
                        }
                    })
        
        except Exception as e:
            print(f"统计异常检测失败: {e}")
        
        return anomalies
    
    def detect_business_anomalies(self, df: pd.DataFrame, query_context: str = "") -> List[Dict]:
        """检测业务异常"""
        anomalies = []
        
        if df.empty:
            return anomalies
        
        try:
            # 检测负利润
            if 'profit' in df.columns or '利润' in df.columns:
                profit_col = 'profit' if 'profit' in df.columns else '利润'
                negative_profit = df[df[profit_col] < 0]
                
                if len(negative_profit) > 0:
                    total_negative = negative_profit[profit_col].sum()
                    negative_sample_records = []
                    for idx in negative_profit.index[:3]:
                        record = df.loc[idx].to_dict()
                        negative_sample_records.append({
                            "row_index": idx,
                            "profit_value": record[profit_col],
                            "full_record": {k: v for k, v in record.items() if pd.notna(v)}
                        })
                    
                    anomalies.append({
                        "type": "negative_profit",
                        "column": profit_col,
                        "severity": "high",
                        "count": len(negative_profit),
                        "total_loss": abs(total_negative),
                        "description": f"发现 {len(negative_profit)} 条负利润记录",
                        "details": f"总损失: {abs(total_negative):,.2f}",
                        "suggestion": "负利润可能表示定价策略问题或成本控制不当",
                        "criteria": {
                            "method": "业务规则检测",
                            "threshold": "利润 < 0",
                            "business_rule": "正常业务中利润应为正值"
                        },
                        "evidence": {
                            "negative_records": negative_sample_records,
                            "financial_impact": {
                                "total_loss": abs(total_negative),
                                "average_loss": abs(total_negative) / len(negative_profit),
                                "worst_loss": negative_profit[profit_col].min(),
                                "affected_percentage": len(negative_profit) / len(df) * 100
                            }
                        }
                    })
            
            # 检测异常高的利润率
            if all(col in df.columns for col in ['sales', 'profit']) or all(col in df.columns for col in ['销售额', '利润']):
                sales_col = 'sales' if 'sales' in df.columns else '销售额'
                profit_col = 'profit' if 'profit' in df.columns else '利润'
                
                # 计算利润率
                df_temp = df[(df[sales_col] > 0) & (df[profit_col].notna())].copy()
                if not df_temp.empty:
                    df_temp['profit_margin'] = df_temp[profit_col] / df_temp[sales_col]
                    
                    # 检测异常高的利润率 (>100%)
                    high_margin = df_temp[df_temp['profit_margin'] > 1.0]
                    if len(high_margin) > 0:
                        high_margin_sample_records = []
                        for idx in high_margin.index[:3]:
                            record = df.loc[idx].to_dict()
                            high_margin_sample_records.append({
                                "row_index": idx,
                                "profit_margin": record.get('profit_margin', 0),
                                "sales": record.get(sales_col, 0),
                                "profit": record.get(profit_col, 0),
                                "full_record": {k: v for k, v in record.items() if pd.notna(v) and k != 'profit_margin'}
                            })
                        
                        anomalies.append({
                            "type": "high_profit_margin",
                            "column": "profit_margin",
                            "severity": "medium",
                            "count": len(high_margin),
                            "max_margin": high_margin['profit_margin'].max(),
                            "description": f"发现 {len(high_margin)} 条利润率超过100%的记录",
                            "details": f"最高利润率: {high_margin['profit_margin'].max():.1%}",
                            "suggestion": "异常高的利润率可能表示数据错误或特殊业务情况",
                            "criteria": {
                                "method": "利润率计算检测",
                                "threshold": "利润率 > 100%",
                                "calculation": "利润率 = 利润 / 销售额",
                                "normal_range": "通常利润率在5%-50%之间"
                            },
                            "evidence": {
                                "high_margin_records": high_margin_sample_records,
                                "margin_statistics": {
                                    "max_margin": high_margin['profit_margin'].max(),
                                    "min_margin": high_margin['profit_margin'].min(),
                                    "avg_margin": high_margin['profit_margin'].mean(),
                                    "affected_percentage": len(high_margin) / len(df_temp) * 100
                                }
                            }
                        })
            
            # 检测销售额与数量的不一致
            if all(col in df.columns for col in ['sales', 'quantity']) or all(col in df.columns for col in ['销售额', '数量']):
                sales_col = 'sales' if 'sales' in df.columns else '销售额'
                qty_col = 'quantity' if 'quantity' in df.columns else '数量'
                
                # 计算单价
                df_temp = df[(df[sales_col] > 0) & (df[qty_col] > 0)].copy()
                if not df_temp.empty:
                    df_temp['unit_price'] = df_temp[sales_col] / df_temp[qty_col]
                    
                    # 检测异常单价
                    q1 = df_temp['unit_price'].quantile(0.25)
                    q3 = df_temp['unit_price'].quantile(0.75)
                    iqr = q3 - q1
                    
                    if iqr > 0:
                        lower_bound = q1 - 3 * iqr  # 更宽松的界限
                        upper_bound = q3 + 3 * iqr
                        
                        price_anomalies = df_temp[(df_temp['unit_price'] < lower_bound) | (df_temp['unit_price'] > upper_bound)]
                        
                        if len(price_anomalies) > 0:
                            price_sample_records = []
                            for idx in price_anomalies.index[:3]:
                                record = df.loc[idx].to_dict()
                                price_sample_records.append({
                                    "row_index": idx,
                                    "unit_price": record.get('unit_price', 0),
                                    "sales": record.get(sales_col, 0),
                                    "quantity": record.get(qty_col, 0),
                                    "full_record": {k: v for k, v in record.items() if pd.notna(v) and k != 'unit_price'}
                                })
                            
                            anomalies.append({
                                "type": "price_anomaly",
                                "column": "unit_price",
                                "severity": "medium",
                                "count": len(price_anomalies),
                                "price_range": f"{price_anomalies['unit_price'].min():.2f} - {price_anomalies['unit_price'].max():.2f}",
                                "description": f"发现 {len(price_anomalies)} 条异常单价记录",
                                "details": f"正常单价范围: {lower_bound:.2f} - {upper_bound:.2f}",
                                "suggestion": "异常单价可能表示折扣、促销或数据录入错误",
                                "criteria": {
                                    "method": "IQR (四分位距) 方法",
                                    "threshold": "3倍IQR",
                                    "calculation": "单价 = 销售额 / 数量",
                                    "q1": q1,
                                    "q3": q3,
                                    "iqr": iqr,
                                    "lower_bound": lower_bound,
                                    "upper_bound": upper_bound
                                },
                                "evidence": {
                                    "price_anomaly_records": price_sample_records,
                                    "price_statistics": {
                                        "normal_price_mean": df_temp['unit_price'].mean(),
                                        "normal_price_median": df_temp['unit_price'].median(),
                                        "anomaly_price_min": price_anomalies['unit_price'].min(),
                                        "anomaly_price_max": price_anomalies['unit_price'].max(),
                                        "affected_percentage": len(price_anomalies) / len(df_temp) * 100
                                    }
                                }
                            })
        
        except Exception as e:
            print(f"业务异常检测失败: {e}")
        
        return anomalies
    
    def detect_trend_anomalies(self, df: pd.DataFrame, query_context: str = "") -> List[Dict]:
        """检测趋势异常"""
        anomalies = []
        
        if df.empty or len(df) < 5:  # 数据点太少无法分析趋势
            return anomalies
        
        try:
            # 寻找时间列 - 支持datetime类型和可转换的字符串
            date_columns = []
            for col in df.columns:
                # 检查datetime类型
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    date_columns.append(col)
                # 检查可转换的字符串
                elif df[col].dtype == 'object':
                    try:
                        pd.to_datetime(df[col].head())
                        date_columns.append(col)
                    except:
                        pass
            
            # 如果没有时间列，尝试使用索引作为时间序列
            if not date_columns:
                # 检查是否有足够的数据点进行趋势分析
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0 and len(df) >= 7:
                    # 使用索引作为时间序列进行趋势分析
                    for num_col in numeric_cols:
                        if df[num_col].notna().sum() < 7:
                            continue
                        
                        values = df[num_col].dropna()
                        if len(values) < 7:
                            continue
                        
                        # 检测持续下降趋势（无需时间列）
                        recent_values = values.tail(10)  # 取最后10个值
                        if len(recent_values) >= 7:
                            # 检查是否连续下降
                            consecutive_declines = 0
                            for i in range(len(recent_values)-1):
                                if recent_values.iloc[i] > recent_values.iloc[i+1]:
                                    consecutive_declines += 1
                                else:
                                    break
                            
                            if consecutive_declines >= 6:  # 至少连续6次下降
                                decline_rate = (recent_values.iloc[-1] - recent_values.iloc[0]) / recent_values.iloc[0] if recent_values.iloc[0] != 0 else 0
                                if decline_rate < -0.15:  # 下降超过15%
                                    # 记录下降趋势的详细信息
                                    decline_records = []
                                    for i in range(len(recent_values)):
                                        decline_records.append({
                                            "period": i + 1,
                                            "index": recent_values.index[i],
                                            "value": recent_values.iloc[i],
                                            "cumulative_decline": (recent_values.iloc[i] - recent_values.iloc[0]) / recent_values.iloc[0] if recent_values.iloc[0] != 0 else 0
                                        })
                                    
                                    anomalies.append({
                                        "type": "declining_trend",
                                        "column": num_col,
                                        "severity": "high",
                                        "decline_rate": abs(decline_rate),
                                        "description": f"{num_col} 呈现持续下降趋势",
                                        "details": f"近期下降幅度: {abs(decline_rate):.1%}",
                                        "suggestion": "持续下降趋势需要关注，可能需要采取干预措施",
                                        "criteria": {
                                            "method": "连续下降检测",
                                            "threshold": f"连续{consecutive_declines}期下降且总降幅 > 15%",
                                            "observation_window": f"最近{len(recent_values)}个数据点",
                                            "decline_threshold": 0.15
                                        },
                                        "evidence": {
                                            "decline_sequence": decline_records,
                                            "trend_statistics": {
                                                "total_decline_rate": decline_rate,
                                                "periods_analyzed": len(recent_values),
                                                "start_value": recent_values.iloc[0],
                                                "end_value": recent_values.iloc[-1],
                                                "average_period_decline": decline_rate / (len(recent_values) - 1),
                                                "consecutive_declines": consecutive_declines
                                            }
                                        }
                                    })
                return anomalies
            
            date_col = date_columns[0]
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for num_col in numeric_cols:
                if df[num_col].notna().sum() < 5:
                    continue
                
                # 按时间排序
                df_sorted = df.sort_values(date_col)
                values = df_sorted[num_col].dropna()
                
                if len(values) < 5:
                    continue
                
                # 计算移动平均和标准差
                window_size = min(5, len(values) // 2)
                rolling_mean = values.rolling(window=window_size).mean()
                rolling_std = values.rolling(window=window_size).std()
                
                # 检测突然的大幅变化
                pct_change = values.pct_change().abs()
                large_changes = pct_change[pct_change > 0.5]  # 50%以上的变化
                
                if len(large_changes) > 0:
                    # 找到变化最大的几个点
                    change_records = []
                    for idx in pct_change.nlargest(3).index:
                        if idx > 0:  # 确保有前一个值进行比较
                            prev_idx = values.index[values.index.get_loc(idx) - 1]
                            current_val = values.loc[idx]
                            prev_val = values.loc[prev_idx]
                            change_pct = (current_val - prev_val) / prev_val if prev_val != 0 else float('inf')
                            
                            change_records.append({
                                "current_index": idx,
                                "previous_index": prev_idx,
                                "current_value": current_val,
                                "previous_value": prev_val,
                                "change_percentage": change_pct,
                                "change_absolute": current_val - prev_val
                            })
                    
                    anomalies.append({
                        "type": "trend_break",
                        "column": num_col,
                        "severity": "medium",
                        "count": len(large_changes),
                        "max_change": large_changes.max(),
                        "description": f"{num_col} 趋势中发现 {len(large_changes)} 个突变点",
                        "details": f"最大变化幅度: {large_changes.max():.1%}",
                        "suggestion": "趋势突变可能表示市场变化、政策影响或异常事件",
                        "criteria": {
                            "method": "百分比变化检测",
                            "threshold": "变化幅度 > 50%",
                            "calculation": "变化率 = (当前值 - 前值) / 前值",
                            "time_window": "逐期比较"
                        },
                        "evidence": {
                            "trend_break_points": change_records,
                            "trend_statistics": {
                                "total_data_points": len(values),
                                "break_points_count": len(large_changes),
                                "max_change_percentage": large_changes.max(),
                                "avg_change_percentage": large_changes.mean(),
                                "break_frequency": len(large_changes) / len(values) * 100
                            }
                        }
                    })
                
                # 检测持续下降趋势
                if len(values) >= 7:
                    recent_values = values.tail(7)
                    if all(recent_values.iloc[i] >= recent_values.iloc[i+1] for i in range(len(recent_values)-1)):
                        decline_rate = (recent_values.iloc[-1] - recent_values.iloc[0]) / recent_values.iloc[0]
                        if decline_rate < -0.2:  # 下降超过20%
                            # 记录下降趋势的详细信息
                            decline_records = []
                            for i, idx in enumerate(recent_values.index):
                                decline_records.append({
                                    "period": i + 1,
                                    "index": idx,
                                    "value": recent_values.iloc[i],
                                    "cumulative_decline": (recent_values.iloc[i] - recent_values.iloc[0]) / recent_values.iloc[0] if recent_values.iloc[0] != 0 else 0
                                })
                            
                            anomalies.append({
                                "type": "declining_trend",
                                "column": num_col,
                                "severity": "high",
                                "decline_rate": abs(decline_rate),
                                "description": f"{num_col} 呈现持续下降趋势",
                                "details": f"近期下降幅度: {abs(decline_rate):.1%}",
                                "suggestion": "持续下降趋势需要关注，可能需要采取干预措施",
                                "criteria": {
                                    "method": "连续下降检测",
                                    "threshold": "连续7期下降且总降幅 > 20%",
                                    "observation_window": "最近7个数据点",
                                    "decline_threshold": 0.2
                                },
                                "evidence": {
                                    "decline_sequence": decline_records,
                                    "trend_statistics": {
                                        "total_decline_rate": decline_rate,
                                        "periods_analyzed": len(recent_values),
                                        "start_value": recent_values.iloc[0],
                                        "end_value": recent_values.iloc[-1],
                                        "average_period_decline": decline_rate / (len(recent_values) - 1),
                                        "consecutive_declines": len(recent_values) - 1
                                    }
                                }
                            })
        
        except Exception as e:
            print(f"趋势异常检测失败: {e}")
        
        return anomalies
    
    def analyze_anomalies(self, df: pd.DataFrame, query_context: str = "") -> Dict:
        """综合异常分析"""
        all_anomalies = []
        
        # 收集各类异常
        all_anomalies.extend(self.detect_statistical_anomalies(df, query_context))
        all_anomalies.extend(self.detect_business_anomalies(df, query_context))
        all_anomalies.extend(self.detect_trend_anomalies(df, query_context))
        
        # 按严重程度排序
        severity_order = {"high": 3, "medium": 2, "low": 1}
        all_anomalies.sort(key=lambda x: severity_order.get(x.get("severity", "low"), 1), reverse=True)
        
        # 为每个异常添加更详细的依据信息
        for anomaly in all_anomalies:
            anomaly_type = anomaly.get('type', 'unknown')
            column = anomaly.get('column', '未知列')
            
            # 根据异常类型添加更详细的依据描述
            if 'evidence' in anomaly:
                evidence = anomaly['evidence']
                
                # 统计异常的依据
                if anomaly_type == 'statistical_outlier':
                    if 'outlier_records' in evidence and evidence['outlier_records']:
                        records = evidence['outlier_records'][:2]  # 取前2个示例
                        details = "**具体异常数据示例**:\n"
                        for rec in records:
                            details += f"• 行{rec['row_index']}: 值={rec['anomaly_value']:.2f}\n"
                            if 'full_record' in rec:
                                details += f"  完整记录: {str(rec['full_record'])[:100]}...\n"
                        anomaly['evidence_details'] = details
                
                elif anomaly_type == 'negative_profit':
                    if 'negative_records' in evidence and evidence['negative_records']:
                        records = evidence['negative_records'][:2]
                        details = "**负利润具体示例**:\n"
                        for rec in records:
                            details += f"• 行{rec['row_index']}: 利润={rec['profit_value']:.2f}\n"
                        anomaly['evidence_details'] = details
                
                elif anomaly_type == 'extreme_value':
                    if 'extreme_records' in evidence and evidence['extreme_records']:
                        records = evidence['extreme_records'][:2]
                        details = "**极端值具体示例**:\n"
                        for rec in records:
                            details += f"• 行{rec['row_index']}: 值={rec['anomaly_value']:.2f}, Z分数={rec['z_score']:.2f}\n"
                        anomaly['evidence_details'] = details
                
                # 添加统计依据
                if 'criteria' in anomaly:
                    criteria = anomaly['criteria']
                    if anomaly_type == 'statistical_outlier':
                        anomaly['statistical_basis'] = f"检测方法: {criteria.get('method', 'IQR方法')}, 正常范围: {criteria.get('lower_bound', 0):.2f} - {criteria.get('upper_bound', 0):.2f}"
                    elif anomaly_type == 'extreme_value':
                        anomaly['statistical_basis'] = f"检测方法: {criteria.get('method', 'Z-Score方法')}, 阈值: Z > {criteria.get('z_threshold', 3)}"
        
        # 生成摘要
        summary = {
            "total_anomalies": len(all_anomalies),
            "high_severity": len([a for a in all_anomalies if a.get("severity") == "high"]),
            "medium_severity": len([a for a in all_anomalies if a.get("severity") == "medium"]),
            "low_severity": len([a for a in all_anomalies if a.get("severity") == "low"]),
            "anomalies": all_anomalies,
            "analysis_time": datetime.now().isoformat()
        }
        
        return summary
    
    def _save_anomaly_to_history(self, analysis: Dict, query_context: str):
        """保存异常分析到历史记录"""
        try:
            with open(self.anomaly_history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 添加新的异常记录
            history_entry = {
                "timestamp": datetime.now().isoformat(),
                "query_context": query_context,
                "total_anomalies": analysis["total_anomalies"],
                "high_severity": analysis["high_severity"],
                "anomaly_types": list(set([a["type"] for a in analysis["anomalies"]]))
            }
            
            data["anomalies"].append(history_entry)
            
            # 保持最近100条记录
            if len(data["anomalies"]) > 100:
                data["anomalies"] = data["anomalies"][-100:]
            
            with open(self.anomaly_history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            print(f"保存异常历史失败: {e}")
    
    def _generate_anomaly_preview(self, anomaly: Dict) -> AnomalyPreview:
        """生成异常预览数据"""
        anomaly_type = anomaly.get('type', 'unknown')
        icon = ANOMALY_TYPE_ICONS.get(anomaly_type, "🔍")
        type_name = ANOMALY_TYPE_NAMES.get(anomaly_type, "未知异常")
        
        # 生成简短描述（限制50字符）
        description = anomaly.get('description', '')
        short_description = description[:47] + "..." if len(description) > 50 else description
        
        # 生成数据样本
        sample_data = []
        if 'evidence' in anomaly:
            evidence = anomaly['evidence']
            
            # 从不同类型的证据中提取样本
            if 'outlier_records' in evidence and evidence['outlier_records']:
                for record in evidence['outlier_records'][:2]:
                    sample_data.append(f"行{record['row_index']}: {record['anomaly_value']:.2f}")
            
            elif 'negative_records' in evidence and evidence['negative_records']:
                for record in evidence['negative_records'][:2]:
                    sample_data.append(f"行{record['row_index']}: 利润{record['profit_value']:.2f}")
            
            elif 'extreme_records' in evidence and evidence['extreme_records']:
                for record in evidence['extreme_records'][:2]:
                    sample_data.append(f"行{record['row_index']}: {record['anomaly_value']:.2f}")
            
            elif 'zero_records' in evidence and evidence['zero_records']:
                for record in evidence['zero_records'][:2]:
                    sample_data.append(f"行{record['row_index']}: 零值")
        
        # 如果没有具体样本，使用异常值
        if not sample_data and 'values' in anomaly:
            values = anomaly['values'][:2]
            sample_data = [f"{val:.2f}" if isinstance(val, (int, float)) else str(val) for val in values]
        
        # 生成简要原因说明
        quick_reason = self._generate_quick_reason(anomaly)
        
        # 生成快速处理建议
        quick_action = self._generate_quick_action(anomaly)
        
        # 计算置信度（简化版本）
        confidence = self._calculate_confidence(anomaly)
        
        return AnomalyPreview(
            type=anomaly_type,
            icon=icon,
            type_name=type_name,
            short_description=short_description,
            sample_data=sample_data,
            impact_level=anomaly.get('severity', 'medium'),
            confidence=confidence,
            quick_reason=quick_reason,
            quick_action=quick_action
        )
    
    def _generate_quick_reason(self, anomaly: Dict) -> str:
        """生成简要原因说明"""
        anomaly_type = anomaly.get('type', '')
        
        if anomaly_type == 'statistical_outlier':
            if 'criteria' in anomaly:
                criteria = anomaly['criteria']
                lower = criteria.get('lower_bound', 0)
                upper = criteria.get('upper_bound', 0)
                return f"数值超出正常范围 {lower:.1f}-{upper:.1f}"
        
        elif anomaly_type == 'extreme_value':
            if 'criteria' in anomaly:
                threshold = anomaly['criteria'].get('z_threshold', 3)
                return f"Z-Score超过{threshold}倍标准差"
        
        elif anomaly_type == 'negative_profit':
            count = anomaly.get('count', 0)
            return f"发现{count}条负利润记录"
        
        elif anomaly_type == 'high_profit_margin':
            max_margin = anomaly.get('max_margin', 0)
            return f"利润率高达{max_margin:.1%}"
        
        elif anomaly_type == 'zero_value_anomaly':
            ratio = anomaly.get('ratio', 0)
            return f"{ratio:.1%}的数据为零值"
        
        elif anomaly_type == 'price_anomaly':
            return "单价异常偏离正常范围"
        
        elif anomaly_type == 'trend_break':
            max_change = anomaly.get('max_change', 0)
            return f"趋势突变幅度达{max_change:.1%}"
        
        elif anomaly_type == 'declining_trend':
            decline_rate = anomaly.get('decline_rate', 0)
            return f"持续下降{decline_rate:.1%}"
        
        return "检测到数据异常"
    
    def _generate_quick_action(self, anomaly: Dict) -> Optional[str]:
        """生成快速处理建议"""
        severity = anomaly.get('severity', 'medium')
        anomaly_type = anomaly.get('type', '')
        
        if severity == 'high':
            if anomaly_type == 'negative_profit':
                return "立即检查成本和定价"
            elif anomaly_type == 'extreme_value':
                return "紧急核实数据来源"
            elif anomaly_type == 'declining_trend':
                return "制定应对措施"
            else:
                return "需要立即关注"
        
        elif severity == 'medium':
            if anomaly_type in ['statistical_outlier', 'price_anomaly']:
                return "建议核实数据"
            elif anomaly_type == 'high_profit_margin':
                return "确认业务合理性"
            else:
                return "建议进一步分析"
        
        return "可选择性处理"
    
    def _calculate_confidence(self, anomaly: Dict) -> float:
        """计算异常检测置信度（简化版本）"""
        anomaly_type = anomaly.get('type', '')
        count = anomaly.get('count', 1)
        
        # 基础置信度
        base_confidence = 0.7
        
        # 根据异常类型调整
        if anomaly_type in ['extreme_value', 'negative_profit']:
            base_confidence = 0.9  # 高置信度
        elif anomaly_type in ['statistical_outlier', 'trend_break']:
            base_confidence = 0.8  # 中高置信度
        elif anomaly_type in ['zero_value_anomaly', 'price_anomaly']:
            base_confidence = 0.75  # 中等置信度
        
        # 根据异常数量调整
        if count >= 5:
            base_confidence += 0.1
        elif count >= 10:
            base_confidence += 0.15
        
        # 确保在合理范围内
        return min(0.95, max(0.5, base_confidence))
    
    def get_anomaly_highlights(self, anomalies: List[Dict]) -> List[int]:
        """获取需要在可视化中高亮的数据点索引"""
        highlight_indices = []
        
        # 这里简化处理，实际应该根据异常类型和数据特征来确定高亮点
        for anomaly in anomalies:
            if anomaly.get("type") == "statistical_outlier" and "values" in anomaly:
                # 对于统计异常，可以尝试找到对应的行索引
                # 这里简化为返回前几个索引
                highlight_indices.extend(range(min(5, len(anomaly.get("values", [])))))
        
        return list(set(highlight_indices))  # 去重
        """获取需要在可视化中高亮的数据点索引"""
        highlight_indices = []
        
        # 这里简化处理，实际应该根据异常类型和数据特征来确定高亮点
        for anomaly in anomalies:
            if anomaly.get("type") == "statistical_outlier" and "values" in anomaly:
                # 对于统计异常，可以尝试找到对应的行索引
                # 这里简化为返回前几个索引
                highlight_indices.extend(range(min(5, len(anomaly.get("values", [])))))
        
        return list(set(highlight_indices))  # 去重

# 全局实例
anomaly_detector = AnomalyDetector()
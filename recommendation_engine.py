"""
Intel DeepInsight 智能推荐引擎
基于查询结果和用户行为，使用AI推理生成个性化问题推荐
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd

class RecommendationEngine:
    """智能推荐引擎"""
    
    def __init__(self):
        self.click_history_file = "data/click_history.json"
        self.recommendation_cache_file = "data/recommendation_cache.json"
        self._ensure_history_files()
    
    def _ensure_history_files(self):
        """确保历史文件存在"""
        os.makedirs("data", exist_ok=True)
        
        if not os.path.exists(self.click_history_file):
            with open(self.click_history_file, 'w', encoding='utf-8') as f:
                json.dump({"clicks": []}, f, indent=2)
        
        if not os.path.exists(self.recommendation_cache_file):
            with open(self.recommendation_cache_file, 'w', encoding='utf-8') as f:
                json.dump({"cache": {}}, f, indent=2)
    
    def record_question_click(self, question: str):
        """记录问题点击"""
        try:
            with open(self.click_history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            click_entry = {
                "question": question,
                "timestamp": datetime.now().isoformat(),
                "session_id": "current"  # 可以后续扩展
            }
            
            data["clicks"].append(click_entry)
            
            # 保持最近100条记录
            if len(data["clicks"]) > 100:
                data["clicks"] = data["clicks"][-100:]
            
            with open(self.click_history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            print(f"记录点击失败: {e}")
    
    def generate_recommendations(self, current_query: str, result_df: pd.DataFrame, 
                               num_recommendations: int = 3, llm_client=None, model_name: str = None) -> List[str]:
        """使用AI生成推荐问题"""
        
        # 如果没有LLM客户端，返回基础推荐
        if not llm_client:
            return self._get_fallback_recommendations(current_query, result_df, num_recommendations)
        
        try:
            # 构建推荐提示词
            prompt = self._build_recommendation_prompt(current_query, result_df, num_recommendations)
            
            # 使用传入的模型名称，如果没有则使用默认值
            model_to_use = model_name if model_name else "deepseek-chat"
            
            # 调用LLM生成推荐
            response = llm_client.chat.completions.create(
                model=model_to_use,  # 使用配置的模型名称
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            
            # 获取响应内容
            response_content = response.choices[0].message.content
            
            # 解析推荐结果
            recommendations = self._parse_recommendations(response_content)
            
            # 缓存推荐结果
            self._cache_recommendations(current_query, recommendations)
            
            return recommendations[:num_recommendations]
        
        except Exception as e:
            print(f"AI推荐生成失败: {e}")
            return self._get_fallback_recommendations(current_query, result_df, num_recommendations)
    
    def _build_recommendation_prompt(self, current_query: str, result_df: pd.DataFrame, num_recommendations: int) -> str:
        """构建推荐提示词"""
        
        # 分析数据特征
        data_summary = self._analyze_data_features(result_df)
        
        # 获取历史点击偏好
        click_patterns = self._get_click_patterns()
        
        prompt = f"""基于用户的当前查询和数据结果，生成{num_recommendations}个相关的后续分析问题。

当前查询: {current_query}

数据特征:
{data_summary}

用户历史偏好:
{click_patterns}

要求:
1. 生成的问题应该与当前查询相关，但提供不同的分析角度
2. 问题应该具体、可执行，适合SQL查询
3. 考虑用户的历史点击偏好
4. 每个问题不超过20个字
5. 直接返回问题列表，每行一个问题，不要编号或其他格式

示例格式:
各地区销售额对比分析
产品类别利润率排名
季度销售趋势变化"""

        return prompt
    
    def _analyze_data_features(self, df: pd.DataFrame) -> str:
        """分析数据特征"""
        if df.empty:
            return "数据为空"
        
        features = []
        
        # 数据规模
        features.append(f"数据量: {len(df)}条记录")
        
        # 列信息
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if numeric_cols:
            features.append(f"数值字段: {', '.join(numeric_cols[:3])}")
        if text_cols:
            features.append(f"分类字段: {', '.join(text_cols[:3])}")
        
        # 关键统计
        if 'sales' in df.columns or '销售额' in df.columns:
            sales_col = 'sales' if 'sales' in df.columns else '销售额'
            total_sales = df[sales_col].sum()
            features.append(f"总销售额: {total_sales:,.0f}")
        
        return "\n".join(features)
    
    def _get_click_patterns(self) -> str:
        """获取用户点击模式"""
        try:
            with open(self.click_history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            recent_clicks = data["clicks"][-10:]  # 最近10次点击
            
            if not recent_clicks:
                return "暂无历史偏好数据"
            
            # 分析点击模式
            patterns = []
            for click in recent_clicks:
                question = click["question"]
                if "销售" in question:
                    patterns.append("销售分析")
                elif "利润" in question:
                    patterns.append("利润分析")
                elif "地区" in question or "城市" in question:
                    patterns.append("地域分析")
                elif "产品" in question or "类别" in question:
                    patterns.append("产品分析")
                elif "趋势" in question:
                    patterns.append("趋势分析")
            
            if patterns:
                unique_patterns = list(set(patterns))
                return f"偏好分析类型: {', '.join(unique_patterns)}"
            else:
                return "偏好分析类型: 综合分析"
        
        except Exception:
            return "暂无历史偏好数据"
    
    def _parse_recommendations(self, response: str) -> List[str]:
        """解析AI推荐结果"""
        try:
            # 按行分割并清理
            lines = response.strip().split('\n')
            recommendations = []
            
            for line in lines:
                line = line.strip()
                # 移除编号、符号等
                line = line.lstrip('0123456789.-• ')
                if line and len(line) <= 30:  # 过滤过长的问题
                    recommendations.append(line)
            
            return recommendations
        
        except Exception:
            return []
    
    def _cache_recommendations(self, query: str, recommendations: List[str]):
        """缓存推荐结果"""
        try:
            with open(self.recommendation_cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            cache_key = query[:50]  # 使用查询前50字符作为key
            data["cache"][cache_key] = {
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat()
            }
            
            # 保持缓存大小
            if len(data["cache"]) > 50:
                # 删除最旧的缓存
                oldest_key = min(data["cache"].keys(), 
                               key=lambda k: data["cache"][k]["timestamp"])
                del data["cache"][oldest_key]
            
            with open(self.recommendation_cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            print(f"缓存推荐失败: {e}")
    
    def _get_fallback_recommendations(self, current_query: str, result_df: pd.DataFrame, 
                                    num_recommendations: int) -> List[str]:
        """获取备用推荐（基于规则）"""
        
        fallback_recommendations = []
        
        # 基于查询内容的推荐
        if "销售" in current_query:
            fallback_recommendations.extend([
                "各地区销售额对比分析",
                "销售额季度趋势变化",
                "热销产品类别排名"
            ])
        elif "利润" in current_query:
            fallback_recommendations.extend([
                "利润率最高的产品类别",
                "各地区利润分布情况",
                "利润趋势月度分析"
            ])
        elif "地区" in current_query or "城市" in current_query:
            fallback_recommendations.extend([
                "地区销售业绩排名",
                "各地区产品偏好分析",
                "地区市场增长潜力"
            ])
        elif "产品" in current_query or "类别" in current_query:
            fallback_recommendations.extend([
                "产品销量排行榜",
                "产品利润率对比",
                "产品市场份额分析"
            ])
        else:
            # 通用推荐
            fallback_recommendations.extend([
                "销售业绩总体概况",
                "利润分析报告",
                "市场趋势洞察"
            ])
        
        # 基于数据特征的推荐
        if not result_df.empty:
            if 'category' in result_df.columns or '类别' in result_df.columns:
                fallback_recommendations.append("产品类别深度分析")
            if 'region' in result_df.columns or '地区' in result_df.columns:
                fallback_recommendations.append("区域市场表现对比")
        
        # 去重并限制数量
        unique_recommendations = list(dict.fromkeys(fallback_recommendations))
        return unique_recommendations[:num_recommendations]

# 全局实例
recommendation_engine = RecommendationEngine()
"""
Intel DeepInsight 查询可能性生成器 - LLM智能版
基于大模型智能分析用户查询中的语义歧义，生成多种理解方式
让用户能和系统清晰无歧义的沟通
"""

import json
import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class QueryPossibility:
    """查询可能性数据结构"""
    rank: int
    description: str
    confidence: float
    key_interpretations: Dict[str, Dict]
    ambiguity_resolutions: Dict[str, Dict]
    generated_sql: Optional[str] = None
    execution_result: Optional[Any] = None
    natural_description: Optional[str] = None
    
    def get_natural_description(self) -> str:
        """生成自然语言描述"""
        if self.natural_description:
            return self.natural_description
        
        if not self.key_interpretations:
            return "标准理解"
        
        desc_parts = []
        for term, interpretation in self.key_interpretations.items():
            if isinstance(interpretation, dict) and 'desc' in interpretation:
                desc_parts.append(f"{term}→{interpretation['desc']}")
            else:
                desc_parts.append(f"{term}→{interpretation}")
        return " | ".join(desc_parts)


class LLMQueryPossibilityGenerator:
    """
    基于LLM的智能查询可能性生成器
    
    核心思路：
    1. 将用户查询发送给LLM，让LLM分析其中可能存在的语义歧义
    2. LLM返回多种可能的理解方式，每种都有清晰的解释
    3. 按用户配置的max_count返回指定数量的可能性
    """
    
    def __init__(self, llm_client=None, model_name: str = None, term_dictionary=None):
        """
        初始化可能性生成器
        
        Args:
            llm_client: OpenAI兼容的LLM客户端
            model_name: 模型名称
            term_dictionary: 术语词典对象
        """
        self.llm_client = llm_client
        self.model_name = model_name
        self.term_dictionary = term_dictionary
    
    def set_llm_client(self, client, model_name: str):
        """设置LLM客户端（延迟初始化）"""
        self.llm_client = client
        self.model_name = model_name
    
    def generate_possibilities(self, query: str, context: str, max_count: int) -> List[QueryPossibility]:
        """
        使用LLM智能分析查询，生成多种可能的理解方式
        
        Args:
            query: 用户的自然语言查询
            context: 数据库Schema上下文
            max_count: 用户配置的最大可能性数量
            
        Returns:
            List[QueryPossibility]: 按置信度排序的可能性列表
        """
        
        # 如果用户只要1种可能性，直接返回标准理解
        if max_count <= 1:
            return [QueryPossibility(
                rank=1,
                description="标准理解",
                confidence=1.0,
                key_interpretations={},
                ambiguity_resolutions={},
                natural_description=f"按字面意思理解：{query}"
            )]
        
        # 如果没有LLM客户端，回退到基础分析
        if not self.llm_client:
            logger.warning("LLM客户端未初始化，使用基础分析")
            return self._fallback_analysis(query, max_count)
        
        try:
            # 使用LLM智能分析歧义
            possibilities = self._llm_analyze_ambiguity(query, context, max_count)
            
            if possibilities and len(possibilities) > 0:
                return possibilities
            else:
                # LLM分析失败，回退到基础分析
                return self._fallback_analysis(query, max_count)
                
        except Exception as e:
            logger.error(f"LLM分析歧义失败: {e}")
            return self._fallback_analysis(query, max_count)
    
    def _llm_analyze_ambiguity(self, query: str, context: str, max_count: int) -> List[QueryPossibility]:
        """
        调用LLM分析查询中的语义歧义
        """
        
        # 构建术语词典上下文
        term_context = ""
        if self.term_dictionary and hasattr(self.term_dictionary, 'terms'):
            terms_list = [f"- {term}: {explanation}" 
                         for term, explanation in list(self.term_dictionary.terms.items())[:20]]
            if terms_list:
                term_context = f"\n\n【业务术语词典】\n" + "\n".join(terms_list)
        
        # 构建Prompt，让LLM分析歧义
        prompt = f"""你是一个专业的数据分析查询理解专家。用户提出了一个数据查询请求，你需要分析这个查询中可能存在的语义歧义，并给出 **恰好 {max_count} 种** 不同的理解方式。

【用户查询】
{query}

【数据库Schema上下文】
{context[:2000] if context else "暂无Schema信息"}{term_context}

【任务要求】
1. 仔细分析用户查询中可能存在歧义的词语或表达
2. 针对每个歧义点，思考不同的理解方式
3. 生成 **恰好 {max_count} 种** 不同的查询理解方式（不多不少）
4. 每种理解方式都应该是合理的、有业务意义的
5. 按照可能性从高到低排序

【歧义分析示例】
- "卖得最好" 可能指：销量最高(数量)、销售额最高(金额)、订单数最多(人气)
- "最近" 可能指：最近7天、最近30天、最近一个季度
- "客户" 可能指：所有客户、活跃客户、VIP客户
- "增长" 可能指：同比增长、环比增长、绝对增长

【输出格式】
请严格按照以下JSON格式输出，必须是有效的JSON数组：
```json
[
  {{
    "rank": 1,
    "interpretation": "对查询的第一种理解方式的完整描述",
    "ambiguity_points": {{
      "歧义词1": "这种理解下的含义",
      "歧义词2": "这种理解下的含义"
    }},
    "confidence": 0.9,
    "reasoning": "为什么这种理解最可能是用户想要的"
  }},
  {{
    "rank": 2,
    "interpretation": "对查询的第二种理解方式的完整描述",
    "ambiguity_points": {{
      "歧义词1": "这种理解下的含义"
    }},
    "confidence": 0.7,
    "reasoning": "这种理解的合理性说明"
  }}
]
```

【重要提醒】
- 必须输出恰好 {max_count} 种理解方式
- 每种理解方式必须有实质性的区别，不能只是措辞不同
- confidence 值范围 0.0-1.0，表示这种理解的可能性
- 如果查询本身没有明显歧义，也要尝试从不同角度给出 {max_count} 种理解
- 只输出JSON数组，不要输出其他内容"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # 解析LLM返回的JSON
            possibilities = self._parse_llm_response(response_text, max_count)
            
            return possibilities
            
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            raise
    
    def _parse_llm_response(self, response_text: str, max_count: int) -> List[QueryPossibility]:
        """
        解析LLM返回的JSON响应
        """
        possibilities = []
        
        try:
            # 尝试提取JSON部分
            json_text = response_text
            
            # 如果包含markdown代码块，提取其中的JSON
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_text = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                json_text = response_text[start:end].strip()
            
            # 尝试找到JSON数组
            if "[" in json_text:
                start = json_text.find("[")
                end = json_text.rfind("]") + 1
                json_text = json_text[start:end]
            
            # 解析JSON
            parsed = json.loads(json_text)
            
            if not isinstance(parsed, list):
                parsed = [parsed]
            
            # 转换为QueryPossibility对象
            for i, item in enumerate(parsed[:max_count]):
                rank = item.get('rank', i + 1)
                interpretation = item.get('interpretation', f"理解方式 {rank}")
                ambiguity_points = item.get('ambiguity_points', {})
                confidence = float(item.get('confidence', 0.5))
                reasoning = item.get('reasoning', '')
                
                # 构建key_interpretations
                key_interpretations = {}
                for term, meaning in ambiguity_points.items():
                    key_interpretations[term] = {
                        'desc': meaning,
                        'confidence': confidence
                    }
                
                possibility = QueryPossibility(
                    rank=rank,
                    description=interpretation,
                    confidence=confidence,
                    key_interpretations=key_interpretations,
                    ambiguity_resolutions=ambiguity_points,
                    natural_description=f"{interpretation}" + (f" ({reasoning})" if reasoning else "")
                )
                possibilities.append(possibility)
            
            # 确保返回指定数量的可能性
            while len(possibilities) < max_count:
                # 如果LLM返回的数量不足，补充默认的可能性
                idx = len(possibilities) + 1
                possibilities.append(QueryPossibility(
                    rank=idx,
                    description=f"备选理解方式 {idx}",
                    confidence=0.3,
                    key_interpretations={},
                    ambiguity_resolutions={},
                    natural_description=f"按常规方式理解查询（备选方案 {idx}）"
                ))
            
            # 按置信度重新排序
            possibilities.sort(key=lambda x: x.confidence, reverse=True)
            
            # 重新分配排名
            for i, p in enumerate(possibilities):
                p.rank = i + 1
            
            return possibilities[:max_count]
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}, 原始响应: {response_text[:500]}")
            raise
    
    def _fallback_analysis(self, query: str, max_count: int) -> List[QueryPossibility]:
        """
        当LLM不可用时的回退分析方法
        基于简单的关键词匹配生成可能性
        """
        possibilities = []
        
        # 定义一些常见的歧义模式
        ambiguity_patterns = [
            {
                "keywords": ["最好", "最佳", "top", "best"],
                "interpretations": [
                    ("按销售数量排名", "销量最高", 0.9),
                    ("按销售金额排名", "销售额最高", 0.85),
                    ("按订单次数排名", "最受欢迎", 0.7),
                    ("按利润排名", "利润最高", 0.6),
                    ("按增长率排名", "增长最快", 0.5),
                ]
            },
            {
                "keywords": ["最近", "近期", "lately", "recent"],
                "interpretations": [
                    ("最近7天", "过去一周", 0.8),
                    ("最近30天", "过去一个月", 0.85),
                    ("最近90天", "过去一个季度", 0.7),
                    ("最近一年", "过去12个月", 0.6),
                    ("本月至今", "当月数据", 0.5),
                ]
            },
            {
                "keywords": ["增长", "上升", "提升", "growth"],
                "interpretations": [
                    ("同比增长", "与去年同期对比", 0.85),
                    ("环比增长", "与上期对比", 0.8),
                    ("绝对增长", "数值变化量", 0.7),
                    ("增长率", "百分比变化", 0.75),
                    ("累计增长", "累计变化", 0.5),
                ]
            },
            {
                "keywords": ["客户", "用户", "顾客", "customer"],
                "interpretations": [
                    ("所有客户", "全部客户数据", 0.9),
                    ("活跃客户", "有近期交易的客户", 0.8),
                    ("新客户", "首次购买的客户", 0.7),
                    ("VIP客户", "高价值客户", 0.6),
                    ("流失客户", "长期未交易的客户", 0.5),
                ]
            },
            {
                "keywords": ["销售", "卖", "售出", "sale"],
                "interpretations": [
                    ("销售数量", "卖出的商品件数", 0.85),
                    ("销售金额", "销售总收入", 0.9),
                    ("销售订单数", "成交订单数量", 0.7),
                    ("销售利润", "销售带来的利润", 0.6),
                    ("销售均价", "平均销售单价", 0.5),
                ]
            },
            {
                "keywords": ["产品", "商品", "货品", "product"],
                "interpretations": [
                    ("所有产品", "全部商品", 0.9),
                    ("在售产品", "当前可购买的商品", 0.8),
                    ("热销产品", "销量较高的商品", 0.7),
                    ("新品", "新上架的商品", 0.6),
                    ("库存产品", "有库存的商品", 0.5),
                ]
            }
        ]
        
        query_lower = query.lower()
        matched_interpretations = []
        
        # 查找匹配的歧义模式
        for pattern in ambiguity_patterns:
            for keyword in pattern["keywords"]:
                if keyword in query_lower or keyword in query:
                    for interp in pattern["interpretations"]:
                        matched_interpretations.append({
                            "keyword": keyword,
                            "desc": interp[0],
                            "meaning": interp[1],
                            "confidence": interp[2]
                        })
                    break
        
        # 如果没有匹配到任何模式，生成通用的可能性
        if not matched_interpretations:
            for i in range(max_count):
                confidence = 0.9 - (i * 0.15)
                possibilities.append(QueryPossibility(
                    rank=i + 1,
                    description=f"理解方式 {i + 1}",
                    confidence=max(0.3, confidence),
                    key_interpretations={},
                    ambiguity_resolutions={},
                    natural_description=f"按常规方式理解查询（方案 {i + 1}）"
                ))
            return possibilities
        
        # 去重并按置信度排序
        seen = set()
        unique_interpretations = []
        for interp in matched_interpretations:
            key = interp["desc"]
            if key not in seen:
                seen.add(key)
                unique_interpretations.append(interp)
        
        unique_interpretations.sort(key=lambda x: x["confidence"], reverse=True)
        
        # 生成可能性
        for i, interp in enumerate(unique_interpretations[:max_count]):
            possibility = QueryPossibility(
                rank=i + 1,
                description=interp["desc"],
                confidence=interp["confidence"],
                key_interpretations={
                    interp["keyword"]: {
                        "desc": interp["meaning"],
                        "confidence": interp["confidence"]
                    }
                },
                ambiguity_resolutions={
                    interp["keyword"]: interp["meaning"]
                },
                natural_description=f"将「{interp['keyword']}」理解为「{interp['meaning']}」: {interp['desc']}"
            )
            possibilities.append(possibility)
        
        # 如果数量不足，补充默认的可能性
        while len(possibilities) < max_count:
            idx = len(possibilities) + 1
            possibilities.append(QueryPossibility(
                rank=idx,
                description=f"备选理解方式 {idx}",
                confidence=0.3,
                key_interpretations={},
                ambiguity_resolutions={},
                natural_description=f"按字面意思理解查询（备选方案 {idx}）"
            ))
        
        return possibilities[:max_count]


# 为了向后兼容，保留原始类名
class QueryPossibilityGenerator(LLMQueryPossibilityGenerator):
    """向后兼容的别名"""
    pass


# 保留原始的EnhancedQueryPossibilityGenerator作为别名
EnhancedQueryPossibilityGenerator = LLMQueryPossibilityGenerator

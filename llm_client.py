import os
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd

# 加载 .env 中的 API Key
load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("❌ 未在 .env 文件中找到 DEEPSEEK_API_KEY，请检查配置。")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)


def generate_sql(query, schema_context):
    """调用 DeepSeek R1 (Reasoner) 生成 SQL"""

    # System Prompt: 赋予 AI 数据库专家的角色，并规定严格的输出格式
    system_prompt = """你是一个 SQLite 数据分析专家。请根据提供的表结构上下文，将用户的自然语言问题转换为可执行的 SQL 查询语句。

    【核心规则】
    1. 表名必须是: sales_orders
    2. 只要涉及“利润率”，计算公式必须是: SUM(profit) * 1.0 / SUM(sales)
    3. 日期字段 (order_date) 是字符串格式 'YYYY-MM-DD'。如果用户问“每年”，请使用 strftime('%Y', order_date)；如果问“每月”，请使用 strftime('%Y-%m', order_date)。
    4. 严禁输出 Markdown 格式（如 ```sql ... ```），请直接输出纯 SQL 文本。
    5. 不要解释你的思考过程，直接给代码。
    """

    user_prompt = f"""
    【数据库 Schema 信息】
    {schema_context}

    【用户问题】
    {query}

    【请生成 SQL】
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",  # 使用 R1 推理模型
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"API_ERROR: {str(e)}"


def generate_insight(query, df):
    """调用 DeepSeek Chat (V3) 生成商业洞察"""
    if df.empty:
        return "数据为空，无法生成洞察。"

    # 将数据前 5 行转为 Markdown 供 AI 阅读
    data_preview = df.head(5).to_markdown(index=False)

    prompt = f"""
    用户的问题是："{query}"
    查询到的数据结果如下（前5行）：
    {data_preview}

    请扮演一位资深商业数据分析师，用简练专业的语言（50字以内），根据上述数据给出一句核心洞察或业务建议。
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",  # 使用 V3 生成文本，速度快且便宜
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return "无法生成洞察 (网络或额度原因)"
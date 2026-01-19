import pandas as pd
import sqlite3
import os
import sys

# 路径配置
CSV_PATH = 'data/superstore.csv'
DB_PATH = 'data/ecommerce.db'


def clean_column_name(col_name):
    """将列名转换为 snake_case (小写+下划线)"""
    return str(col_name).strip().lower().replace('-', '_').replace(' ', '_')


def run_etl():
    print(f"🚀 开始 ETL 流程...")
    print(f"📂 正在读取文件: {CSV_PATH}")

    df = None

    # --- 阶段 1: 智能文件读取 ---
    try:
        # 尝试 1: 标准 CSV 读取 (UTF-8)
        df = pd.read_csv(CSV_PATH, encoding='utf-8')
    except (UnicodeDecodeError, pd.errors.ParserError):
        print("⚠️ 发现文件可能是 Excel 格式 (或编码错误)，尝试切换读取模式...")
        try:
            # 尝试 2: 强制作为 Excel 读取 (即使后缀是 .csv)
            # engine='xlrd' 用于 .xls, engine='openpyxl' 用于 .xlsx
            # 这里先尝试 xlrd (对应 0xd0 错误)
            try:
                df = pd.read_excel(CSV_PATH, engine='xlrd')
            except ImportError:
                print("❌ 错误: 缺少 'xlrd' 库。请在终端运行: pip install xlrd")
                return
            except Exception:
                # 如果 xlrd 失败，尝试 openpyxl
                df = pd.read_excel(CSV_PATH, engine='openpyxl')

        except Exception as e:
            print(f"❌ 读取失败: 无法识别该文件格式。请确认它是有效的 CSV 或 Excel 文件。\n详细错误: {e}")
            return

    print(f"✅ 成功读取 {len(df)} 行数据")

    # --- 阶段 2: 数据清洗 ---
    # 字段标准化
    df.columns = [clean_column_name(c) for c in df.columns]
    print(f"📝 列名已标准化: {list(df.columns)}")

    # 必要的列检查
    required_cols = ['order_date', 'sales', 'profit']
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ 严重错误: 数据中缺少必要列 '{col}'。请检查源文件。")
            return

    # 日期清洗 (兼容 Excel 序列值 和 字符串)
    print("⏳ 正在转换日期格式...")

    def parse_date(x):
        # 如果是 Excel 序列数字 (float/int)
        if isinstance(x, (float, int)):
            return pd.to_datetime(x, unit='D', origin='1899-12-30')
        # 如果是字符串
        return pd.to_datetime(x)

    for col in ['order_date', 'ship_date']:
        if col in df.columns:
            try:
                df[col] = df[col].apply(parse_date)
                df[col] = df[col].dt.strftime('%Y-%m-%d')
            except Exception as e:
                print(f"⚠️ 警告: 列 {col} 日期转换部分失败: {e}")

    # --- 阶段 3: 入库 ---
    conn = sqlite3.connect(DB_PATH)
    df.to_sql('sales_orders', conn, if_exists='replace', index=False)

    # 索引优化
    cursor = conn.cursor()
    try:
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON sales_orders (order_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON sales_orders (category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_region ON sales_orders (region)')
        conn.commit()
    except Exception as e:
        print(f"⚠️ 索引创建警告: {e}")

    conn.close()
    print(f"🎉 ETL 完成！数据库已保存至: {DB_PATH}")
    print("👀 数据预览 (前3条):")
    print(df[['order_date', 'sales', 'profit']].head(3))


if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print(f"❌ 错误: 未找到文件 {CSV_PATH}。请将 superstore.csv 放入 data/ 目录。")
    else:
        run_etl()
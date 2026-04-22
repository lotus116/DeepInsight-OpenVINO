# setup_db.py
import argparse
import os
import pymysql

def execute_sql_file(connection, cursor, file_path):
    """安全读取并执行 SQL 文件中的每条语句"""
    with open(file_path, 'r', encoding='utf-8') as f:
        sql = f.read()

    # 分割语句（按 ; 分割，忽略空语句）
    statements = [s.strip() for s in sql.split(';') if s.strip()]
    
    for stmt in statements:
        try:
            cursor.execute(stmt)
        except Exception as e:
            print(f"⚠️ 跳过错误语句（前100字符）:\n{stmt[:100]}...\n错误: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(description="创建 MySQL 数据库并导入 SQL 文件（使用 PyMySQL）")
    parser.add_argument("--user", required=True, help="MySQL 用户名")
    parser.add_argument("--password", required=True, help="MySQL 密码")
    parser.add_argument("--db", required=True, help="要创建和使用的数据库名")
    parser.add_argument("--sql", required=True, help="SQL 文件路径（支持中文路径）")
    parser.add_argument("--host", default="localhost", help="MySQL 主机地址（默认: localhost）")
    parser.add_argument("--port", type=int, default=3306, help="MySQL 端口（默认: 3306）")

    args = parser.parse_args()

    # 检查 SQL 文件是否存在
    if not os.path.isfile(args.sql):
        print(f"❌ 错误：SQL 文件不存在 → {args.sql}")
        return

    conn = None
    try:
        # 第一步：连接 MySQL（不指定数据库）
        conn = pymysql.connect(
            host=args.host,
            port=args.port,
            user=args.user,
            password=args.password,
            charset='utf8mb4',
            autocommit=False  # 手动控制事务
        )
        cursor = conn.cursor()

        # 第二步：创建数据库（使用反引号避免关键字冲突）
        safe_db = args.db.replace('`', '')  # 简单过滤反引号（数据库名本身不应包含）
        create_db_sql = f"CREATE DATABASE IF NOT EXISTS `{safe_db}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
        cursor.execute(create_db_sql)
        print(f"✅ 数据库 `{safe_db}` 已创建（或已存在）")

        # 第三步：切换到该数据库
        cursor.execute(f"USE `{safe_db}`;")

        # 第四步：执行 SQL 文件
        print(f"📥 正在执行 SQL 文件: {args.sql}")
        execute_sql_file(conn, cursor, args.sql)

        # 提交所有更改
        conn.commit()
        print("🎉 数据库初始化完成！")

    except pymysql.MySQLError as e:
        print(f"❌ MySQL 错误: {e}")
        if conn:
            conn.rollback()
    except Exception as e:
        print(f"💥 其他错误: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn and conn.open:
            conn.close()
            print("🔌 数据库连接已关闭")

if __name__ == "__main__":
    main()
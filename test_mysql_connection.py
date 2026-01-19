#!/usr/bin/env python3
"""
MySQL连接测试工具
用于测试MySQL数据库连接配置是否正确
"""

import sys
import traceback
from sqlalchemy import create_engine, text
import pymysql

def test_mysql_connection(host, port, user, password, database):
    """
    测试MySQL连接
    
    Args:
        host: 主机地址
        port: 端口号
        user: 用户名
        password: 密码
        database: 数据库名
    
    Returns:
        dict: 测试结果
    """
    result = {
        "success": False,
        "message": "",
        "details": {},
        "error": None
    }
    
    try:
        # 构建连接字符串
        connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        
        print(f"🔗 正在测试MySQL连接...")
        print(f"   主机: {host}")
        print(f"   端口: {port}")
        print(f"   用户: {user}")
        print(f"   数据库: {database}")
        print(f"   连接字符串: mysql+pymysql://{user}:***@{host}:{port}/{database}")
        print()
        
        # 1. 测试基础连接
        print("1️⃣ 测试基础连接...")
        engine = create_engine(
            connection_string,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={
                "connect_timeout": 10,
                "read_timeout": 30,
                "write_timeout": 30
            }
        )
        
        # 2. 测试连接池
        print("2️⃣ 测试连接池...")
        with engine.connect() as conn:
            # 3. 测试基本查询
            print("3️⃣ 测试基本查询...")
            version_result = conn.execute(text("SELECT VERSION() as version"))
            version = version_result.fetchone()[0]
            
            # 4. 测试数据库信息
            print("4️⃣ 获取数据库信息...")
            db_info = conn.execute(text("SELECT DATABASE() as current_db")).fetchone()[0]
            
            # 5. 测试表列表
            print("5️⃣ 获取表列表...")
            tables_result = conn.execute(text("SHOW TABLES"))
            tables = [row[0] for row in tables_result.fetchall()]
            
            # 6. 测试字符集
            print("6️⃣ 检查字符集...")
            charset_result = conn.execute(text(
                "SELECT @@character_set_database as charset, @@collation_database as collation"
            ))
            charset_info = charset_result.fetchone()
            
            result.update({
                "success": True,
                "message": "MySQL连接测试成功！",
                "details": {
                    "mysql_version": version,
                    "current_database": db_info,
                    "table_count": len(tables),
                    "tables": tables[:10],  # 只显示前10个表
                    "charset": charset_info[0] if charset_info else "unknown",
                    "collation": charset_info[1] if charset_info else "unknown",
                    "connection_string": connection_string
                }
            })
            
        print("✅ 所有测试通过！")
        
    except pymysql.err.OperationalError as e:
        error_code, error_msg = e.args
        result.update({
            "success": False,
            "message": f"MySQL连接失败 (错误代码: {error_code})",
            "error": error_msg,
            "details": {
                "error_type": "OperationalError",
                "error_code": error_code,
                "suggestions": get_connection_suggestions(error_code)
            }
        })
        
    except Exception as e:
        result.update({
            "success": False,
            "message": f"连接测试失败: {str(e)}",
            "error": str(e),
            "details": {
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
        })
    
    return result

def get_connection_suggestions(error_code):
    """根据错误代码提供建议"""
    suggestions = {
        1045: [
            "检查用户名和密码是否正确",
            "确认用户是否有访问该数据库的权限",
            "尝试使用MySQL客户端直接连接测试"
        ],
        2003: [
            "检查MySQL服务是否正在运行",
            "确认主机地址和端口号是否正确",
            "检查防火墙设置是否阻止了连接",
            "确认MySQL配置允许远程连接"
        ],
        1049: [
            "检查数据库名称是否正确",
            "确认数据库是否存在",
            "尝试先连接到mysql系统数据库"
        ],
        1044: [
            "检查用户是否有访问该数据库的权限",
            "联系数据库管理员分配相应权限"
        ]
    }
    
    return suggestions.get(error_code, [
        "检查所有连接参数是否正确",
        "确认MySQL服务正常运行",
        "查看MySQL错误日志获取更多信息"
    ])

def print_test_result(result):
    """打印测试结果"""
    print("\n" + "="*60)
    print("📊 MySQL连接测试结果")
    print("="*60)
    
    if result["success"]:
        print("✅ 状态: 连接成功")
        print(f"💬 消息: {result['message']}")
        
        details = result["details"]
        print(f"\n📋 数据库信息:")
        print(f"   MySQL版本: {details.get('mysql_version', 'N/A')}")
        print(f"   当前数据库: {details.get('current_database', 'N/A')}")
        print(f"   字符集: {details.get('charset', 'N/A')}")
        print(f"   排序规则: {details.get('collation', 'N/A')}")
        print(f"   表数量: {details.get('table_count', 0)}")
        
        if details.get('tables'):
            print(f"   表列表 (前10个): {', '.join(details['tables'])}")
            
    else:
        print("❌ 状态: 连接失败")
        print(f"💬 消息: {result['message']}")
        
        if result.get('error'):
            print(f"🔍 错误详情: {result['error']}")
            
        details = result.get("details", {})
        if details.get('suggestions'):
            print(f"\n💡 建议解决方案:")
            for i, suggestion in enumerate(details['suggestions'], 1):
                print(f"   {i}. {suggestion}")
    
    print("="*60)

def main():
    """主函数 - 命令行交互"""
    print("🔧 MySQL连接测试工具")
    print("="*40)
    
    try:
        # 获取连接参数
        host = input("请输入MySQL主机地址 (默认: localhost): ").strip() or "localhost"
        port = input("请输入MySQL端口 (默认: 3306): ").strip() or "3306"
        user = input("请输入MySQL用户名 (默认: root): ").strip() or "root"
        password = input("请输入MySQL密码: ").strip()
        database = input("请输入数据库名称: ").strip()
        
        if not database:
            print("❌ 数据库名称不能为空！")
            return
            
        # 转换端口为整数
        try:
            port = int(port)
        except ValueError:
            print("❌ 端口号必须是数字！")
            return
            
        # 执行测试
        result = test_mysql_connection(host, port, user, password, database)
        
        # 打印结果
        print_test_result(result)
        
        # 如果成功，提供连接字符串
        if result["success"]:
            connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
            print(f"\n🔗 可用的连接字符串:")
            print(f"   {connection_string}")
            print(f"\n💡 您可以将此连接字符串复制到DeepInsight的数据库配置中")
            
    except KeyboardInterrupt:
        print("\n\n👋 测试已取消")
    except Exception as e:
        print(f"\n❌ 程序错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # 检查依赖
    try:
        import pymysql
        import sqlalchemy
    except ImportError as e:
        print(f"❌ 缺少依赖库: {e}")
        print("请安装: pip install pymysql sqlalchemy")
        sys.exit(1)
    
    main()
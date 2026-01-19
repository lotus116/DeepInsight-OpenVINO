import json
import os
import uuid
from datetime import datetime

CONFIG_FILE = "data/config.json"
HISTORY_FILE = "data/history.json"

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f: 
                config = json.load(f)
                # 向后兼容：添加缺失的新配置项
                if "enable_ai_recommendations" not in config:
                    config["enable_ai_recommendations"] = True
                if "recommendation_use_separate_api" not in config:
                    config["recommendation_use_separate_api"] = False
                if "recommendation_api_base" not in config:
                    config["recommendation_api_base"] = "https://api.deepseek.com"
                if "recommendation_api_key" not in config:
                    config["recommendation_api_key"] = ""
                if "recommendation_model_name" not in config:
                    config["recommendation_model_name"] = "deepseek-reasoner"
                if "enable_history_context" not in config:
                    config["enable_history_context"] = True
                if "max_context_items" not in config:
                    config["max_context_items"] = 3
                
                # 新增：分离的数据库配置
                if "sqlite_config" not in config:
                    config["sqlite_config"] = {
                        "db_path": "data/ecommerce.db",
                        "schema_path": "data/schema_northwind.json"
                    }
                if "mysql_config" not in config:
                    config["mysql_config"] = {
                        "host": "localhost",
                        "port": "3306",
                        "user": "root",
                        "password": "",
                        "database": "ecommerce",
                        "schema_path": ""
                    }
                
                return config
        except: pass
    return {
        "api_key": "", "api_base": "https://api.deepseek.com", "model_name": "deepseek-reasoner",
        "db_type": "SQLite", "db_path": "data/ecommerce.db", "db_uris": ["sqlite:///data/ecommerce.db"],
        "schema_path": "data/schema_northwind.json", "kb_paths_list": [], "model_path": "models/bge-small-ov",
        "max_retries": 3, "max_candidates": 1, "log_file": "data/agent.log",
        "enable_ai_recommendations": True,
        "recommendation_use_separate_api": False,
        "recommendation_api_base": "https://api.deepseek.com",
        "recommendation_api_key": "",
        "recommendation_model_name": "deepseek-reasoner",
        "enable_history_context": True,
        "max_context_items": 3,
        # 分离的数据库配置
        "sqlite_config": {
            "db_path": "data/ecommerce.db",
            "schema_path": "data/schema_northwind.json"
        },
        "mysql_config": {
            "host": "localhost",
            "port": "3306",
            "user": "root",
            "password": "",
            "database": "ecommerce",
            "schema_path": ""
        }
    }

def save_config(config):
    os.makedirs("data", exist_ok=True)
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f: json.dump(config, f, indent=4)

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f: return json.load(f)
        except: pass
    return {}

def save_history(history):
    os.makedirs("data", exist_ok=True)
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f: json.dump(history, f, indent=4)

def create_new_session(history):
    sid = str(uuid.uuid4())
    history[sid] = {"title": f"新对话 {datetime.now().strftime('%H:%M')}", "messages": []}
    save_history(history)
    return sid, history

def delete_session(history, sid):
    """删除指定会话"""
    if sid in history:
        del history[sid]
        save_history(history)
    return history

def update_session_messages(sid, msgs, history):
    if sid in history:
        history[sid]["messages"] = msgs
        # 自动更新标题：取第一条消息的前10个字
        if len(msgs) > 0 and history[sid]["title"].startswith("新对话"):
            first_msg = msgs[0]["content"]
            history[sid]["title"] = (first_msg[:10] + "...") if len(first_msg) > 10 else first_msg
        save_history(history)
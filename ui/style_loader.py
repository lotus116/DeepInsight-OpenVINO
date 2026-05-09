"""
Intel® DeepInsight - 样式加载器
提供统一的样式和脚本注入功能
"""

import streamlit as st
import os
from pathlib import Path


def get_styles_dir() -> Path:
    """获取样式目录路径"""
    return Path(__file__).parent / "styles"


def load_css_file(filename: str = "main.css") -> str:
    """加载CSS文件内容"""
    css_path = get_styles_dir() / filename
    if css_path.exists():
        return css_path.read_text(encoding="utf-8")
    return ""


def load_js_file(filename: str = "scripts.js") -> str:
    """加载JavaScript文件内容"""
    js_path = get_styles_dir() / filename
    if js_path.exists():
        return js_path.read_text(encoding="utf-8")
    return ""


def inject_styles_and_scripts():
    """
    注入CSS样式和JavaScript脚本到Streamlit应用
    
    使用方法:
        from ui.style_loader import inject_styles_and_scripts
        inject_styles_and_scripts()
    """
    # 加载CSS
    css_content = load_css_file("main.css")
    
    # 加载JavaScript
    js_content = load_js_file("scripts.js")
    
    # 构建HTML
    html_content = f"""
<style>
{css_content}
</style>

<!-- 移动端侧边栏控制和键盘快捷键JavaScript -->
<script>
{js_content}
</script>
"""
    
    # 注入到页面
    st.markdown(html_content, unsafe_allow_html=True)


def inject_custom_css(css_string: str):
    """
    注入自定义CSS字符串
    
    Args:
        css_string: CSS样式字符串
    """
    st.markdown(f"<style>{css_string}</style>", unsafe_allow_html=True)


def inject_custom_js(js_string: str):
    """
    注入自定义JavaScript字符串
    
    Args:
        js_string: JavaScript代码字符串
    """
    st.markdown(f"<script>{js_string}</script>", unsafe_allow_html=True)


# 便捷函数 - 一行调用即可加载所有样式
def load_all_styles():
    """
    加载所有DeepInsight样式和脚本的便捷函数
    
    使用方法:
        from ui.style_loader import load_all_styles
        load_all_styles()
    """
    inject_styles_and_scripts()


if __name__ == "__main__":
    # 测试样式加载
    print(f"样式目录: {get_styles_dir()}")
    print(f"CSS文件存在: {(get_styles_dir() / 'main.css').exists()}")
    print(f"JS文件存在: {(get_styles_dir() / 'scripts.js').exists()}")

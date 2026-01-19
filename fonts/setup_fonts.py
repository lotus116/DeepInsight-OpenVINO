#!/usr/bin/env python3
"""
字体安装脚本
自动下载并安装PDF导出所需的中文字体
"""

import os
import sys
import requests
from pathlib import Path

def download_font(url, filename, description):
    """下载字体文件"""
    font_dir = Path("fonts")
    font_dir.mkdir(exist_ok=True)
    
    font_path = font_dir / filename
    
    if font_path.exists():
        print(f"✅ {description} 已存在: {font_path}")
        return True
    
    print(f"📥 正在下载 {description}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(font_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ {description} 下载完成: {font_path}")
        return True
        
    except Exception as e:
        print(f"❌ {description} 下载失败: {e}")
        return False

def setup_fonts():
    """设置字体"""
    print("🚀 开始设置PDF导出字体...")
    
    # 创建字体目录
    font_dir = Path("fonts")
    font_dir.mkdir(exist_ok=True)
    
    # 字体下载列表（使用开源字体）
    fonts = [
        {
            "url": "https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf",
            "filename": "SourceHanSansSC-Regular.otf",
            "description": "思源黑体 (简体中文)"
        },
        {
            "url": "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
            "filename": "NotoSansCJKsc-Regular.otf", 
            "description": "Noto Sans CJK (简体中文)"
        }
    ]
    
    success_count = 0
    
    for font in fonts:
        if download_font(font["url"], font["filename"], font["description"]):
            success_count += 1
    
    if success_count > 0:
        print(f"\n🎉 成功安装 {success_count} 个字体文件")
        print("💡 现在可以正常导出包含中文的PDF报告了")
    else:
        print("\n⚠️ 没有成功下载任何字体文件")
        print("💡 请手动下载中文字体文件并放入 fonts/ 文件夹")
        print("   推荐字体：")
        print("   - 思源黑体: https://github.com/adobe-fonts/source-han-sans")
        print("   - Noto Sans CJK: https://fonts.google.com/noto/specimen/Noto+Sans+SC")
    
    return success_count > 0

def check_fonts():
    """检查字体状态"""
    print("🔍 检查字体状态...")
    
    font_dir = Path("fonts")
    if not font_dir.exists():
        print("❌ fonts 文件夹不存在")
        return False
    
    font_files = list(font_dir.glob("*.ttf")) + list(font_dir.glob("*.otf")) + list(font_dir.glob("*.ttc"))
    
    if not font_files:
        print("❌ 未找到字体文件")
        return False
    
    print(f"✅ 找到 {len(font_files)} 个字体文件:")
    for font_file in font_files:
        size_mb = font_file.stat().st_size / (1024 * 1024)
        print(f"  📄 {font_file.name} ({size_mb:.1f} MB)")
    
    return True

def main():
    """主函数"""
    print("🎨 Intel® DeepInsight 字体安装工具")
    print("=" * 50)
    
    # 检查当前字体状态
    if check_fonts():
        print("\n✅ 字体已安装，无需重新下载")
        return 0
    
    # 询问用户是否要下载字体
    try:
        choice = input("\n是否要自动下载开源中文字体？(y/n): ").lower().strip()
        
        if choice in ['y', 'yes', '是']:
            if setup_fonts():
                print("\n🎉 字体安装完成！")
                return 0
            else:
                print("\n❌ 字体安装失败")
                return 1
        else:
            print("\n💡 请手动将中文字体文件放入 fonts/ 文件夹")
            return 0
            
    except KeyboardInterrupt:
        print("\n\n👋 用户取消操作")
        return 0
    except Exception as e:
        print(f"\n❌ 操作失败: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
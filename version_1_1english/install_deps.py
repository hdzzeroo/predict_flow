#!/usr/bin/env python3
"""
依赖安装脚本
自动安装项目所需的所有Python包
"""

import subprocess
import sys
import os

def install_package(package_name):
    """
    安装单个Python包
    """
    try:
        print(f"📦 正在安装 {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ {package_name} 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {package_name} 安装失败: {e}")
        return False

def main():
    """
    安装所有必要的依赖包
    """
    print("🚀 LLM文件路径生成器 - 依赖安装")
    print("=" * 50)
    
    # 必需的包
    required_packages = [
        'openai',           # OpenAI API客户端
        'python-dotenv',    # .env文件支持
        'pandas',           # 数据处理
        'numpy',            # 数值计算
        'matplotlib',       # 图表绘制
    ]
    
    # 可选的包（用于聚类功能）
    optional_packages = [
        'scikit-learn',     # 机器学习库
        'scipy',            # 科学计算
    ]
    
    print("🔧 将安装以下必需包:")
    for pkg in required_packages:
        print(f"  - {pkg}")
    
    print("\n📋 可选包（用于聚类功能）:")
    for pkg in optional_packages:
        print(f"  - {pkg}")
    
    # 询问用户
    choice = input("\n是否开始安装必需包? (y/n, 默认y): ").strip().lower()
    if choice == 'n':
        print("❌ 取消安装")
        return
    
    # 安装必需包
    print("\n📦 开始安装必需包...")
    success_count = 0
    for package in required_packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n✅ 必需包安装完成: {success_count}/{len(required_packages)} 成功")
    
    # 询问是否安装可选包
    if success_count == len(required_packages):
        choice = input("\n是否安装可选包（聚类功能）? (y/n, 默认y): ").strip().lower()
        if choice != 'n':
            print("\n📦 开始安装可选包...")
            optional_success = 0
            for package in optional_packages:
                if install_package(package):
                    optional_success += 1
            print(f"\n✅ 可选包安装完成: {optional_success}/{len(optional_packages)} 成功")
    
    print("\n🎉 安装完成!")
    print("=" * 30)
    print("下一步:")
    print("1. 运行 'python setup_env.py' 来设置 .env 文件")
    print("2. 运行 'python chat_test.py' 来测试LLM功能")

if __name__ == "__main__":
    main() 
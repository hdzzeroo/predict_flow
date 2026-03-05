"""
配置文件 - 管理API密钥和系统设置
"""

import os
from typing import Optional

class Config:
    """
    系统配置类
    """
    
    def __init__(self):
        # OpenAI API配置
        self.openai_api_key: Optional[str] = None
        self.openai_model: str = "gpt-4o"  # 使用gpt-4o (更强的推理能力)
        self.openai_temperature: float = 0.0  # 设置为0，确保输出结果完全一致
        self.openai_max_tokens: int = 8000  # 足够输出detailed thinking + hotspots
        self.openai_timeout: int = 60  # gpt-4o稍慢，保持60秒超时
        
        # 数据路径配置（相对于当前文件位置）
        self.data_base_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        
        # LLM配置
        self.use_real_llm: bool = True  # 是否使用真实LLM API
        self.fallback_on_error: bool = True  # API失败时是否降级
        
        # 加载配置
        self.load_config()
    
    def load_config(self):
        """
        从环境变量或配置文件加载设置
        """
        # 从环境变量加载OpenAI API密钥
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        # 从环境变量加载其他设置
        if os.getenv('OPENAI_MODEL'):
            self.openai_model = os.getenv('OPENAI_MODEL')
        
        if os.getenv('DATA_BASE_DIR'):
            self.data_base_dir = os.getenv('DATA_BASE_DIR')
    
    def set_openai_api_key(self, api_key: str):
        """
        设置OpenAI API密钥
        """
        self.openai_api_key = api_key
        print("✅ OpenAI API密钥已设置")
    
    def get_openai_api_key(self) -> Optional[str]:
        """
        获取OpenAI API密钥
        """
        return self.openai_api_key
    
    def is_llm_available(self) -> bool:
        """
        检查LLM是否可用
        """
        return self.use_real_llm and self.openai_api_key is not None
    
    def print_config(self):
        """
        打印当前配置（不显示敏感信息）
        """
        print("📋 当前配置:")
        print(f"  🤖 OpenAI模型: {self.openai_model}")
        print(f"  🌡️  温度: {self.openai_temperature}")
        print(f"  📊 最大Token: {self.openai_max_tokens}")
        print(f"  ⏱️  超时: {self.openai_timeout}秒")
        print(f"  📁 数据目录: {self.data_base_dir}")
        print(f"  🔑 API密钥: {'已设置' if self.openai_api_key else '未设置'}")
        print(f"  🤖 使用真实LLM: {'是' if self.use_real_llm else '否'}")

# 全局配置实例
config = Config()

def setup_api_key():
    """
    交互式设置API密钥
    """
    if config.openai_api_key:
        print(f"✅ API密钥已存在: {config.openai_api_key[:8]}...")
        choice = input("是否要更新API密钥? (y/n, 默认n): ").strip().lower()
        if choice not in ['y', 'yes']:
            return
    
    print("\n🔑 设置OpenAI API密钥")
    print("─" * 30)
    print("你可以通过以下方式设置API密钥:")
    print("1. 直接输入")
    print("2. 设置环境变量 OPENAI_API_KEY")
    print("3. 跳过（使用本地解析）")
    
    choice = input("\n请选择 (1/2/3, 默认1): ").strip() or "1"
    
    if choice == "1":
        api_key = input("请输入你的OpenAI API密钥: ").strip()
        if api_key:
            config.set_openai_api_key(api_key)
        else:
            print("❌ API密钥不能为空")
    
    elif choice == "2":
        print("请在终端中运行:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print("然后重新启动程序")
    
    elif choice == "3":
        print("⚠️ 跳过API密钥设置，将使用本地正则表达式解析")
        config.use_real_llm = False
    
    else:
        print("❌ 无效选择")

if __name__ == "__main__":
    # 测试配置
    config.print_config()
    setup_api_key()
    config.print_config() 
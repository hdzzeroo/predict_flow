# Version 1.2 - 快速命令参考

## 目录位置
```bash
cd /home/dizhihuang/graduate/predict_workflow/version1_2
```

## 迁移命令历史

### 1. 创建新目录
```bash
cd /home/dizhihuang/graduate/predict_workflow
mkdir -p version1_2
```

### 2. 复制核心代码文件
```bash
cd version_1_1english
cp -v test_complete_workflow.py implementation.py functions.py config.py \
      stub.py spec.yml excel_output_generator.py ../version1_2/
```

### 3. 复制配置和文档
```bash
cp -v README.md PROJECT_SUMMARY.md .env ../version1_2/
```

### 4. 创建输出目录
```bash
mkdir -p ../version1_2/output
```

## 日常使用命令

### 运行主程序
```bash
python test_complete_workflow.py
```

### 查看项目结构
```bash
tree -L 2
```

### 查看文件大小
```bash
ls -lh *.py
```

### 清理输出文件
```bash
rm -rf output/*.png output/*.csv output/*.json
```

### 返回源目录对比
```bash
cd ../version_1_1english
diff config.py ../version1_2/config.py
```

## 数据文件路径
```bash
# 原始数据目录
/home/dizhihuang/graduate/predict_workflow/data/processed_data/

# 查看可用数据文件
ls -lh ../data/processed_data/ | grep 関越道
```

## Git操作（如果需要版本控制）
```bash
# 初始化仓库
git init

# 添加文件
git add *.py *.yml *.md

# 创建.gitignore
cat > .gitignore << 'GITIGNORE'
__pycache__/
*.pyc
*.pyo
output/*.png
output/*.csv
output/*.json
.env
GITIGNORE

# 首次提交
git commit -m "Initial commit: Version 1.2 - Core functionality"
```

## 依赖检查
```bash
# 检查Python版本
python --version

# 检查已安装包
pip list | grep -E "(pandas|numpy|matplotlib|sklearn|openai)"

# 安装缺失依赖
pip install pandas numpy matplotlib scikit-learn openai python-dotenv
```

## 测试运行
```bash
# 快速测试（使用Python脚本）
python -c "
from implementation import compiled_agent
result = compiled_agent.invoke({'user_input': '関越道2024年4月数据'})
print('✅ 测试成功')
"
```

## 文件对比
```bash
# 对比两个版本的差异
diff -r ../version_1_1english/ . --exclude='*.pyc' --exclude='__pycache__' \
     --exclude='output' --exclude='test_*' --exclude='*validation*' --exclude='llm_chatbot.py'
```

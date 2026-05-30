#!/bin/bash
# 评估系统快速启动脚本

echo "=========================================="
echo "交通拥堵预测评估系统"
echo "=========================================="
echo ""

# 检查配置文件是否存在
if [ ! -f "evaluation_config.json" ]; then
    echo "❌ 配置文件不存在: evaluation_config.json"
    echo "请先创建配置文件，或使用模板:"
    echo "  cp evaluation_config.json.template evaluation_config.json"
    exit 1
fi

# 显示菜单
echo "请选择操作:"
echo "  1. 测试评估模块（使用模拟数据）"
echo "  2. 测试评估模块（使用真实数据）"
echo "  3. 批量评估（使用配置文件）"
echo "  4. 查看评估结果"
echo ""
read -p "请输入选项 (1-4): " choice

case $choice in
    1)
        echo ""
        echo "运行测试（模拟数据）..."
        python test_evaluation.py --mode sample
        ;;
    2)
        echo ""
        echo "运行测试（真实数据）..."
        python test_evaluation.py --mode real
        ;;
    3)
        echo ""
        echo "运行批量评估..."
        python batch_evaluation.py --config evaluation_config.json
        ;;
    4)
        echo ""
        echo "评估结果文件列表:"
        ls -lh output/evaluation/*.json 2>/dev/null || echo "暂无评估结果"
        echo ""
        read -p "按Enter键继续..."
        ;;
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "操作完成"
echo "=========================================="

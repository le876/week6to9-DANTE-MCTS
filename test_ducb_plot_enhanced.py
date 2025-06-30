#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试增强版的DUCB趋势图功能，包括模型预测值的负值曲线
"""

import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append('/media/ubuntu/19C1027D35EB273A/ML_projects/ML_training/week6')

from rosenbrock.plot_utils import plot_nte_ducb_trends

def create_test_ducb_data():
    """创建测试用的DUCB趋势数据"""
    np.random.seed(42)  # 固定随机种子以便复现
    
    ducb_trends_data = []
    
    for i in range(3):  # 三个初始点
        rollout_rounds = 50
        rollout_indices = list(range(rollout_rounds))
        
        # 生成DUCB值（模拟递减然后震荡的趋势）
        base_ducb = 10.0 - i * 2.0  # 不同起始点有不同的基础DUCB值
        ducb_values = []
        for j in range(rollout_rounds):
            trend_factor = np.exp(-j/20)  # 指数衰减
            noise = np.random.normal(0, 0.5)  # 随机噪声
            ducb_val = base_ducb * trend_factor + noise
            ducb_values.append(ducb_val)
        
        # 生成模型预测值的负值（与DUCB相关但不完全相同）
        model_pred_values = []
        for j in range(rollout_rounds):
            # 模型预测值应该相对稳定，变化较小
            base_pred = 5.0 + i * 1.5  # 不同起始点有不同的基础预测值
            pred_noise = np.random.normal(0, 0.2)
            model_pred_val = base_pred + pred_noise
            model_pred_values.append(model_pred_val)
        
        # 随机生成一些节点变更位置
        node_changes = []
        for _ in range(np.random.randint(3, 8)):  # 每个点3-7次节点变更
            change_idx = np.random.randint(5, rollout_rounds-5)
            if change_idx not in node_changes:
                node_changes.append(change_idx)
        node_changes.sort()
        
        trend_data = {
            'ducb_values': ducb_values,
            'rollout_indices': rollout_indices,
            'node_changes': node_changes,
            'model_pred_values': model_pred_values
        }
        
        ducb_trends_data.append(trend_data)
    
    return ducb_trends_data

def main():
    """主测试函数"""
    print("开始测试增强版DUCB趋势图...")
    
    # 创建测试数据
    test_ducb_data = create_test_ducb_data()
    
    # 设置测试参数
    output_dir = "/media/ubuntu/19C1027D35EB273A/ML_projects/ML_training/week6"
    iteration = 999  # 测试迭代次数
    dynamic_exploration_factor = 2.35  # 测试探索因子
    new_global_best_value = 3.14159e-6  # 测试新的全局最优值
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 调用绘图函数
        saved_path = plot_nte_ducb_trends(
            ducb_trends_data=test_ducb_data,
            output_dir=output_dir,
            iteration=iteration,
            dynamic_exploration_factor=dynamic_exploration_factor,
            new_global_best_value=new_global_best_value,
            filename="test_enhanced_ducb_trends.png"
        )
        
        if saved_path:
            print(f"✅ 测试成功！增强版DUCB趋势图已保存至: {saved_path}")
            print(f"   图表包含:")
            print(f"   - 3条DUCB曲线（实线）")
            print(f"   - 3条模型预测值-v_ML曲线（虚线）")
            print(f"   - 节点变更标记")
            print(f"   - 动态探索因子: {dynamic_exploration_factor}")
            print(f"   - 新全局最优值: {new_global_best_value:.6e}")
        else:
            print("❌ 测试失败：图表保存失败")
            
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
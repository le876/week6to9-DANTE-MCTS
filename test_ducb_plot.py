import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加项目路径
sys.path.append('/media/ubuntu/19C1027D35EB273A/ML_projects/ML_training/week6')

from rosenbrock.plot_utils import plot_nte_ducb_trends

# 创建测试目录
test_dir = '/tmp/test_ducb_plot'
os.makedirs(test_dir, exist_ok=True)

# 创建模拟的DUCB趋势数据
ducb_trends_data = []

for i in range(3):  # 3个初始点
    # 模拟DUCB值变化（下降趋势）
    rollout_indices = list(range(100))
    ducb_values = [5.0 - i * 0.5 + np.random.random() * 0.5 - idx * 0.02 + np.sin(idx * 0.1) * 0.3 for idx in rollout_indices]
    
    # 模拟一些节点变化位置
    node_changes = [10 + i * 3, 25 + i * 5, 50 + i * 7, 75 + i * 2]
    
    trend_data = {
        'ducb_values': ducb_values,
        'rollout_indices': rollout_indices,
        'node_changes': node_changes
    }
    ducb_trends_data.append(trend_data)

# 测试绘制函数
try:
    result = plot_nte_ducb_trends(
        ducb_trends_data=ducb_trends_data,
        output_dir=test_dir,
        iteration=100,
        new_global_best_value=1.234567e-4
    )
    print(f'测试成功！图表已保存至: {result}')
except Exception as e:
    print(f'测试失败: {e}')
    import traceback
    traceback.print_exc() 
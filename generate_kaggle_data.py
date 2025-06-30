#!/usr/bin/env python3
"""
生成Kaggle DANTE训练所需的Rosenbrock数据集
运行此脚本生成数据，然后上传到Kaggle Datasets
"""

import numpy as np
import os

def rosenbrock_function(x):
    """计算Rosenbrock函数值"""
    return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0, axis=0)

def generate_rosenbrock_data():
    """生成20维Rosenbrock数据集"""
    
    print("🔄 开始生成20维Rosenbrock数据集...")
    
    # 参数设置
    dimension = 20
    domain_min, domain_max = -2.048, 2.048
    n_train = 800
    n_test = 200
    
    # 设置随机种子确保可重现性
    np.random.seed(42)
    
    # 生成训练数据
    print(f"📊 生成训练数据: {n_train}个样本...")
    X_train = np.random.uniform(domain_min, domain_max, (n_train, dimension))
    y_train = np.array([rosenbrock_function(x) for x in X_train]).reshape(-1, 1)
    
    # 生成测试数据
    print(f"📊 生成测试数据: {n_test}个样本...")
    X_test = np.random.uniform(domain_min, domain_max, (n_test, dimension))
    y_test = np.array([rosenbrock_function(x) for x in X_test]).reshape(-1, 1)
    
    # 创建输出目录
    output_dir = "rosenbrock_data_raw"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存数据
    print(f"💾 保存数据到 {output_dir}/...")
    np.save(os.path.join(output_dir, "Rosenbrock_x_train.npy"), X_train)
    np.save(os.path.join(output_dir, "Rosenbrock_y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "Rosenbrock_x_test.npy"), X_test)
    np.save(os.path.join(output_dir, "Rosenbrock_y_test.npy"), y_test)
    
    # 验证数据
    print("\n✅ 数据生成完成，验证信息:")
    print(f"   训练集 X: {X_train.shape}, 类型: {X_train.dtype}")
    print(f"   训练集 y: {y_train.shape}, 类型: {y_train.dtype}")
    print(f"   测试集 X: {X_test.shape}, 类型: {X_test.dtype}")
    print(f"   测试集 y: {y_test.shape}, 类型: {y_test.dtype}")
    
    # 数据统计
    print(f"\n📈 数据统计:")
    print(f"   训练集y范围: [{y_train.min():.2e}, {y_train.max():.2e}]")
    print(f"   测试集y范围: [{y_test.min():.2e}, {y_test.max():.2e}]")
    print(f"   全局最优值: {rosenbrock_function(np.ones(dimension)):.2e}")
    
    # 创建压缩包说明
    print(f"\n📦 下一步操作:")
    print(f"1. 将 '{output_dir}' 文件夹压缩为 'rosenbrock-data-20d-800.zip'")
    print(f"2. 上传到Kaggle Datasets，命名为 'rosenbrock-data-20d-800'")
    print(f"3. 在Kaggle Notebook中添加此数据集为输入")
    
    return output_dir

if __name__ == "__main__":
    output_directory = generate_rosenbrock_data()
    print(f"\n🎉 数据生成完成！输出目录: {output_directory}")
    print("✨ 现在可以上传到Kaggle了！") 
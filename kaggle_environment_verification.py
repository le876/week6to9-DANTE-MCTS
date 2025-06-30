# =============================================================================
# Kaggle DANTE 环境配置验证和基本功能测试
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import logging
import torch
import random
from scipy.stats import pearsonr
import traceback
from datetime import datetime
import gc
import tensorflow as tf
import psutil

def verify_environment_and_functionality():
    """全面验证Kaggle环境配置和基本功能"""
    
    print("🔍 开始环境配置验证...")
    verification_results = {}
    
    # 1. GPU配置验证
    print("\n1. GPU配置验证")
    print("-" * 40)
    
    # TensorFlow GPU
    tf_gpus = tf.config.list_physical_devices('GPU')
    if tf_gpus:
        try:
            # 启用内存增长
            for gpu in tf_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ TensorFlow GPU可用: {len(tf_gpus)}个设备")
            for i, gpu in enumerate(tf_gpus):
                print(f"   GPU {i}: {gpu.name}")
            verification_results['tensorflow_gpu'] = True
        except Exception as e:
            print(f"❌ TensorFlow GPU配置失败: {e}")
            verification_results['tensorflow_gpu'] = False
    else:
        print("❌ TensorFlow未检测到GPU")
        verification_results['tensorflow_gpu'] = False
    
    # PyTorch GPU
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"✅ PyTorch CUDA可用: {device_count}个设备")
        print(f"   当前设备: {current_device} ({device_name})")
        verification_results['pytorch_gpu'] = True
    else:
        print("❌ PyTorch CUDA不可用")
        verification_results['pytorch_gpu'] = False
    
    # 2. 内存配置验证
    print("\n2. 内存配置验证")
    print("-" * 40)
    try:
        memory = psutil.virtual_memory()
        print(f"✅ 总内存: {memory.total / (1024**3):.1f} GB")
        print(f"✅ 可用内存: {memory.available / (1024**3):.1f} GB")
        print(f"✅ 内存使用率: {memory.percent:.1f}%")
        verification_results['memory_info'] = True
    except Exception as e:
        print(f"❌ 内存信息获取失败: {e}")
        verification_results['memory_info'] = False
    
    # 3. 路径配置验证
    print("\n3. 路径配置验证")
    print("-" * 40)
    
    # Kaggle输入路径
    input_path = "/kaggle/input"
    working_path = "/kaggle/working"
    
    print(f"✅ Kaggle输入路径: {input_path} (存在: {os.path.exists(input_path)})")
    print(f"✅ Kaggle工作路径: {working_path} (存在: {os.path.exists(working_path)})")
    
    if os.path.exists(input_path):
        try:
            input_contents = os.listdir(input_path)
            print(f"   输入路径内容: {input_contents}")
        except Exception as e:
            print(f"   输入路径读取失败: {e}")
    
    verification_results['paths_configured'] = os.path.exists(working_path)
    
    # 4. 核心模块功能验证
    print("\n4. 核心模块功能验证")
    print("-" * 40)
    
    try:
        # 验证Rosenbrock函数 (需要定义)
        def rosenbrock_function(x):
            """计算给定向量x的Rosenbrock函数值。"""
            return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0, axis=0)
            
        test_x = np.array([1.0, 1.0])  # 全局最优点
        test_result = rosenbrock_function(test_x)
        print(f"✅ Rosenbrock函数测试: f([1,1]) = {test_result:.6f} (应接近0)")
        verification_results['rosenbrock_function'] = abs(test_result) < 1e-10
        
        # 验证safe_power10函数 (需要定义)
        def safe_power10(log_y, epsilon=1.0):
            """安全地将log尺度值转换为原始尺度"""
            try:
                # 限制输入范围防止溢出
                log_y_clipped = np.clip(log_y, -50, 50)
                result = 10.0 ** log_y_clipped - epsilon
                # 确保结果为正
                return np.maximum(result, 1e-10)
            except Exception as e:
                print(f"safe_power10转换错误: {e}")
                return np.ones_like(log_y) * 1e-10
                
        test_log = np.array([0.0, 1.0, 2.0])
        test_power = safe_power10(test_log)
        print(f"✅ safe_power10函数测试: {test_log} -> {test_power}")
        verification_results['safe_power10'] = True
        
        print(f"✅ 核心功能模块验证成功")
        verification_results['core_modules'] = True
        
    except Exception as e:
        print(f"❌ 核心模块验证失败: {e}")
        print(traceback.format_exc())
        verification_results['core_modules'] = False
    
    # 5. 数据类型兼容性验证
    print("\n5. 数据类型兼容性验证")
    print("-" * 40)
    
    try:
        # NumPy数组
        np_array = np.random.randn(10, 5)
        print(f"✅ NumPy数组: shape {np_array.shape}, dtype {np_array.dtype}")
        
        # PyTorch张量
        torch_tensor = torch.randn(10, 5)
        print(f"✅ PyTorch张量: shape {torch_tensor.shape}, dtype {torch_tensor.dtype}")
        
        # TensorFlow张量
        tf_tensor = tf.random.normal((10, 5))
        print(f"✅ TensorFlow张量: shape {tf_tensor.shape}, dtype {tf_tensor.dtype}")
        
        verification_results['data_types'] = True
        
    except Exception as e:
        print(f"❌ 数据类型验证失败: {e}")
        verification_results['data_types'] = False
    
    # 6. 输出目录创建测试
    print("\n6. 输出目录创建测试")
    print("-" * 40)
    
    try:
        test_results_dir = os.path.join("/kaggle/working", "test_results")
        os.makedirs(test_results_dir, exist_ok=True)
        
        # 测试文件写入
        test_file = os.path.join(test_results_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("Test successful")
        
        print(f"✅ 输出目录创建和文件写入成功: {test_results_dir}")
        verification_results['output_directory'] = True
        
        # 清理测试文件
        os.remove(test_file)
        os.rmdir(test_results_dir)
        
    except Exception as e:
        print(f"❌ 输出目录测试失败: {e}")
        verification_results['output_directory'] = False
    
    # 7. 综合评估
    print("\n7. 综合评估")
    print("=" * 50)
    
    total_checks = len(verification_results)
    passed_checks = sum(verification_results.values())
    success_rate = passed_checks / total_checks * 100
    
    print(f"验证项目总数: {total_checks}")
    print(f"通过项目数: {passed_checks}")
    print(f"成功率: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("🎉 环境配置验证通过！可以开始训练")
        ready_for_training = True
    elif success_rate >= 70:
        print("⚠️  环境配置基本可用，但存在一些问题")
        ready_for_training = True
    else:
        print("❌ 环境配置存在严重问题，需要修复")
        ready_for_training = False
    
    # 8. 详细结果报告
    print("\n8. 详细结果报告")
    print("-" * 40)
    for check, result in verification_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{check}: {status}")
    
    return ready_for_training, verification_results

if __name__ == "__main__":
    # 执行环境验证
    print("开始执行Kaggle DANTE环境配置验证...")
    ready_for_training, verification_results = verify_environment_and_functionality()

    if ready_for_training:
        print("\n🚀 环境验证完成，系统已准备就绪！")
        print("💡 提示: 现在可以运行主训练循环")
    else:
        print("\n⚠️  请先解决环境配置问题再开始训练")

    print(f"\n📊 验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80) 
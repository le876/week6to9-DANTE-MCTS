#!/usr/bin/env python3
"""
DANTE项目环境检查脚本
"""

import sys
import os

def run_environment_check():
    """运行环境检查"""
    print('🔍 开始环境检查...')
    print('✅ Python版本:', sys.version.split()[0])

    # 检查基础库
    libraries = [
        ('numpy', 'np'),
        ('pandas', 'pd'), 
        ('matplotlib.pyplot', 'plt'),
        ('tensorflow', 'tf'),
        ('torch', 'torch'),
        ('scipy', 'scipy'),
        ('sklearn', 'sklearn')
    ]
    
    failed_imports = []
    
    for lib_name, alias in libraries:
        try:
            if lib_name == 'matplotlib.pyplot':
                import matplotlib.pyplot as plt
                print(f'✅ {lib_name}可用')
            elif lib_name == 'tensorflow':
                import tensorflow as tf
                print(f'✅ {lib_name}版本: {tf.__version__}')
            elif lib_name == 'torch':
                import torch
                print(f'✅ {lib_name}版本: {torch.__version__}')
                if torch.cuda.is_available():
                    print(f'✅ CUDA可用, GPU数量: {torch.cuda.device_count()}')
                    print(f'   当前GPU: {torch.cuda.get_device_name(0)}')
                else:
                    print('⚠️ CUDA不可用，将使用CPU')
            elif lib_name == 'numpy':
                import numpy as np
                print(f'✅ {lib_name}版本: {np.__version__}')
            elif lib_name == 'pandas':
                import pandas as pd
                print(f'✅ {lib_name}版本: {pd.__version__}')
            elif lib_name == 'scipy':
                import scipy
                print(f'✅ {lib_name}版本: {scipy.__version__}')
            elif lib_name == 'sklearn':
                import sklearn
                print(f'✅ {lib_name}版本: {sklearn.__version__}')
        except Exception as e:
            print(f'❌ {lib_name}导入失败: {e}')
            failed_imports.append(lib_name)

    # 检查工作目录
    print(f'\n📁 当前工作目录: {os.getcwd()}')
    
    # 检查数据目录
    data_dirs = ['rosenbrock/data_raw', 'MCTS', 'rosenbrock/results']
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            print(f'✅ 目录存在: {data_dir}')
        else:
            print(f'⚠️ 目录不存在: {data_dir}')
    
    print('\n🎯 环境检查完成')
    
    if failed_imports:
        print(f'\n⚠️ 以下库导入失败: {", ".join(failed_imports)}')
        print('请确保已安装所有必需的依赖库')
        return False
    else:
        print('\n✅ 所有基础库检查通过！')
        return True

if __name__ == '__main__':
    run_environment_check() 
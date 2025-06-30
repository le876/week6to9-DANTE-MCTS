# =============================================================================
# Cell 8: 主训练循环 - DANTE优化与dynamic_exploration_factor调试
# =============================================================================

print("🚀 开始DANTE主训练循环...")

def main_training_loop():
    """DANTE主训练循环，专注于dynamic_exploration_factor调试"""
    
    # === 参数配置 ===
    ROSENBROCK_DIMENSION = 20
    domain_min, domain_max = -2.048, 2.048
    ROSENBROCK_DOMAIN = [(domain_min, domain_max)] * ROSENBROCK_DIMENSION
    
    # 训练参数
    INITIAL_DATA_SIZE = 800
    N_AL_ITERATIONS = 400      # 可调整为更小值如100进行快速测试
    CNN_TRAINING_EPOCHS = 200
    CNN_EARLY_STOPPING_PATIENCE = 30
    CNN_VALIDATION_SPLIT = 0.2
    CNN_BATCH_SIZE = 32
    CNN_DROPOUT_RATE = 0.2
    CNN_LEARNING_RATE = 1e-3
    
    # NTE参数
    NTE_C0 = 1
    NTE_ROLLOUT_ROUNDS = 100
    VALIDATION_BATCH_SIZE = 20
    
    # 可视化参数
    viz_interval = 10  # 每10次迭代可视化一次
    
    print(f"📋 DANTE配置:")
    print(f"   维度: {ROSENBROCK_DIMENSION}D")
    print(f"   初始样本: {INITIAL_DATA_SIZE}")
    print(f"   AL迭代: {N_AL_ITERATIONS}")
    print(f"   CNN epochs: {CNN_TRAINING_EPOCHS}")
    print(f"   可视化间隔: {viz_interval}")
    
    # === 数据加载 ===
    try:
        print("\n📁 加载初始数据...")
        x_train_path = os.path.join(DATA_PATH, "Rosenbrock_x_train.npy")
        y_train_path = os.path.join(DATA_PATH, "Rosenbrock_y_train.npy")
        x_test_path = os.path.join(DATA_PATH, "Rosenbrock_x_test.npy")
        y_test_path = os.path.join(DATA_PATH, "Rosenbrock_y_test.npy")
        
        # 加载训练数据
        initial_X = np.load(x_train_path)
        initial_y = np.load(y_train_path)
        if initial_y.ndim == 1:
            initial_y = initial_y.reshape(-1, 1)
        
        # 加载测试数据
        X_test = np.load(x_test_path) if os.path.exists(x_test_path) else None
        y_test = np.load(y_test_path) if os.path.exists(y_test_path) else None
        if y_test is not None and y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)
            
        print(f"✅ 训练数据: {initial_X.shape}, {initial_y.shape}")
        if X_test is not None:
            print(f"✅ 测试数据: {X_test.shape}, {y_test.shape}")
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # === 初始化组件 ===
    print("\n🔧 初始化核心组件...")
    
    # 创建结果目录
    results_dir = create_results_directory(RESULTS_PATH)
    
    # 数据管理器
    data_manager = DataManager(
        initial_X=initial_X,
        initial_y=initial_y,
        dimension=ROSENBROCK_DIMENSION
    )
    
    # CNN模型
    cnn_model = CNN1DSurrogate(input_dim=ROSENBROCK_DIMENSION)
    
    # 历史记录
    history_iterations = []
    history_best_y = []
    history_pearson_all = []
    history_pearson_best_samples = []
    history_nte_candidates_count = []
    search_boundaries_history = []
    max_ducb_history = []
    cnn_performance_history = []
    
    # 连续无改进追踪
    no_improvement_streak = 0
    previous_best_y = float('inf')
    previous_iteration_samples_x = None
    
    print(f"✅ 初始化完成，数据集大小: {data_manager.num_samples}")
    
    # === NTE替代模型包装器 ===
    class NTESurrogateWrapper:
        def __init__(self, surrogate_model, data_manager_instance):
            self.surrogate_model = surrogate_model
            self.dm = data_manager_instance
            
        def predict(self, X_batch_orig: np.ndarray) -> np.ndarray:
            if X_batch_orig.shape[0] == 0:
                return np.array([])
            if X_batch_orig.ndim == 1:
                X_batch_orig = X_batch_orig.reshape(1, -1)
                
            try:
                pred_log = self.surrogate_model.predict(X_batch_orig)
                return pred_log[0] if X_batch_orig.shape[0] == 1 else pred_log
            except Exception as e:
                logger.error(f"NTE预测错误: {e}")
                return np.ones(X_batch_orig.shape[0]) * np.log10(1e6 + 1.0)
    
    # === 主训练循环 ===
    print(f"\n🎯 开始{N_AL_ITERATIONS}次主动学习迭代...")
    print("=" * 80)
    
    for al_iteration in range(N_AL_ITERATIONS):
        current_iteration_number = al_iteration + 1
        history_iterations.append(current_iteration_number)
        
        # 可视化决策
        should_generate_plots = (current_iteration_number % viz_interval == 0) or (current_iteration_number == N_AL_ITERATIONS)
        
        print(f"\n{'='*20} 迭代 {current_iteration_number}/{N_AL_ITERATIONS} {'='*20}")
        
        # === CNN训练 ===
        logger.info(f"训练CNN模型 (迭代 {current_iteration_number})")
        training_start_time = time.time()
        
        training_history = cnn_model.train(
            data_manager=data_manager,
            epochs=CNN_TRAINING_EPOCHS,
            batch_size=CNN_BATCH_SIZE,
            validation_split=CNN_VALIDATION_SPLIT,
            patience=CNN_EARLY_STOPPING_PATIENCE,
            verbose=0
        )
        
        training_time = time.time() - training_start_time
        logger.info(f"CNN训练完成，耗时: {training_time:.1f}秒")
        
        # === CNN性能评估 ===
        try:
            # 训练集性能
            X_train = data_manager.get_all_x_orig()
            y_train_orig = data_manager.get_all_y_orig().flatten()
            train_predictions_log = cnn_model.predict(X_train)
            train_predictions_orig = safe_power10(train_predictions_log)
            train_mse = np.mean((train_predictions_orig - y_train_orig) ** 2)
            train_rmse = np.sqrt(train_mse)
            
            # 测试集性能
            if X_test is not None:
                test_predictions_log = cnn_model.predict(X_test)
                test_predictions_orig = safe_power10(test_predictions_log)
                test_ground_truth = y_test.flatten()
                test_mse = np.mean((test_predictions_orig - test_ground_truth) ** 2)
                test_rmse = np.sqrt(test_mse)
                test_pearson, _ = pearsonr(test_predictions_orig, test_ground_truth)
                test_pearson = 0.0 if np.isnan(test_pearson) else test_pearson
            else:
                test_rmse, test_pearson = 0.0, 0.0
            
            history_pearson_all.append(test_pearson)
            
            # 最佳样本性能
            best_pearson, min_y_best, max_y_best = calculate_pearson_on_best_samples(cnn_model, data_manager)
            history_pearson_best_samples.append(best_pearson)
            
            logger.info(f"CNN性能 - 训练RMSE: {train_rmse:.4e}, 测试Pearson: {test_pearson:.4f}, 最佳样本Pearson: {best_pearson:.4f}")
            
        except Exception as e:
            logger.error(f"CNN性能评估失败: {e}")
            history_pearson_all.append(0.0)
            history_pearson_best_samples.append(0.0)
        
        # === 关键部分：动态探索因子计算与NTE搜索 ===
        logger.info("开始NTE搜索...")
        
        # 计算动态探索因子
        dynamic_exploration_factor = 0.2 + 1 * min(20, 2 ** no_improvement_streak)
        logger.info(f"🎯 动态探索因子: {dynamic_exploration_factor:.2f} (连续无改进: {no_improvement_streak}次)")
        
        # NTE搜索
        try:
            nte_model_wrapper = NTESurrogateWrapper(cnn_model, data_manager)
            nte_searcher = NTESearcher(
                dimension=ROSENBROCK_DIMENSION,
                domain=ROSENBROCK_DOMAIN,
                c0=NTE_C0,
                rollout_rounds=NTE_ROLLOUT_ROUNDS,
                validation_batch_size=VALIDATION_BATCH_SIZE
            )
            
            current_best_x_orig, current_best_y_overall = data_manager.get_current_best()
            
            # 执行NTE搜索
            search_start_time = time.time()
            nte_search_result = nte_searcher.search(
                surrogate_model=nte_model_wrapper,
                current_best_state=current_best_x_orig,
                min_y_observed=current_best_y_overall,
                dynamic_exploration_factor=dynamic_exploration_factor,
                all_states=data_manager.X_data,
                all_values=data_manager.get_all_y_orig(),
                previous_iteration_samples=previous_iteration_samples_x,
                current_iteration=al_iteration,
                return_ducb_trends=True
            )
            
            nte_selected_samples_x, ducb_trends_data = nte_search_result
            search_time = time.time() - search_start_time
            
            if isinstance(nte_selected_samples_x, list):
                nte_selected_samples_x = np.array(nte_selected_samples_x)
            
            logger.info(f"NTE搜索完成，耗时: {search_time:.1f}秒，找到 {len(nte_selected_samples_x)} 个候选点")
            history_nte_candidates_count.append(len(nte_selected_samples_x))
            
            # === 候选样本评估 ===
            if len(nte_selected_samples_x) > 0:
                logger.info(f"评估 {len(nte_selected_samples_x)} 个新候选样本...")
                nte_selected_samples_y_true = np.array([
                    rosenbrock_evaluator(x) for x in nte_selected_samples_x
                ]).reshape(-1, 1)
                
                # 添加到数据集
                data_manager.add_samples(nte_selected_samples_x, nte_selected_samples_y_true, re_normalize=True)
                previous_iteration_samples_x = nte_selected_samples_x.copy()
                
                # 更新最佳值
                current_best_x_orig, current_best_y_overall = data_manager.get_current_best()
                history_best_y.append(current_best_y_overall)
                
                # 检查是否有改进
                iteration_improved = current_best_y_overall < previous_best_y
                if iteration_improved:
                    logger.info(f"🎉 发现更优解: {current_best_y_overall:.6e} (提升: {previous_best_y - current_best_y_overall:.6e})")
                    no_improvement_streak = 0
                    previous_best_y = current_best_y_overall
                else:
                    no_improvement_streak += 1
                    logger.info(f"本轮无改进，连续无改进次数: {no_improvement_streak}")
                
                # 可视化
                if should_generate_plots:
                    try:
                        # DUCB趋势图
                        plot_nte_ducb_trends(
                            ducb_trends_data=ducb_trends_data,
                            output_dir=results_dir,
                            iteration=current_iteration_number,
                            dynamic_exploration_factor=dynamic_exploration_factor,
                            new_global_best_value=current_best_y_overall if iteration_improved else None
                        )
                        
                        # 全局最小值趋势
                        plot_global_min_value_trend(
                            history_iterations[:len(history_best_y)], 
                            history_best_y, 
                            results_dir,
                            filename=f"global_min_iter_{current_iteration_number}.png"
                        )
                        
                        # 保存样本
                        csv_path = os.path.join(results_dir, f"iteration_{current_iteration_number}_samples.csv")
                        data_dict = {}
                        for i in range(nte_selected_samples_x.shape[1]):
                            data_dict[f'x{i+1}'] = nte_selected_samples_x[:, i]
                        data_dict['y_true'] = nte_selected_samples_y_true.flatten()
                        data_dict['dynamic_factor'] = [dynamic_exploration_factor] * len(nte_selected_samples_y_true)
                        pd.DataFrame(data_dict).to_csv(csv_path, index=False)
                        
                    except Exception as e:
                        logger.error(f"可视化失败: {e}")
            else:
                history_nte_candidates_count.append(0)
                history_best_y.append(history_best_y[-1] if history_best_y else float('inf'))
                no_improvement_streak += 1
                
        except Exception as e:
            logger.error(f"NTE搜索失败: {e}")
            logger.error(traceback.format_exc())
            history_nte_candidates_count.append(0)
            no_improvement_streak += 1
        
        # === 进度报告 ===
        current_best_x_orig, current_best_y_overall = data_manager.get_current_best()
        logger.info(f"迭代 {current_iteration_number} 完成:")
        logger.info(f"  📊 数据集大小: {data_manager.num_samples}")
        logger.info(f"  🎯 当前最优值: {current_best_y_overall:.6e}")
        logger.info(f"  🔍 动态探索因子: {dynamic_exploration_factor:.2f}")
        logger.info(f"  ⏱️  连续无改进: {no_improvement_streak}次")
        
        # 内存清理
        if current_iteration_number % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # === 训练完成处理 ===
    print("\n" + "="*80)
    print("🎉 DANTE训练完成！")
    print("="*80)
    
    final_best_x, final_best_y = data_manager.get_current_best()
    print(f"🏆 最终结果:")
    print(f"   最优函数值: {final_best_y:.6e}")
    print(f"   总样本数: {data_manager.num_samples}")
    print(f"   最终数据集大小: {INITIAL_DATA_SIZE + N_AL_ITERATIONS * VALIDATION_BATCH_SIZE}")
    
    # 保存最终结果
    try:
        # 数据管理器状态
        final_data_path = os.path.join(results_dir, "final_data_manager_state.npz")
        data_manager.save_data(final_data_path)
        
        # 性能历史
        final_history = {
            'iterations': history_iterations,
            'best_y_values': history_best_y,
            'pearson_all': history_pearson_all,
            'pearson_best_samples': history_pearson_best_samples,
            'nte_candidates_count': history_nte_candidates_count
        }
        history_df = pd.DataFrame(final_history)
        history_csv_path = os.path.join(results_dir, "training_history.csv")
        history_df.to_csv(history_csv_path, index=False)
        
        print(f"💾 结果已保存至: {results_dir}")
        
        # 最终图表
        plot_global_min_value_trend(
            history_iterations[:len(history_best_y)], 
            history_best_y, 
            results_dir,
            filename="global_min_value_trend_final.png"
        )
        
    except Exception as e:
        logger.error(f"保存最终结果失败: {e}")
    
    return results_dir, final_best_y, data_manager.num_samples

# 计算最佳样本Pearson相关系数的辅助函数
def calculate_pearson_on_best_samples(model, data_manager, n_best=60):
    """计算最佳样本的Pearson相关系数"""
    try:
        all_X = data_manager.get_all_x_orig()
        all_y_orig = data_manager.get_all_y_orig()
        
        if all_X.shape[0] < 2:
            return 0.0, float('nan'), float('nan')
        
        sorted_indices = np.argsort(all_y_orig.flatten())
        sorted_X = all_X[sorted_indices[:n_best]]
        sorted_y_orig = all_y_orig[sorted_indices[:n_best]]
        
        min_y_orig = np.min(sorted_y_orig)
        max_y_orig = np.max(sorted_y_orig)
        
        predictions_log = model.predict(sorted_X)
        predictions_orig = safe_power10(predictions_log)
        true_values_orig = sorted_y_orig.flatten()
        
        if len(predictions_orig) < 2 or np.var(predictions_orig) == 0 or np.var(true_values_orig) == 0:
            return 0.0, min_y_orig, max_y_orig
        
        correlation, _ = pearsonr(predictions_orig, true_values_orig)
        return (correlation if not np.isnan(correlation) else 0.0), min_y_orig, max_y_orig
        
    except Exception as e:
        logger.error(f"计算最佳样本Pearson失败: {e}")
        return 0.0, float('nan'), float('nan')

# === 执行选项 ===
print("\n🚀 准备开始训练...")
print("⚡ 如需快速测试，可以修改 N_AL_ITERATIONS = 50")
print("🔥 如需完整实验，保持 N_AL_ITERATIONS = 400")

# 开始训练
results_directory, best_value, total_samples = main_training_loop()

print(f"\n🎯 训练完成总结:")
print(f"   结果目录: {results_directory}")
print(f"   最优值: {best_value:.6e}")
print(f"   总样本数: {total_samples}")
print(f"   运行时间: {datetime.now()}") 
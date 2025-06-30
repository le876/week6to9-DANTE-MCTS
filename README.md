# ���亯���Ż���Ŀ��DANTE��LA-MCTS�㷨ʵ��

## ��Ŀ���

����Ŀ�ֱ�ʵ�������ֺ��亯���Ż��㷨��**DANTE**��Deep Active Learning with Neural-Surrogate-Guided Tree Exploration����**LA-MCTS**��Learning Search Space Partition for Black-box Optimization using Monte Carlo Tree Search����������20D Rosenbrock�Լ�20D Schwefel�������ݼ������Ż�������Ϊһ�����ֵ�����ÿ�ε���ֻ��Ŀ�꺯���ύ20������������ֵ���㡣�Ա������㷨�ڸ�ά�����Ż������ϵ����ܲ��졣

## ��Ŀ�ṹ

```
week6/
������ rosenbrock/                    # DANTE�㷨��Rosenbrock�����ϵ�ʵ��
��   ������ main.py                   # ���������
��   ������ data_manager.py           # ���ݹ�����
��   ������ cnn_1d_surrogate.py       # 1D CNN����ģ��
��   ������ nte_searcher.py           # NTE��������ʵ��
��   ������ plot_utils.py             # ���ӻ�����
��   ������ performance_optimizer.py   # �����Ż���
��   ������ data_raw/                 # ԭʼ���ݼ�
��   ������ results/                  # �Ż����
������ schwefel/                     # DANTE�㷨��Schwefel�����ϵ�ʵ��
��   ������ main.py                   # ���������
��   ������ search_strategy.py        # ����ѧϰ�Ż���
��   ������ surrogate_model.py        # ��˹���̴���ģ��
��   ������ cooperative_sa.py         # Эͬģ���˻��㷨
��   ������ data_raw/                 # ԭʼ���ݼ�
��   ������ results/                  # �Ż����
������ MCTS/                         # LA-MCTS�㷨ʵ��
��   ������ rosenbrock_LaMCTS/        # LA-MCTS��Rosenbrock�����ϵ�Ӧ��
��   ������ schwefel_LaMCTS/          # LA-MCTS��Schwefel�����ϵ�Ӧ��
��   ������ LaMCTSԭ��.md             # LA-MCTS�㷨ԭ��˵��
������ rosenbrock_data_raw/          # Rosenbrock�������ݼ�
������ README.md                     # ��Ŀ˵���ĵ�
```

## �㷨ʵ��

### 1. DANTE�㷨

DANTE��һ�ֻ���������������ģ�͵�����ѧϰ�Ż���ܣ�ͨ����������������������������Ч̽����ά�����ռ䡣

![DANTE����ͼ](image.png)

#### �������
- **���ݹ�����**��`DataManager`��������ѵ�����ݺ�������һ��
- **CNN����ģ��**��ʹ��1D���������ѧϰĿ�꺯���Ľ���ӳ��
- **NTE������**��ʵ������������������������
- **�����Ż���**�����GPUѵ�������ڴ�ͼ����Ż�

#### �����ص�
- **�����չ����**����ϵ���ͻ�䡢���������ͻ��͵����ƶ�
- **����ѡ�����**������������������ֲ�����
- **�ֲ����򴫲�**��ֻ����ѡ���ڵ�ķ��ʴ���
- **��̬̽������**�����������޸Ľ���������̽��ǿ��

### 2. LA-MCTS�㷨

LA-MCTSͨ�����ؿ�����������̬ѧϰ�����ռ仮�֣�����ϣ������������ʹ�ñ�Ҷ˹�Ż����оֲ�������
![LA-MCTS����ͼ](image-1.png)

#### �������
- **MCTS���ṹ**����̬������ά��������
- **���ģ��**��ʹ�ø�˹���̻ع���Ϊ�ֲ�����ģ��
- **�ռ仮��**��ͨ��K-Means�����SVMѧϰ���߽߱�
- **�ֲ��Ż���**������TuRBO���׼��Ҷ˹�Ż�

#### �����ص�
- **ѧϰʽ�ռ仮��**������ѧϰ��ӦĿ�꺯�����ԵĿռ仮��
- **UCBѡ�����**��ƽ��̽��������
- **�����Ծ��߽߱�**��ʹ��SVM�˺���ʵ�ָ��ӱ߽�
- **Ԫ�㷨���**���������Ż�����Ϊ�ֲ������

## ʵ������

### Ŀ�꺯��

#### Rosenbrock����
- **ά��**��20ά
- **����**��`f(x) = ��[100*(x[i+1] - x[i]?)? + (x[i] - 1)?]`
- **������**��`[-2.048, 2.048]^20`
- **ȫ������**��`x* = [1, 1, ..., 1]`, `f(x*) = 0`

#### Schwefel����
- **ά��**��20ά
- **����**��`f(x) = 418.9829*d - ��[x[i]*sin(��|x[i]|)]`
- **������**��`[-500, 500]^20`
- **ȫ������**��`x* = [420.9687, ..., 420.9687]`, `f(x*) = 0`

## ���л���

### ϵͳҪ��
- Python 3.8+
- CUDA 11.x��GPU���٣�
- 16GB+ RAM
- 4GB+ GPU�ڴ�

### ������
```bash
# ��������
numpy>=1.19.0
tensorflow==2.10.0
torch>=1.12.0
scikit-learn==1.2.2
scipy>=1.6.0
matplotlib>=3.3.0
pandas>=1.2.0

# GPU���٣���ѡ��
cupy-cuda11x>=11.0.0
```

## ʹ�÷���

### 1. DANTE�㷨����

#### Rosenbrock�����Ż�
```bash
cd rosenbrock/
python main.py
```

#### Schwefel�����Ż�
```bash
cd schwefel/
python main.py
```

### 2. LA-MCTS�㷨����

#### Rosenbrock�����Ż�
```bash
cd MCTS/rosenbrock_LaMCTS/src/
python main_optimizer.py
```

#### Schwefel�����Ż�
```bash
cd MCTS/schwefel_LaMCTS/src/
python main_optimizer_schwefel.py
```
## �������

### ���ӻ����

#### DANTE�㷨���ӻ�
- **ѵ����ʷ**��CNNģ��ѵ����ʧ����
- **Ԥ������**��Ԥ��ֵvs��ʵֵɢ��ͼ
- **��������**��NTE�����߽��DUCB����
- **��������**��ȫ������ֵ�仯����
- **����Է���**��Pearson���ϵ���仯
Ackley���ӻ���������[Ackley DANTE���](rosenbrock/results/ackley_20250604_181825)
Rosenbrock���ӻ���������[Rosenbrock DANTE���](rosenbrock/results/20250605_152624-3.39)
Schwefel���ӻ�������:[Schwefel DANTE���](schwefel/results/20250603_204123-309.0092)
#### LA-MCTS�㷨���ӻ�
- **���ṹ**��MCTS���������ӻ�
- **�ռ仮��**�������ռ�ָ���
- **�����ֲ�**��PCA��ά��������ֲ�
- **��������**������ֵ������仯
Rosenbrock���ӻ���������[Rosenbrock LA-MCTS���](MCTS/rosenbrock_LaMCTS/results/run_20250512-120846-421)
Schwefel���ӻ���������[Schwefel LA-MCTS���](MCTS/schwefel_LaMCTS/results/run_schwefel_20250512-111443)


## ��������

1. **DANTE�㷨**��Wei, J., et al. "Deep Active Learning with Neural-Surrogate-Guided Tree Exploration for Global Optimization." Research Square (2024). https://www.researchsquare.com/article/rs-5434645/v1

2. **LA-MCTS�㷨**��Wang, L., et al. "Learning Search Space Partition for Black-box Optimization using Monte Carlo Tree Search." arXiv preprint arXiv:2007.00708 (2020). http://arxiv.org/abs/2007.00708

## �����Դ

- **LA-MCTS�ٷ�����**��https://github.com/facebookresearch/LaMCTS
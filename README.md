# 波士顿房价预测 — 使用指南

## 项目概述

A等级波士顿房价预测，C语言实现。
- 数据集：506条样本，13个特征 + 1个目标（MEDV）
- 算法：多元线性回归 + 梯度下降
- 特征筛选：皮尔逊相关系数 Top 4
- 评估指标：RMSE

## 环境要求

| 组件 | 说明 |
|------|------|
| C编译器 | GCC 8.1+（MinGW-W64） |
| Python | 3.7+（仅可视化） |
| Python包 | matplotlib, numpy |

安装 Python 依赖：
```
pip install matplotlib numpy
```

## 文件说明

| 文件 | 用途 |
|------|------|
| `boston_housing.c` | 源代码 |
| `housing-price.txt` | 数据集（506 x 14） |
| `plot_results.py` | 可视化脚本 |
| `boston_housing.exe` | 编译好的可执行文件 |

运行时自动生成：
| 文件 | 内容 |
|------|------|
| `loss.csv` | 训练损失曲线数据（epoch, mse） |
| `corr.csv` | 13个特征的皮尔逊相关系数（feature, r） |
| `model_params.csv` | 模型参数：权重、偏置、各特征及MEDV的最小/最大值 |
| `pred.csv` | 测试集预测结果（true_k, pred_k） |
| `user_pred.csv` | 用户交互预测记录（LSTAT, RM, PTRATIO, INDUS, pred_k） |
| `pred_vs_true.png` | 2x2 子图：四个Top4特征 vs MEDV 的关系图 |

## 编译

```
gcc -O2 -Wall -o boston_housing.exe boston_housing.c -lm
```

编译选项：
- `-O2`：优化等级
- `-lm`：链接数学库

## 运行

### 1. 训练 + 交互预测（一键完成）

```
.\boston_housing.exe
```

程序依次执行：
1. 加载 `housing-price.txt`
2. Min-Max 归一化
3. 计算13个特征与MEDV的皮尔逊相关系数
4. 自动筛选 Top 4：**LSTAT, RM, PTRATIO, INDUS**
5. 梯度下降训练（学习率=0.01，迭代=10000轮）
6. 输出模型权重和RMSE
7. 保存 `loss.csv`、`corr.csv`、`model_params.csv`、`pred.csv`
8. 进入交互预测模式
9. 退出预测后自动调用 Python 生成 `pred_vs_true.png`

### 2. 交互预测

训练完成后，程序等待输入特征值：

```
Enter: LSTAT, RM, PTRATIO, INDUS
> 5.0 6.5 17.0 7.0
Predicted MEDV = $28.62k
```

输入4个数字（空格分隔），回车即可获取预测房价。输入 `q` 退出。

每次成功预测会自动记录到 `user_pred.csv`。

### 3. 自动出图

输入q退出预测模式后，程序自动调用 Python 生成 `pred_vs_true.png`——一张 2x2 子图：

- **灰色散点**：原始数据集中各特征与MEDV的分布
- **蓝色曲线**：二次多项式拟合线，展示特征与房价的非线性趋势
- **红色菱形**：用户本次输入的所有预测点（标注在对应特征子图上）

### 4. 非交互运行（仅生成数据，不预测）

```
echo q | .\boston_housing.exe
```

## 手动绘图

```
python plot_results.py
```

生成 `pred_vs_true.png`。若存在 `user_pred.csv`，图上会用红色菱形标出用户预测点。

## 结果摘要

### 相关系数排名（按 |r| 降序）

```
LSTAT    -0.7377   低收入人群比例（最强负相关）
RM       +0.6954   住宅平均房间数（最强正相关）
PTRATIO  -0.5078   师生比例
INDUS    -0.4837   非零售商业用地比例
TAX      -0.4685   每万美元财产税率
NOX      -0.4273   一氧化氮浓度
CRIM     -0.3883   犯罪率
RAD      -0.3816   高速公路便利指数
AGE      -0.3770   老旧住房比例
ZN       +0.3604   住宅用地比例
B        +0.3335   黑人比例指标
DIS      +0.2499   到就业中心距离
CHAS     +0.1753   是否临近查尔斯河
```

### 模型权重

```
偏置 b    = +0.364
w_LSTAT   = -0.430    （低收入比例越高 → 房价越低）
w_RM      = +0.553    （房间数越多 → 房价越高）
w_PTRATIO = -0.203    （师生比越高 → 房价越低）
w_INDUS   = -0.020    （非零售用地越多 → 房价略低）
```

### 评估结果

| 指标 | 训练集 | 测试集 |
|------|:------:|:------:|
| RMSE（归一化） | 0.117 | 0.112 |
| RMSE（$k） | 10.26 | 10.03 |

## 常见问题

**提示 "Cannot open housing-price.txt"**
将 `housing-price.txt` 放在执行文件同目录下。

**提示 "No module named 'matplotlib'"**
```
pip install matplotlib numpy
```

**自动出图失败**
确认 Python 已安装且在 PATH 中，matplotlib、numpy 可用：
```
python -c "import matplotlib; import numpy"
```

**CSV 文件乱码**
所有输出均为纯 ASCII 编码，无编码问题。

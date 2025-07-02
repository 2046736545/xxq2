# 美国南瓜数据集分析项目

## 项目概述

本项目旨在对美国南瓜数据集进行全面分析，包括数据预处理、数据探索性分析、数据可视化和数据分析等步骤，最后对结果进行总结，以帮助我们更好地理解南瓜市场的价格动态。

## 目录结构

```
us-pumpkins-analysis/
├── 2.py    # 主要分析代码
├── US-pumpkins.csv       # 数据集
├── results/                # 结果存储目录
│   ├── analysis_visualization.png # 分析可视化图
│   └── analysis_report.md # 分析报告
└── README.md               # 项目说明文档
```

## 项目背景与目的

南瓜在美国是一种重要的农产品，其价格受到品种、产地和尺寸等多种因素的影响。本项目通过数据分析的方法，对美国南瓜数据集进行深入挖掘，旨在：

1. 探索不同品种、产地和尺寸的南瓜价格分布情况
2. 分析各类别南瓜价格的差异
3. 为南瓜市场的参与者提供决策参考

## 数据集说明

### 数据集来源
本数据集包含1757条美国南瓜的信息，包含丰富的南瓜元数据和价格信息。

### 数据特征

数据集包含26个特征，具体说明如下：

| 特征名称 | 类型 | 非空值 | 缺失值 | 说明 |
|---------|------|------|------|------|
| City Name | object | 1757 | 0 | 待补充 |
| Type | object | 45 | 1712 | 待补充 |
| Package | object | 1757 | 0 | 待补充 |
| Variety | object | 1752 | 5 | 待补充 |
| Sub Variety | object | 296 | 1461 | 待补充 |
| Grade | float64 | 0 | 1757 | 待补充 |
| Date | object | 1757 | 0 | 待补充 |
| Low Price | float64 | 1757 | 0 | 待补充 |
| High Price | float64 | 1757 | 0 | 待补充 |
| Mostly Low | float64 | 1654 | 103 | 待补充 |
| Mostly High | float64 | 1654 | 103 | 待补充 |
| Origin | object | 1754 | 3 | 待补充 |
| Origin District | object | 131 | 1626 | 待补充 |
| Item Size | object | 1478 | 279 | 待补充 |
| Color | object | 1141 | 616 | 待补充 |
| Environment | float64 | 0 | 1757 | 待补充 |
| Unit of Sale | object | 162 | 1595 | 待补充 |
| Quality | float64 | 0 | 1757 | 待补充 |
| Condition | float64 | 0 | 1757 | 待补充 |
| Appearance | float64 | 0 | 1757 | 待补充 |
| Storage | float64 | 0 | 1757 | 待补充 |
| Crop | float64 | 0 | 1757 | 待补充 |
| Repack | object | 1757 | 0 | 待补充 |
| Trans Mode | float64 | 0 | 1757 | 待补充 |
| Unnamed: 24 | float64 | 0 | 1757 | 待补充 |
| Unnamed: 25 | object | 103 | 1654 | 待补充 |

### 数据预览

数据集前5条记录：

```
City Name|Type|Package|Variety|Sub Variety|Grade|Date|Low Price|High Price|Mostly Low|Mostly High|Origin|Origin District|Item Size|Color|Environment|Unit of Sale|Quality|Condition|Appearance|Storage|Crop|Repack|Trans Mode|Unnamed: 24|Unnamed: 25|
|BALTIMORE|nan|24 inch bins|nan|nan|nan|4/29/17|270.0|280.0|270.0|280.0|MARYLAND|nan|lge|nan|nan|nan|nan|nan|nan|nan|nan|E|nan|nan|nan|
|BALTIMORE|nan|24 inch bins|nan|nan|nan|5/6/17|270.0|280.0|270.0|280.0|MARYLAND|nan|lge|nan|nan|nan|nan|nan|nan|nan|nan|E|nan|nan|nan|
|BALTIMORE|nan|24 inch bins|HOWDEN TYPE|nan|nan|9/24/16|160.0|160.0|160.0|160.0|DELAWARE|nan|med|ORANGE|nan|nan|nan|nan|nan|nan|nan|N|nan|nan|nan|
|BALTIMORE|nan|24 inch bins|HOWDEN TYPE|nan|nan|9/24/16|160.0|160.0|160.0|160.0|VIRGINIA|nan|med|ORANGE|nan|nan|nan|nan|nan|nan|nan|N|nan|nan|nan|
|BALTIMORE|nan|24 inch bins|HOWDEN TYPE|nan|nan|11/5/16|90.0|100.0|90.0|100.0|MARYLAND|nan|lge|ORANGE|nan|nan|nan|nan|nan|nan|nan|N|nan|nan|nan|
|```

## 技术栈

本项目使用的主要技术和库如下：

- **数据处理**：Pandas
- **数据可视化**：Matplotlib, Seaborn
- **数值计算**：NumPy

## 方法与步骤

### 1. 数据预处理
- 去除无关的列
- 转换数据类型，将Low Price和High Price列转换为数值类型
- 去除缺失值

### 2. 数据探索性分析
- 检查异常值
- 检查数据分布的偏度和峰度

### 3. 数据可视化
- 使用Matplotlib和Seaborn绘制价格分布图、箱线图、条形图和小提琴图

### 4. 数据分析
- 计算不同品种、产地和尺寸的平均价格

## 结果分析

### 1. 不同品种的平均价格

                           Low Price  High Price
Variety                                         
BIG MACK TYPE             123.016892  136.670270
BLUE TYPE                 222.105263  231.315789
CINDERELLA                169.486420  178.153086
FAIRYTALE                 191.791212  197.753333
HOWDEN TYPE               144.193727  157.684041
HOWDEN WHITE TYPE         151.673469  152.489796
KNUCKLE HEAD              189.750000  199.500000
MINIATURE                  26.449194   27.605645
MIXED HEIRLOOM VARIETIES  159.824561  163.070175
PIE TYPE                  125.629274  133.702991

### 2. 不同产地的平均价格

                 Low Price  High Price
Origin                                
ALABAMA         140.000000  145.000000
CALIFORNIA       84.234358   89.852936
CANADA          137.406977  161.651163
COSTA RICA       28.333333   29.000000
DELAWARE         75.055556   75.055556
FLORIDA          29.000000   29.000000
ILLINOIS        128.479167  130.892628
INDIANA         109.000000  117.500000
MARYLAND        114.590361  124.048193
MASSACHUSETTS   155.000000  178.563452
MEXICO          206.066667  211.400000
MICHIGAN        107.738397  114.652954
MISSOURI         72.272727   72.727273
NEW JERSEY      126.757576  139.787879
NEW MEXICO      100.000000  100.000000
NEW YORK        141.076923  146.076923
NORTH CAROLINA  168.119403  169.701493
OHIO             32.941441   33.923423
PENNSYLVANIA    142.096000  150.082000
TENNESSEE        78.250000   79.000000
TEXAS           159.130435  164.782609
VERMONT         190.833333  197.500000
VIRGINIA        147.138462  150.492308
WASHINGTON      139.200000  139.900000

### 3. 不同尺寸的平均价格

            Low Price  High Price
Item Size                        
exjbo      145.294118  161.176471
jbo        156.953125  166.353516
lge        150.247458  159.203390
med        121.559347  126.402077
med-lge    164.145865  179.418421
sml         84.211433   91.770661
xlge       188.902439  206.479268

通过以上分析，我们可以看到不同品种、产地和尺寸的南瓜价格分布情况。例如，某些品种的南瓜价格波动较大，而某些产地的南瓜价格相对稳定。这些信息可以帮助我们更好地理解南瓜市场的价格动态。

## 运行指南

### 环境要求

- Python 3.x
- 所需依赖包：
  - pandas
  - matplotlib
  - seaborn
  - numpy

### 安装依赖

```bash
pip install pandas matplotlib seaborn numpy
```

### 运行代码

1. 将数据集`US-pumpkins.csv`与代码`2.py`放在同一目录下
2. 打开终端，进入代码所在目录
3. 运行以下命令：

```bash
python 2.py
```

4. 运行结果将输出分析结果和可视化图表

## 项目改进方向

1. **特征扩展**：添加更多与南瓜相关的特征，如种植方式、季节等
2. **模型构建**：尝试使用机器学习模型预测南瓜价格
3. **时间序列分析**：结合时间因素分析南瓜价格的变化趋势

## 生成信息

本README文件由Python脚本自动生成于 2025年06月30日 18:03:32
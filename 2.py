# 导入必要的库
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 加载数据集
data = pd.read_csv('US-pumpkins.csv')

# 数据预处理
# 查看数据集的前几行
print("数据集的前几行：")
print(data.head())

# 查看数据集的基本信息
print("\n数据集的基本信息：")
print(data.info())

# 查看数据集的统计信息
print("\n数据集的统计信息：")
print(data.describe())

# 数据清洗
# 去除无关的列
columns_to_drop = ['Type', 'Package', 'Sub Variety', 'Grade', 'Mostly Low', 'Mostly High', 'Environment', 'Unit of Sale', 'Quality', 'Condition', 'Appearance', 'Storage', 'Crop', 'Repack', 'Trans Mode']
data_cleaned = data.drop(columns=columns_to_drop)

# 转换数据类型
# 尝试转换 Low Price 和 High Price 列为数值类型，非数值的转换为 NaN
data_cleaned['Low Price'] = pd.to_numeric(data_cleaned['Low Price'], errors='coerce')
data_cleaned['High Price'] = pd.to_numeric(data_cleaned['High Price'], errors='coerce')

# 去除缺失值
data_cleaned = data_cleaned.dropna(subset=['Low Price', 'High Price'])

# 查看清洗后的数据
print("\n清洗后的数据：")
print(data_cleaned.head())

# 数据探索性分析
# 检查异常值
print("\n描述性统计信息：")
print(data_cleaned.describe(include='all'))

# 检查数据分布的偏度和峰度
print("\n偏度和峰度：")
print(data_cleaned['Low Price'].skew())
print(data_cleaned['Low Price'].kurtosis())

# 数据可视化
# 使用 Matplotlib 绘制价格分布图
plt.figure(figsize=(12, 6))  # 增大图例尺寸
sns.histplot(data_cleaned['Low Price'], bins=20, alpha=0.5, label='Low Price', kde=True)
sns.histplot(data_cleaned['High Price'], bins=20, alpha=0.5, label='High Price', kde=True)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)  # 增加网格线
plt.show()

# 使用 Seaborn 绘制箱线图
plt.figure(figsize=(12, 6))  # 增大图例尺寸
sns.boxplot(x='Variety', y='Low Price', data=data_cleaned)
plt.title('Low Price by Variety')
plt.xticks(rotation=45)
plt.show()

# 使用 Seaborn 绘制条形图
plt.figure(figsize=(12, 6))  # 增大图例尺寸
sns.barplot(x='Variety', y='High Price', data=data_cleaned)
plt.title('High Price by Variety')
plt.xticks(rotation=45)
plt.show()

# 使用 Seaborn 绘制小提琴图
plt.figure(figsize=(12, 6))
sns.violinplot(x='Variety', y='Low Price', data=data_cleaned, inner='quartile')
plt.title('Violin Plot of Low Price by Variety')
plt.xticks(rotation=45)
plt.show()

# 数据分析
# 计算不同品种的平均价格
average_prices = data_cleaned.groupby('Variety')[['Low Price', 'High Price']].mean()
print("\n不同品种的平均价格：")
print(average_prices)

# 计算不同产地的平均价格
average_prices_by_origin = data_cleaned.groupby('Origin')[['Low Price', 'High Price']].mean()
print("\n不同产地的平均价格：")
print(average_prices_by_origin)

# 计算不同尺寸的平均价格
average_prices_by_size = data_cleaned.groupby('Item Size')[['Low Price', 'High Price']].mean()
print("\n不同尺寸的平均价格：")
print(average_prices_by_size)

# 总结
print("\n总结：")
print("通过可视化和分析，我们可以看到不同品种、产地和尺寸的南瓜价格分布情况。")
print("例如，某些品种的南瓜价格波动较大，而某些产地的南瓜价格相对稳定。")
print("这些信息可以帮助我们更好地理解南瓜市场的价格动态。")
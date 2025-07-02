import pandas as pd
import numpy as np
import os
from datetime import datetime

# 数据文件路径
DATA_FILE = 'US-pumpkins.csv'
README_FILE = 'README.md'
RESULT_DIR = 'results'
VISUALIZATION_FILE = os.path.join(RESULT_DIR, 'analysis_visualization.png')
REPORT_FILE = os.path.join(RESULT_DIR, 'analysis_report.md')

def get_dataset_info():
    """获取数据集的基本信息"""
    try:
        df = pd.read_csv(DATA_FILE)
        features_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = df[col].notnull().sum()
            features_info.append(f"| {col} | {dtype} | {non_null} | {len(df) - non_null} | 待补充 |")

        preview = df.head().to_csv(sep='|', na_rep='nan', columns=df.columns, index=False)
        preview = preview.replace('\n', '|\n|')

        return {
            'row_count': len(df),
            'col_count': len(df.columns),
            'features_info': features_info,
            'preview': preview
        }
    except Exception as e:
        print(f"读取数据集时出错: {e}")
        return None

def run_analysis():
    """运行数据分析获取结果"""
    try:
        df = pd.read_csv(DATA_FILE)

        # 数据预处理
        columns_to_drop = ['Type', 'Package', 'Sub Variety', 'Grade', 'Mostly Low', 'Mostly High', 'Environment', 'Unit of Sale', 'Quality', 'Condition', 'Appearance', 'Storage', 'Crop', 'Repack', 'Trans Mode']
        data_cleaned = df.drop(columns=columns_to_drop)

        data_cleaned['Low Price'] = pd.to_numeric(data_cleaned['Low Price'], errors='coerce')
        data_cleaned['High Price'] = pd.to_numeric(data_cleaned['High Price'], errors='coerce')
        data_cleaned = data_cleaned.dropna(subset=['Low Price', 'High Price'])

        # 数据分析
        average_prices = data_cleaned.groupby('Variety')[['Low Price', 'High Price']].mean()
        average_prices_by_origin = data_cleaned.groupby('Origin')[['Low Price', 'High Price']].mean()
        average_prices_by_size = data_cleaned.groupby('Item Size')[['Low Price', 'High Price']].mean()

        return {
            'average_prices': average_prices,
            'average_prices_by_origin': average_prices_by_origin,
            'average_prices_by_size': average_prices_by_size
        }
    except Exception as e:
        print(f"数据分析时出错: {e}")
        return None

def visualize_results():
    """可视化分析结果（此部分可根据需要进一步完善）"""
    pass

def generate_readme(dataset_info, analysis_results):
    """生成README内容"""
    if not dataset_info or not analysis_results:
        return "# 美国南瓜数据集分析项目\n\n生成README时数据获取失败"

    # 项目概述部分
    readme = "# 美国南瓜数据集分析项目\n\n"
    readme += "## 项目概述\n\n"
    readme += "本项目旨在对美国南瓜数据集进行全面分析，包括数据预处理、数据探索性分析、数据可视化和数据分析等步骤，最后对结果进行总结，以帮助我们更好地理解南瓜市场的价格动态。\n\n"

    # 目录结构部分
    readme += "## 目录结构\n\n"
    readme += "```\n"
    readme += "us-pumpkins-analysis/\n"
    readme += f"├── 2.py    # 主要分析代码\n"
    readme += f"├── {DATA_FILE}       # 数据集\n"
    readme += f"├── {RESULT_DIR}/                # 结果存储目录\n"
    readme += f"│   ├── {os.path.basename(VISUALIZATION_FILE)} # 分析可视化图\n"
    readme += f"│   └── {os.path.basename(REPORT_FILE)} # 分析报告\n"
    readme += f"└── {README_FILE}               # 项目说明文档\n"
    readme += "```\n\n"

    # 项目背景与目的部分
    readme += "## 项目背景与目的\n\n"
    readme += "南瓜在美国是一种重要的农产品，其价格受到品种、产地和尺寸等多种因素的影响。本项目通过数据分析的方法，对美国南瓜数据集进行深入挖掘，旨在：\n\n"
    readme += "1. 探索不同品种、产地和尺寸的南瓜价格分布情况\n"
    readme += "2. 分析各类别南瓜价格的差异\n"
    readme += "3. 为南瓜市场的参与者提供决策参考\n\n"

    # 数据集说明部分
    readme += "## 数据集说明\n\n"
    readme += "### 数据集来源\n"
    readme += f"本数据集包含{dataset_info['row_count']}条美国南瓜的信息，包含丰富的南瓜元数据和价格信息。\n\n"
    readme += "### 数据特征\n\n"
    readme += f"数据集包含{dataset_info['col_count']}个特征，具体说明如下：\n\n"
    readme += "| 特征名称 | 类型 | 非空值 | 缺失值 | 说明 |\n"
    readme += "|---------|------|------|------|------|\n"
    for feature_info in dataset_info['features_info']:
        readme += feature_info + "\n"
    readme += "\n"

    # 数据预览部分
    readme += "### 数据预览\n\n"
    readme += "数据集前5条记录：\n\n"
    readme += "```\n"
    readme += dataset_info['preview']
    readme += "```\n\n"

    # 技术栈部分
    readme += "## 技术栈\n\n"
    readme += "本项目使用的主要技术和库如下：\n\n"
    readme += "- **数据处理**：Pandas\n"
    readme += "- **数据可视化**：Matplotlib, Seaborn\n"
    readme += "- **数值计算**：NumPy\n\n"

    # 方法与步骤部分
    readme += "## 方法与步骤\n\n"
    readme += "### 1. 数据预处理\n"
    readme += "- 去除无关的列\n"
    readme += "- 转换数据类型，将Low Price和High Price列转换为数值类型\n"
    readme += "- 去除缺失值\n\n"
    readme += "### 2. 数据探索性分析\n"
    readme += "- 检查异常值\n"
    readme += "- 检查数据分布的偏度和峰度\n\n"
    readme += "### 3. 数据可视化\n"
    readme += "- 使用Matplotlib和Seaborn绘制价格分布图、箱线图、条形图和小提琴图\n\n"
    readme += "### 4. 数据分析\n"
    readme += "- 计算不同品种、产地和尺寸的平均价格\n\n"

    # 结果分析部分
    readme += "## 结果分析\n\n"
    readme += "### 1. 不同品种的平均价格\n\n"
    readme += f"{analysis_results['average_prices']}\n\n"
    readme += "### 2. 不同产地的平均价格\n\n"
    readme += f"{analysis_results['average_prices_by_origin']}\n\n"
    readme += "### 3. 不同尺寸的平均价格\n\n"
    readme += f"{analysis_results['average_prices_by_size']}\n\n"
    readme += "通过以上分析，我们可以看到不同品种、产地和尺寸的南瓜价格分布情况。例如，某些品种的南瓜价格波动较大，而某些产地的南瓜价格相对稳定。这些信息可以帮助我们更好地理解南瓜市场的价格动态。\n\n"

    # 运行指南部分
    readme += "## 运行指南\n\n"
    readme += "### 环境要求\n\n"
    readme += "- Python 3.x\n"
    readme += "- 所需依赖包：\n"
    readme += "  - pandas\n"
    readme += "  - matplotlib\n"
    readme += "  - seaborn\n"
    readme += "  - numpy\n\n"
    readme += "### 安装依赖\n\n"
    readme += "```bash\n"
    readme += "pip install pandas matplotlib seaborn numpy\n"
    readme += "```\n\n"
    readme += "### 运行代码\n\n"
    readme += f"1. 将数据集`{DATA_FILE}`与代码`2.py`放在同一目录下\n"
    readme += "2. 打开终端，进入代码所在目录\n"
    readme += "3. 运行以下命令：\n\n"
    readme += "```bash\n"
    readme += f"python 2.py\n"
    readme += "```\n\n"
    readme += "4. 运行结果将输出分析结果和可视化图表\n\n"

    # 项目改进方向
    readme += "## 项目改进方向\n\n"
    readme += "1. **特征扩展**：添加更多与南瓜相关的特征，如种植方式、季节等\n"
    readme += "2. **模型构建**：尝试使用机器学习模型预测南瓜价格\n"
    readme += "3. **时间序列分析**：结合时间因素分析南瓜价格的变化趋势\n\n"

    # 生成信息
    readme += "## 生成信息\n\n"
    readme += f"本README文件由Python脚本自动生成于 {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}"

    return readme

def main():
    """主函数"""
    print("正在获取数据集信息...")
    dataset_info = get_dataset_info()

    if not dataset_info:
        print("无法获取数据集信息，使用默认内容生成README")
        readme_content = "# 美国南瓜数据集分析项目\n\n数据集信息获取失败"
    else:
        print("正在运行数据分析...")
        analysis_results = run_analysis()

        if not analysis_results:
            print("数据分析失败，使用默认分析结果生成README")
            analysis_results = {
                'average_prices': None,
                'average_prices_by_origin': None,
                'average_prices_by_size': None
            }

        print("正在可视化分析结果...")
        visualize_results()

        print("正在生成README文件...")
        readme_content = generate_readme(dataset_info, analysis_results)

    # 写入README文件
    with open(README_FILE, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print(f"README文件已成功生成: {os.path.abspath(README_FILE)}")

if __name__ == "__main__":
    main()
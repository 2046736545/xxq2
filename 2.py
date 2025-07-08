# 导入必要的库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import seaborn as sns


# 数据预处理函数
def preprocess_data(df):
    # 日期解析
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y', errors='coerce')
    df = df.dropna(subset=['Date'])

    # 衍生时间特征
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Season'] = df['Month'].apply(
        lambda x: 'Spring' if 3 <= x <= 5 else 'Summer' if 6 <= x <= 8 else 'Autumn' if 9 <= x <= 11 else 'Winter'
    )

    # 计算目标变量（平均价格）
    df['Average Price'] = (df['Low Price'] + df['High Price']) / 2

    # 缺失值填充
    for col in ['Type', 'Item Size', 'Color']:
        df[col] = df[col].fillna(df[col].mode()[0])

    # 高基数类别处理
    for col in ['City Name', 'Origin']:
        top_10_cats = df[col].value_counts().head(10).index
        df[col] = df[col].apply(lambda x: x if x in top_10_cats else 'Other')

    # 删除冗余列
    drop_cols = [
        'Grade', 'Environment', 'Unit of Sale', 'Quality',
        'Condition', 'Appearance', 'Storage', 'Crop',
        'Trans Mode', 'Unnamed: 24', 'Unnamed: 25',
        'Low Price', 'High Price', 'Mostly Low', 'Mostly High',
        'Sub Variety', 'Origin District', 'Repack'
    ]
    df = df.drop(drop_cols, axis=1, errors='ignore')

    return df


# 特征工程函数
def engineer_features(df):
    numerical_features = ['Month', 'Year']
    categorical_features = [
        'City Name', 'Type', 'Package', 'Variety',
        'Origin', 'Item Size', 'Color', 'Season'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    return preprocessor, numerical_features, categorical_features


# 模型训练和评估函数
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2


# 主函数
def main():
    # 加载数据
    data = pd.read_csv('US-pumpkins.csv')

    # 数据预处理
    data_processed = preprocess_data(data)

    # 特征工程
    preprocessor, _, _ = engineer_features(data_processed)

    # 数据划分
    X = data_processed.drop(columns=['Average Price'], errors='ignore')
    y = data_processed['Average Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 预处理数据
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 随机森林模型
    rf_model = RandomForestRegressor(random_state=42)
    rf_mse, rf_r2 = train_and_evaluate_model(rf_model, X_train_processed, X_test_processed, y_train, y_test)
    print(f"随机森林回归的均方误差（MSE）: {rf_mse}")
    print(f"随机森林回归的决定系数（R²）: {rf_r2}")

    # XGBoost模型
    xgb_model = XGBRegressor(random_state=42)
    xgb_mse, xgb_r2 = train_and_evaluate_model(xgb_model, X_train_processed, X_test_processed, y_train, y_test)
    print(f"XGBoost回归的均方误差（MSE）: {xgb_mse}")
    print(f"XGBoost回归的决定系数（R²）: {xgb_r2}")

    # 支持向量机模型
    svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr_mse, svr_r2 = train_and_evaluate_model(svr_model, X_train_processed, X_test_processed, y_train, y_test)
    print(f"支持向量机回归的均方误差（MSE）: {svr_mse}")
    print(f"支持向量机回归的决定系数（R²）: {svr_r2}")

    # 数据可视化
    plt.figure(figsize=(12, 6))
    sns.histplot(data_processed['Average Price'], bins=20, alpha=0.5, label='Average Price', kde=True)
    plt.title('Average Price Distribution')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Variety', y='Average Price', data=data_processed)
    plt.title('Average Price by Variety')
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Variety', y='Average Price', data=data_processed)
    plt.title('Average Price by Variety')
    plt.xticks(rotation=45)
    plt.show()

    # 数据探索性分析
    print("\n数据集的前几行：")
    print(data.head())

    print("\n数据集的基本信息：")
    print(data.info())

    print("\n数据集的统计信息：")
    print(data.describe())

    print("\n清洗后的数据：")
    print(data_processed.head())

    print("\n描述性统计信息：")
    print(data_processed.describe(include='all'))

    print("\n偏度和峰度：")
    print(data_processed['Average Price'].skew())
    print(data_processed['Average Price'].kurtosis())

    # 计算不同品种的平均价格
    average_prices = data_processed.groupby('Variety')[['Average Price']].mean()
    print("\n不同品种的平均价格：")
    print(average_prices)

    # 计算不同产地的平均价格
    average_prices_by_origin = data_processed.groupby('Origin')[['Average Price']].mean()
    print("\n不同产地的平均价格：")
    print(average_prices_by_origin)

    # 计算不同尺寸的平均价格
    average_prices_by_size = data_processed.groupby('Item Size')[['Average Price']].mean()
    print("\n不同尺寸的平均价格：")
    print(average_prices_by_size)

    print("\n总结：")
    print("通过可视化和分析，我们可以看到不同品种、产地和尺寸的南瓜价格分布情况。")
    print("例如，某些品种的南瓜价格波动较大，而某些产地的南瓜价格相对稳定。")
    print("这些信息可以帮助我们更好地理解南瓜市场的价格动态。")


if __name__ == "__main__":
    main()
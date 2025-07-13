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
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek

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
    numerical_features = ['Month', 'Year', 'Day', 'DayOfWeek']
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
    print(f"MSE: {mse}, R²: {r2}")
    return mse, r2

# 改进后的模型训练和评估
def improved_model_training(X_train, X_test, y_train, y_test):
    # 随机森林回归
    print("开始训练随机森林回归模型...")
    rf_model = RandomForestRegressor(random_state=42)
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search_rf.fit(X_train, y_train)
    best_rf_model = grid_search_rf.best_estimator_
    rf_mse, rf_r2 = train_and_evaluate_model(best_rf_model, X_train, X_test, y_train, y_test)
    print(f"优化后的随机森林回归的均方误差（MSE）: {rf_mse}")
    print(f"优化后的随机森林回归的决定系数（R²）: {rf_r2}")

    # XGBoost回归
    print("开始训练XGBoost回归模型...")
    xgb_model = XGBRegressor(random_state=42)
    param_grid_xgb = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'lambda': [0.1, 1, 10]
    }
    grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search_xgb.fit(X_train, y_train)
    best_xgb_model = grid_search_xgb.best_estimator_
    xgb_mse, xgb_r2 = train_and_evaluate_model(best_xgb_model, X_train, X_test, y_train, y_test)
    print(f"优化后的XGBoost回归的均方误差（MSE）: {xgb_mse}")
    print(f"优化后的XGBoost回归的决定系数（R²）: {xgb_r2}")

    # 支持向量机回归
    print("开始训练支持向量机回归模型...")
    svr_model = SVR()
    param_grid_svr = {
        'C': [0.1, 1, 10],
        'epsilon': [0.1, 0.2, 0.3],
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto', 0.1, 0.01]
    }
    grid_search_svr = GridSearchCV(estimator=svr_model, param_grid=param_grid_svr, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search_svr.fit(X_train, y_train)
    best_svr_model = grid_search_svr.best_estimator_
    svr_mse, svr_r2 = train_and_evaluate_model(best_svr_model, X_train, X_test, y_train, y_test)
    print(f"优化后的支持向量机回归的均方误差（MSE）: {svr_mse}")
    print(f"优化后的支持向量机回归的决定系数（R²）: {svr_r2}")

# 主函数
def main():
    # 加载数据
    print("开始加载数据...")
    data = pd.read_csv('US-pumpkins.csv')
    print(data.head())

    # 数据预处理
    print("开始数据预处理...")
    data_processed = preprocess_data(data)
    print(data_processed.head())

    # 特征工程
    print("开始特征工程...")
    preprocessor, _, _ = engineer_features(data_processed)

    # 数据划分
    print("开始数据划分...")
    X = data_processed.drop(columns=['Average Price'], errors='ignore')
    y = data_processed['Average Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 预处理数据
    print("开始预处理数据...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    print(X_train_processed.shape)
    print(X_test_processed.shape)

    # 改进后的模型训练和评估
    print("开始改进后的模型训练和评估...")
    improved_model_training(X_train_processed, X_test_processed, y_train, y_test)

if __name__ == "__main__":
    main()
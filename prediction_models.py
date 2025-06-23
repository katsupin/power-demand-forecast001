"""
チケット #003: 予測モデル実装
Prophet + scikit-learn による電力需要予測モデル
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Prophet関連
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️ Prophet がインストールされていません。pip install prophet でインストールしてください。")

# scikit-learn関連
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import joblib

class PowerDemandPredictor:
    """電力需要予測の統合クラス"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.trained = False
        
    def train_all_models(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        全モデルの学習実行
        
        Args:
            data: 学習用データ
            
        Returns:
            各モデルの評価結果
        """
        print("🚀 全モデルの学習を開始...")
        
        results = {}
        
        # 1. Prophet モデル
        if PROPHET_AVAILABLE:
            results['Prophet'] = self._train_prophet(data)
        else:
            print("⚠️ Prophet をスキップします")
        
        # 2. 線形回帰
        results['LinearRegression'] = self._train_linear_regression(data)
        
        # 3. Random Forest
        results['RandomForest'] = self._train_random_forest(data)
        
        self.trained = True
        print("✅ 全モデルの学習完了!")
        
        return results
    
    def predict(self, model_name: str, 
                future_data: pd.DataFrame) -> pd.DataFrame:
        """
        指定モデルで予測実行
        
        Args:
            model_name: モデル名
            future_data: 予測用データ
            
        Returns:
            予測結果
        """
        if not self.trained:
            raise ValueError("モデルが学習されていません。train_all_models()を先に実行してください。")
        
        if model_name == 'Prophet' and PROPHET_AVAILABLE:
            return self._predict_prophet(future_data)
        elif model_name in ['LinearRegression', 'RandomForest']:
            return self._predict_sklearn(model_name, future_data)
        else:
            raise ValueError(f"不明なモデル名: {model_name}")
    
    def _train_prophet(self, data: pd.DataFrame) -> Dict[str, float]:
        """Prophet モデルの学習"""
        print("📊 Prophet モデルを学習中...")
        
        # Prophet用のデータ形式に変換
        prophet_data = data[['datetime', 'demand', 'temperature']].copy()
        prophet_data.rename(columns={'datetime': 'ds', 'demand': 'y'}, inplace=True)
        
        # 訓練・テスト分割
        split_idx = int(len(prophet_data) * 0.8)
        train_data = prophet_data[:split_idx]
        test_data = prophet_data[split_idx:]
        
        # モデル作成と学習
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode='multiplicative',
            interval_width=0.95
        )
        model.add_regressor('temperature')
        
        model.fit(train_data)
        
        # 予測と評価
        future = test_data[['ds', 'temperature']].copy()
        forecast = model.predict(future)
        
        # 評価指標計算
        y_true = test_data['y'].values
        y_pred = forecast['yhat'].values
        
        metrics = self._calculate_metrics(y_true, y_pred, "Prophet")
        
        # モデル保存
        self.models['Prophet'] = model
        
        return metrics
    
    def _train_linear_regression(self, data: pd.DataFrame) -> Dict[str, float]:
        """線形回帰モデルの学習"""
        print("📊 線形回帰モデルを学習中...")
        
        # 特徴量準備
        features = ['temperature', 'hour', 'weekday', 'month', 'is_holiday']
        X = data[features]
        y = data['demand']
        
        # 訓練・テスト分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False, random_state=42
        )
        
        # 標準化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # モデル学習
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # 予測と評価
        y_pred = model.predict(X_test_scaled)
        metrics = self._calculate_metrics(y_test, y_pred, "LinearRegression")
        
        # モデル・スケーラー保存
        self.models['LinearRegression'] = model
        self.scalers['LinearRegression'] = scaler
        
        return metrics
    
    def _train_random_forest(self, data: pd.DataFrame) -> Dict[str, float]:
        """Random Forest モデルの学習"""
        print("📊 Random Forest モデルを学習中...")
        
        # データのコピーを作成
        data = data.copy()
        
        # sin/cos特徴量を生成（予測時と同じ方法）
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['day_of_year_sin'] = np.sin(2 * np.pi * data['day_of_year'] / 365)
        data['day_of_year_cos'] = np.cos(2 * np.pi * data['day_of_year'] / 365)
        
        # 特徴量準備（より多くの特徴量を使用）
        features = [
            'temperature', 'hour', 'weekday', 'month', 'is_holiday',
            'hour_sin', 'hour_cos', 'day_of_year_sin', 'day_of_year_cos'
        ]
        
        # 利用可能な特徴量のみ選択
        available_features = [f for f in features if f in data.columns]
        X = data[available_features]
        y = data['demand']
        
        # 欠損値除去
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        # 訓練・テスト分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False, random_state=42
        )
        
        # モデル学習
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # 予測と評価
        y_pred = model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred, "RandomForest")
        
        # 特徴量重要度
        feature_importance = dict(zip(available_features, model.feature_importances_))
        metrics['feature_importance'] = feature_importance
        
        # モデル保存
        self.models['RandomForest'] = model
        
        return metrics
    
    def _predict_prophet(self, future_data: pd.DataFrame) -> pd.DataFrame:
        """Prophet による予測"""
        model = self.models['Prophet']
        
        # データ形式変換
        future = future_data[['datetime', 'temperature_forecast']].copy()
        future.rename(columns={
            'datetime': 'ds', 
            'temperature_forecast': 'temperature'
        }, inplace=True)
        
        # 予測実行
        forecast = model.predict(future)
        
        # 予測値の範囲制限
        predicted_demand = np.clip(forecast['yhat'], 9000, 60000)
        
        # 結果整形
        result = pd.DataFrame({
            'datetime': future['ds'],
            'predicted_demand': predicted_demand,
            'lower_bound': forecast['yhat_lower'],
            'upper_bound': forecast['yhat_upper'],
            'model': 'Prophet'
        })
        
        return result
    
    def _predict_sklearn(self, model_name: str, 
                        future_data: pd.DataFrame) -> pd.DataFrame:
        """scikit-learn モデルによる予測"""
        model = self.models[model_name]
        
        # データのコピーを作成
        future_data = future_data.copy()
        
        # temperature_forecastをtemperatureにリネーム（学習時の特徴量名に合わせる）
        if 'temperature_forecast' in future_data.columns:
            future_data['temperature'] = future_data['temperature_forecast']
        
        # 特徴量準備
        if model_name == 'LinearRegression':
            features = ['temperature', 'hour', 'weekday', 'month', 'is_holiday']
            scaler = self.scalers[model_name]
        else:  # RandomForest
            # day_of_year を生成（学習時と同じ方法）
            future_data['day_of_year'] = future_data['datetime'].dt.dayofyear
            
            # sin/cos特徴量を生成（学習時と同じ方法）
            future_data['hour_sin'] = np.sin(2 * np.pi * future_data['hour'] / 24)
            future_data['hour_cos'] = np.cos(2 * np.pi * future_data['hour'] / 24)
            future_data['day_of_year_sin'] = np.sin(2 * np.pi * future_data['day_of_year'] / 365)
            future_data['day_of_year_cos'] = np.cos(2 * np.pi * future_data['day_of_year'] / 365)
            
            features = [
                'temperature', 'hour', 'weekday', 'month', 'is_holiday',
                'hour_sin', 'hour_cos', 'day_of_year_sin', 'day_of_year_cos'
            ]
        
        # 不足している特徴量を0で埋める
        for f in features:
            if f not in future_data.columns:
                future_data[f] = 0
        
        X = future_data[features]
        
        # 予測実行
        if model_name == 'LinearRegression':
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)
        else:
            predictions = model.predict(X)
        
        # 予測値の範囲制限（現実的な範囲に収める）
        predictions = np.clip(predictions, 9000, 60000)  # 9-60 GW
        
        # 結果整形
        result = pd.DataFrame({
            'datetime': future_data['datetime'],
            'predicted_demand': predictions,
            'model': model_name
        })
        
        return result
    
    def _calculate_metrics(self, y_true: np.ndarray, 
                          y_pred: np.ndarray, 
                          model_name: str) -> Dict[str, float]:
        """評価指標の計算"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        print(f"   {model_name} - MAPE: {mape:.2f}%, R²: {r2:.3f}, RMSE: {rmse:.0f}")
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
    
    def compare_models(self, results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """モデル比較表の作成"""
        comparison_data = []
        
        for model_name, metrics in results.items():
            if 'feature_importance' in metrics:
                # 特徴量重要度は除外
                clean_metrics = {k: v for k, v in metrics.items() 
                               if k != 'feature_importance'}
            else:
                clean_metrics = metrics
            
            comparison_data.append({
                'Model': model_name,
                **clean_metrics
            })
        
        return pd.DataFrame(comparison_data)
    
    def save_models(self, filepath: str):
        """モデルの保存"""
        save_data = {
            'models': self.models,
            'scalers': self.scalers,
            'trained': self.trained
        }
        joblib.dump(save_data, filepath)
        print(f"💾 モデルを保存しました: {filepath}")
    
    def load_models(self, filepath: str):
        """モデルの読み込み"""
        save_data = joblib.load(filepath)
        self.models = save_data['models']
        self.scalers = save_data['scalers']
        self.trained = save_data['trained']
        print(f"📂 モデルを読み込みました: {filepath}")

# ユーティリティ関数
def create_ensemble_prediction(predictions: Dict[str, pd.DataFrame], 
                             weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    アンサンブル予測の作成
    
    Args:
        predictions: 各モデルの予測結果
        weights: 各モデルの重み
        
    Returns:
        アンサンブル予測結果
    """
    if weights is None:
        # デフォルト重み（Random Forest重視）
        weights = {
            'Prophet': 0.3,
            'LinearRegression': 0.2,
            'RandomForest': 0.5
        }
    
    # 最初の予測をベースに結果を初期化
    first_pred = list(predictions.values())[0]
    result = first_pred[['datetime']].copy()
    
    # 重み付き平均を計算
    ensemble_pred = np.zeros(len(result))
    total_weight = 0
    
    for model_name, pred_df in predictions.items():
        if model_name in weights:
            weight = weights[model_name]
            ensemble_pred += weight * pred_df['predicted_demand'].values
            total_weight += weight
    
    # 正規化
    ensemble_pred /= total_weight
    
    result['predicted_demand'] = ensemble_pred
    result['model'] = 'Ensemble'
    
    return result

# テスト実行
if __name__ == "__main__":
    from data_preparation import create_demo_dataset
    
    print("🧪 予測モデルのテスト実行...")
    
    # データ準備
    demo_data = create_demo_dataset()
    
    # モデル学習
    predictor = PowerDemandPredictor()
    results = predictor.train_all_models(demo_data['processed_data'])
    
    # 結果比較
    comparison = predictor.compare_models(results)
    print("\n📊 モデル比較結果:")
    print(comparison.round(3))
    
    # 予測テスト
    if predictor.trained:
        forecast_24h = demo_data['weather_forecast_24h']
        
        # 各モデルで予測
        predictions = {}
        for model_name in predictor.models.keys():
            pred = predictor.predict(model_name, forecast_24h)
            predictions[model_name] = pred
            print(f"\n{model_name} 予測結果（先頭5件）:")
            print(pred.head())
        
        print("✅ 予測モデルテスト完了!")
"""
チケット #002: データ準備
電力需給予測デモ用のダミーデータ生成と前処理機能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Tuple, Optional

class PowerDemandDataGenerator:
    """電力需要データのダミー生成クラス"""
    
    def __init__(self, seed: int = 42):
        """
        初期化
        
        Args:
            seed: 乱数シード（再現性のため）
        """
        np.random.seed(seed)
        random.seed(seed)
        
        # 基本パラメータ
        self.base_demand = 30000  # 基本需要量（MW）
        self.temp_optimal = 22    # 最適温度（℃）
        
        # 実データベース季節係数（過去実績より算出想定）
        self.seasonal_factors_cache = None
        
    def generate_historical_data(self, 
                                start_date: str = None, 
                                days: int = 365*3) -> pd.DataFrame:
        """
        過去データの生成（学習用）
        
        Args:
            start_date: 開始日（Noneの場合は現在から過去に遡る）
            days: 生成日数
            
        Returns:
            過去の電力需要データ
        """
        print(f"📊 過去{days}日分のデータを生成中...")
        
        # 日時インデックス作成（1時間単位）
        if start_date is None:
            # 現在時刻から過去に遡って生成
            from datetime import datetime, timedelta
            end_time = datetime.now().replace(minute=0, second=0, microsecond=0)
            start = end_time - timedelta(days=days)
        else:
            start = pd.to_datetime(start_date)
        dates = pd.date_range(start=start, periods=days*24, freq='H')
        
        data = []
        for date in dates:
            # 1. 基本的な時間パターン
            hour_factor = self._get_hourly_factor(date.hour)
            
            # 2. 曜日パターン
            weekday_factor = self._get_weekday_factor(date.weekday())
            
            # 3. 季節パターン
            season_factor = self._get_seasonal_factor(date.dayofyear)
            
            # 4. 気温生成と電力への影響
            temperature = self._generate_temperature(date)
            temp_factor = self._get_temperature_factor(temperature)
            
            # 5. 特殊日の考慮
            special_factor = self._get_special_day_factor(date)
            
            # 6. ランダムノイズ
            noise = np.random.normal(1.0, 0.05)
            
            # 最終的な電力需要計算
            demand = (self.base_demand * 
                     hour_factor * 
                     weekday_factor * 
                     season_factor * 
                     temp_factor * 
                     special_factor * 
                     noise)
            
            # データ記録（需要の上下限設定）
            demand_clamped = max(demand, self.base_demand * 0.3)  # 最低需要保証
            demand_clamped = min(demand_clamped, self.base_demand * 2.0)  # 最大需要制限
            
            data.append({
                'datetime': date,
                'demand': demand_clamped,
                'temperature': temperature,
                'hour': date.hour,
                'weekday': date.weekday(),
                'month': date.month,
                'day_of_year': date.dayofyear,
                'is_holiday': 1 if date.weekday() >= 5 else 0,
                'is_special_day': 1 if special_factor != 1.0 else 0,
                'season': self._get_season(date.month)
            })
        
        df = pd.DataFrame(data)
        print(f"✅ データ生成完了: {len(df):,}行")
        return df
    
    def generate_weather_forecast(self, 
                                 start_date: Optional[str] = None,
                                 hours: int = 24) -> pd.DataFrame:
        """
        天気予報データの生成（予測用）
        
        Args:
            start_date: 開始日時（Noneの場合は現在時刻）
            hours: 予測時間数
            
        Returns:
            天気予報データ
        """
        if start_date is None:
            start = datetime.now().replace(minute=0, second=0, microsecond=0)
        else:
            start = pd.to_datetime(start_date)
        
        dates = pd.date_range(start=start, periods=hours, freq='H')
        
        forecast_data = []
        for date in dates:
            temp = self._generate_temperature(date, is_forecast=True)
            forecast_data.append({
                'datetime': date,
                'temperature_forecast': temp,
                'hour': date.hour,
                'weekday': date.weekday(),
                'month': date.month,
                'is_holiday': 1 if date.weekday() >= 5 else 0
            })
        
        return pd.DataFrame(forecast_data)
    
    def _get_hourly_factor(self, hour: int) -> float:
        """時間帯による需要変動係数"""
        # 朝夕にピークを持つパターン
        if 6 <= hour <= 9:  # 朝のピーク
            return 1.0 + 0.3 * np.sin((hour - 6) * np.pi / 4)
        elif 17 <= hour <= 21:  # 夕方のピーク
            return 1.0 + 0.4 * np.sin((hour - 17) * np.pi / 5)
        elif 22 <= hour <= 5:  # 夜間の低需要
            return 0.7 + 0.1 * np.sin((hour - 22) * np.pi / 8)
        else:  # 日中
            return 0.9 + 0.1 * np.sin((hour - 10) * np.pi / 8)
    
    def _get_weekday_factor(self, weekday: int) -> float:
        """曜日による需要変動係数"""
        if weekday < 5:  # 平日
            return 1.0
        elif weekday == 5:  # 土曜日
            return 0.85
        else:  # 日曜日
            return 0.80
    
    def _get_seasonal_factor(self, day_of_year: int) -> float:
        """季節による需要変動係数"""
        # 日本の電力実績に基づく季節係数（夏>冬>春秋の順）
        # 夏季冷房需要が最大、冬季暖房が次点、春秋が最小
        monthly_factors = {
            1: 1.10,   # 1月（冬）
            2: 1.06,   # 2月（冬）
            3: 0.95,   # 3月（春）
            4: 0.92,   # 4月（春）
            5: 0.95,   # 5月（春末）
            6: 1.20,   # 6月（夏開始）
            7: 1.35,   # 7月（夏ピーク）
            8: 1.32,   # 8月（夏ピーク）
            9: 1.10,   # 9月（秋）
            10: 0.98,  # 10月（秋）
            11: 1.05,  # 11月（冬開始）
            12: 1.12   # 12月（冬）
        }
        
        # day_of_yearから月を推定
        from datetime import datetime, timedelta
        base_date = datetime(2024, 1, 1) + timedelta(days=day_of_year - 1)
        month = base_date.month
        
        return monthly_factors.get(month, 1.0)
    
    def _generate_temperature(self, date: datetime, is_forecast: bool = False) -> float:
        """気温の生成"""
        # 基本的な季節変動
        base_temp = 15 + 12 * np.sin((date.dayofyear - 80) * 2 * np.pi / 365)
        
        # 時間による変動
        hour_variation = 5 * np.sin((date.hour - 6) * np.pi / 12)
        
        # ランダム変動
        if is_forecast:
            # 予報は少し不確実性を追加
            noise = np.random.normal(0, 3)
        else:
            noise = np.random.normal(0, 2)
        
        return base_temp + hour_variation + noise
    
    def _get_temperature_factor(self, temperature: float) -> float:
        """気温による需要変動係数"""
        # 22℃を最適とし、そこから離れるほど需要増（影響を最小限に）
        temp_diff = abs(temperature - self.temp_optimal)
        if temp_diff <= 5:
            return 1.0
        else:
            return 1.0 + 0.008 * (temp_diff - 5)
    
    def _get_special_day_factor(self, date: datetime) -> float:
        """特殊日による需要変動係数"""
        # 大型連休や年末年始の需要減
        if (date.month == 8 and 13 <= date.day <= 16) or \
           (date.month == 12 and date.day >= 29) or \
           (date.month == 1 and date.day <= 3):
            return 0.75
        
        # ゴールデンウィーク
        if date.month == 5 and 3 <= date.day <= 5:
            return 0.80
        
        return 1.0
    
    def _get_season(self, month: int) -> str:
        """月から季節を判定"""
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "autumn"

class DataPreprocessor:
    """データ前処理クラス"""
    
    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        機械学習用の特徴量を作成
        
        Args:
            df: 元データ
            
        Returns:
            特徴量追加済みデータ
        """
        df = df.copy()
        
        # 時間的特徴量
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # ラグ特徴量
        df['demand_lag_1h'] = df['demand'].shift(1)
        df['demand_lag_24h'] = df['demand'].shift(24)
        df['demand_lag_168h'] = df['demand'].shift(168)  # 1週間前
        
        # 移動平均
        df['demand_ma_24h'] = df['demand'].rolling(window=24, min_periods=1).mean()
        df['temp_ma_3h'] = df['temperature'].rolling(window=3, min_periods=1).mean()
        
        # 温度関連特徴量
        df['temp_squared'] = df['temperature'] ** 2
        df['temp_dev_from_optimal'] = abs(df['temperature'] - 22)
        
        return df
    
    @staticmethod
    def prepare_ml_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        機械学習用にデータを準備
        
        Args:
            df: 前処理済みデータ
            
        Returns:
            特徴量、目的変数
        """
        # 特徴量選択
        feature_columns = [
            'temperature', 'hour', 'weekday', 'month', 'is_holiday',
            'hour_sin', 'hour_cos', 'day_of_year_sin', 'day_of_year_cos',
            'demand_lag_1h', 'demand_lag_24h', 'demand_lag_168h',
            'temp_ma_3h', 'temp_squared', 'temp_dev_from_optimal'
        ]
        
        # 欠損値を含む行を除去
        df_clean = df.dropna()
        
        X = df_clean[feature_columns]
        y = df_clean['demand']
        
        return X, y

# ユーティリティ関数
def load_sample_data(days: int = 365*2) -> pd.DataFrame:
    """
    サンプルデータの読み込み
    
    Args:
        days: 生成日数
        
    Returns:
        サンプルデータ
    """
    generator = PowerDemandDataGenerator()
    return generator.generate_historical_data(days=days)

def create_demo_dataset() -> dict:
    """
    デモ用のデータセット作成
    
    Returns:
        デモ用データセット
    """
    print("🚀 デモ用データセットを作成中...")
    
    generator = PowerDemandDataGenerator()
    
    # 過去2年分のデータ
    historical_data = generator.generate_historical_data(days=365*2)
    
    # 今後24時間の天気予報
    weather_forecast = generator.generate_weather_forecast(hours=24)
    
    # 今後1週間の天気予報
    weekly_forecast = generator.generate_weather_forecast(hours=24*7)
    
    # 前処理
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.create_features(historical_data)
    X, y = preprocessor.prepare_ml_data(processed_data)
    
    dataset = {
        'historical_data': historical_data,
        'weather_forecast_24h': weather_forecast,
        'weather_forecast_7d': weekly_forecast,
        'processed_data': processed_data,
        'features': X,
        'targets': y
    }
    
    print("✅ データセット作成完了!")
    print(f"   - 過去データ: {len(historical_data):,}行")
    print(f"   - 24時間予報: {len(weather_forecast)}行")
    print(f"   - 1週間予報: {len(weekly_forecast)}行")
    print(f"   - 機械学習用データ: {len(X):,}行 × {len(X.columns)}特徴量")
    
    return dataset

# テスト実行
if __name__ == "__main__":
    # デモデータセット作成
    demo_data = create_demo_dataset()
    
    # 基本統計の表示
    print("\n📈 基本統計:")
    print(demo_data['historical_data']['demand'].describe())
    
    print("\n🌡️ 気温統計:")
    print(demo_data['historical_data']['temperature'].describe())
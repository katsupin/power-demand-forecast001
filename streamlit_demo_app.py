"""
電力需給予測システム - Streamlit デモアプリケーション
AI技術を活用した電力需給予測と最適化のデモンストレーション
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 自作モジュール
try:
    from data_preparation import create_demo_dataset, PowerDemandDataGenerator
    from prediction_models import PowerDemandPredictor, create_ensemble_prediction
    from auth_manager import AuthManager
except ImportError:
    st.error("❌ 必要なモジュールが見つかりません。data_preparation.py と prediction_models.py が同じディレクトリにあることを確認してください。")
    st.stop()

# ページ設定
st.set_page_config(
    page_title="電力需給予測システム - デモ",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 5px solid #1f77b4;
}
.improvement-positive {
    color: #00ff00;
}
.improvement-negative {
    color: #ff0000;
}
</style>
""", unsafe_allow_html=True)

# データとモデルの初期化（キャッシュ機能付き）
@st.cache_data
def load_demo_data_fast():
    """デモデータの生成"""
    with st.spinner("📊 データ生成中..."):
        return create_demo_dataset()

@st.cache_resource
def load_models_fast():
    """AIモデルの学習"""
    with st.spinner("🤖 モデル学習中..."):
        demo_data = load_demo_data_fast()
        predictor = PowerDemandPredictor()
        
        # 全モデルを学習
        results = predictor.train_all_models(demo_data['historical_data'])
        
        return predictor, results

# サイドバーの構成（上部: デモ機能、下部: データ管理）

# メイン関数
def main():
    # 認証チェック
    auth_manager = AuthManager()
    
    if not auth_manager.is_authenticated():
        auth_manager.login_form()
        return
    
    # ヘッダー（認証後）
    col_header, col_logout = st.columns([4, 1])
    with col_header:
        st.markdown('<h1 class="main-header">⚡ 電力需給予測システム - AI実証デモ</h1>', unsafe_allow_html=True)
    with col_logout:
        if st.button("🚪 ログアウト", help="認証を解除してログアウト"):
            auth_manager.logout()
    
    st.markdown("---")
    
    # データとモデルの準備（高速読み込み）
    demo_data = load_demo_data_fast()
    predictor, model_results = load_models_fast()
    
    # サイドバー上部: デモ機能
    st.sidebar.header("🎛️ デモ設定")
    demo_mode = st.sidebar.radio(
        "デモモード選択",
        ["📊 予測精度の実証", "💰 コスト削減シミュレーション", "🔮 リアルタイム予測", "📈 モデル比較分析"],
        index=0
    )
    
    # サイドバー下部: データ管理
    st.sidebar.markdown("---")  # 区切り線
    st.sidebar.header("🗂️ データ管理")
    
    # データ管理ボタン（2列配置）
    st.sidebar.caption("💡 再構築：新しい学習データ生成+AIモデル再学習")
    col_btn1, col_btn2 = st.sidebar.columns(2)
    
    with col_btn1:
        if st.sidebar.button("🔄 学習データ・モデル再構築", help="学習データを新規生成し、全AIモデルを再学習します（数分かかります）"):
            with st.spinner("📊 学習データ再生成中... 🤖 AIモデル再学習中..."):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.sidebar.success("✅ 学習データ・モデル再構築完了！")
                st.rerun()
    
    with col_btn2:
        if st.sidebar.button("🗑️ クリア", help="全てのキャッシュファイルを削除"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.sidebar.success("✅ クリア完了！")
            st.rerun()
    
    # CSVエクスポート機能
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 データエクスポート")
    
    if demo_data is not None:
        hist_data = demo_data['historical_data']
        st.sidebar.success("✅ エクスポート可能")
        st.sidebar.caption(f"データ: {len(hist_data):,}行 | モデル: 3個")
        
        # エクスポートボタン
        if st.sidebar.button("💾 CSVダウンロード", help="履歴データをCSV形式でダウンロード"):
            # CSVデータを生成
            csv = hist_data.to_csv(index=False)
            
            # ダウンロードボタン
            st.sidebar.download_button(
                label="📊 historical_data.csv",
                data=csv,
                file_name=f"power_demand_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="クリックしてCSVファイルをダウンロード"
            )
    else:
        st.sidebar.warning("⚠️ データなし")
        st.sidebar.caption("先にデータ生成が必要です")
    
    # 各モードの実行
    if demo_mode == "📊 予測精度の実証":
        show_accuracy_demo(demo_data, predictor, model_results)
    elif demo_mode == "💰 コスト削減シミュレーション":
        show_cost_simulation()
    elif demo_mode == "🔮 リアルタイム予測":
        show_realtime_prediction(demo_data, predictor)
    elif demo_mode == "📈 モデル比較分析":
        show_model_comparison(model_results)

def show_accuracy_demo(demo_data, predictor, model_results):
    """予測精度実証モード"""
    st.header("📊 予測精度の実証")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 過去の電力需要推移")
        
        # 直近1週間のデータ表示（最新の168時間）
        recent_data = demo_data['historical_data'].tail(24*7)
        
        # 実際の日付範囲を表示
        if len(recent_data) > 0:
            start_date = recent_data['datetime'].min()
            end_date = recent_data['datetime'].max()
            st.caption(f"📅 データ期間: {start_date.strftime('%Y-%m-%d %H:%M')} ～ {end_date.strftime('%Y-%m-%d %H:%M')}")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent_data['datetime'],
            y=recent_data['demand'],
            mode='lines',
            name='実績需要',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title="直近1週間の電力需要実績",
            xaxis_title="日時",
            yaxis_title="電力需要 (MW)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🔍 需要パターン分析")
        
        # 時間帯別・曜日別分析
        tab1, tab2 = st.tabs(["時間帯別", "曜日別"])
        
        with tab1:
            # 季節別時間帯需要分析
            hist_data = demo_data['historical_data'].copy()
            
            # 季節マッピング
            season_map = {
                'spring': '春 (3-5月)',
                'summer': '夏 (6-8月)', 
                'autumn': '秋 (9-11月)',
                'winter': '冬 (12-2月)'
            }
            
            fig = go.Figure()
            
            # 季節別の時間帯平均を計算・プロット
            colors = {'spring': '#90EE90', 'summer': '#FF6347', 'autumn': '#DEB887', 'winter': '#87CEEB'}
            
            for season, season_name in season_map.items():
                season_data = hist_data[hist_data['season'] == season]
                if len(season_data) > 0:
                    hourly_avg = season_data.groupby('hour')['demand'].mean()
                    
                    fig.add_trace(go.Scatter(
                        x=hourly_avg.index,
                        y=hourly_avg.values,
                        mode='lines+markers',
                        name=season_name,
                        line=dict(color=colors[season], width=2),
                        hovertemplate='%{fullData.name}<br>時刻: %{x}時<br>需要: %{y:,.0f} MW<extra></extra>'
                    ))
            
            # 年間平均も追加
            yearly_avg = hist_data.groupby('hour')['demand'].mean()
            fig.add_trace(go.Scatter(
                x=yearly_avg.index,
                y=yearly_avg.values,
                mode='lines',
                name='年間平均',
                line=dict(color='black', width=3, dash='dash'),
                hovertemplate='年間平均<br>時刻: %{x}時<br>需要: %{y:,.0f} MW<extra></extra>'
            ))
            
            fig.update_layout(
                title="季節別時間帯需要パターン",
                xaxis_title="時刻",
                yaxis_title="平均需要 (MW)",
                hovermode='x unified',
                legend=dict(x=0.02, y=0.98)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            daily_avg = demo_data['historical_data'].groupby('weekday')['demand'].mean()
            days = ['月', '火', '水', '木', '金', '土', '日']
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=days,
                y=daily_avg.values,
                name='平均需要',
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title="曜日別平均需要",
                xaxis_title="曜日",
                yaxis_title="平均需要 (MW)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # AIモデル性能比較
    st.subheader("🤖 AIモデル性能比較")
    
    # 技術進歩順に表示するための順序定義
    model_order = ['LinearRegression', 'Prophet', 'RandomForest']
    model_display_names = {
        'LinearRegression': '📊 線形回帰（従来手法）',
        'Prophet': '📈 Prophet（時系列AI）', 
        'RandomForest': '🚀 Random Forest（高性能ML）'
    }
    
    # 指標説明用のヘルプテキスト
    metrics_help = {
        'MAPE': 'Mean Absolute Percentage Error: 予測誤差率（小さいほど高精度。5%以下で優秀、3%以下で非常に優秀）',
        'R2': 'R-squared: 決定係数（1.0に近いほど優秀。0.9以上で高性能、0.95以上で非常に高性能）'
    }
    
    col1, col2, col3 = st.columns(3)
    columns = [col1, col2, col3]
    
    # 技術進歩順に表示
    for i, model_name in enumerate(model_order):
        if model_name in model_results and 'MAPE' in model_results[model_name]:
            metrics = model_results[model_name]
            with columns[i]:
                # 改善率計算（RandomForestのみ）
                improvement = ""
                if model_name == 'RandomForest' and 'LinearRegression' in model_results:
                    baseline_mape = model_results['LinearRegression']['MAPE']
                    current_mape = metrics['MAPE']
                    improvement_pct = ((baseline_mape - current_mape) / baseline_mape) * 100
                    improvement = f" ({improvement_pct:.1f}% 改善)"
                
                # 表示名取得
                display_name = model_display_names.get(model_name, model_name)
                
                # メトリック表示（ヘルプ付き）
                st.metric(
                    label=display_name,
                    value=f"MAPE: {metrics['MAPE']:.2f}%",
                    delta=f"R²: {metrics['R2']:.3f}{improvement}",
                    help=f"**MAPE**: {metrics_help['MAPE']}\n\n**R²**: {metrics_help['R2']}"
                )
    
    # 予測結果の可視化
    st.subheader("🎯 予測精度の可視化")
    
    # テストデータで予測比較
    test_period_hours = st.slider("予測期間（時間）", 24, 168, 72)
    
    # 予測実行
    forecast_data = demo_data['weather_forecast_24h']
    if test_period_hours > 24:
        extended_forecast = demo_data['weather_forecast_7d'].head(test_period_hours)
        forecast_data = extended_forecast
    
    predictions = {}
    for model_name in predictor.models.keys():
        try:
            pred = predictor.predict(model_name, forecast_data)
            predictions[model_name] = pred
        except Exception as e:
            st.warning(f"⚠️ {model_name} の予測に失敗: {str(e)}")
    
    if predictions:
        # アンサンブル予測
        ensemble_pred = create_ensemble_prediction(predictions)
        
        # 可視化
        fig = go.Figure()
        
        # 実績データは時間軸が混在するため非表示
        # 必要に応じて将来バージョンで時間軸調整を実装
        
        # 各モデルの予測
        colors = ['red', 'blue', 'green', 'purple']
        for i, (model_name, pred_df) in enumerate(predictions.items()):
            fig.add_trace(go.Scatter(
                x=pred_df['datetime'],
                y=pred_df['predicted_demand'],
                mode='lines',
                name=f'{model_name}予測',
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        # アンサンブル予測と信頼区間
        if ensemble_pred is not None:
            # 予測値の標準偏差から信頼区間を計算
            pred_values = []
            for pred_df in predictions.values():
                pred_values.append(pred_df['predicted_demand'].values)
            
            import numpy as np
            pred_array = np.array(pred_values)
            pred_std = np.std(pred_array, axis=0)
            ensemble_mean = ensemble_pred['predicted_demand'].values
            
            # 95%信頼区間（±1.96σ）
            upper_bound = ensemble_mean + 1.96 * pred_std
            lower_bound = ensemble_mean - 1.96 * pred_std
            
            # 信頼区間の影エリア
            fig.add_trace(go.Scatter(
                x=list(ensemble_pred['datetime']) + list(ensemble_pred['datetime'][::-1]),
                y=list(upper_bound) + list(lower_bound[::-1]),
                fill='toself',
                fillcolor='rgba(128,128,128,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95%信頼区間',
                hoverinfo='skip'
            ))
            
            # アンサンブル予測線
            fig.add_trace(go.Scatter(
                x=ensemble_pred['datetime'],
                y=ensemble_pred['predicted_demand'],
                mode='lines',
                name='アンサンブル予測',
                line=dict(color='black', width=3)
            ))
        
        # Y軸範囲を動的に設定
        all_predictions = []
        for pred_df in predictions.values():
            all_predictions.extend(pred_df['predicted_demand'].tolist())
        if ensemble_pred is not None:
            all_predictions.extend(ensemble_pred['predicted_demand'].tolist())
        
        if all_predictions:
            min_val = min(all_predictions)
            max_val = max(all_predictions)
            margin = (max_val - min_val) * 0.1  # 10%のマージン
            y_min = max(0, min_val - margin)
            y_max = max_val + margin
        else:
            y_min, y_max = 20000, 50000
        
        fig.update_layout(
            title=f"今後{test_period_hours}時間の電力需要予測",
            xaxis_title="日時",
            yaxis_title="電力需要 (MW)",
            hovermode='x unified',
            height=500,
            yaxis=dict(
                range=[y_min, y_max],  # 動的なY軸範囲
                title="電力需要 (MW)"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_cost_simulation():
    """コスト削減シミュレーションモード"""
    st.header("💰 コスト削減シミュレーション")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ パラメータ設定")
        
        annual_demand = st.number_input(
            "年間電力需要 (GWh)", 
            min_value=10, 
            max_value=1000, 
            value=150, 
            step=10,
            help="製造業の年間電力消費量（参考：小規模10-50GWh、中規模50-200GWh、大規模200-1000GWh）"
        )
        
        current_error = st.slider(
            "現在の予測誤差率 (%)", 
            min_value=3.0, 
            max_value=20.0, 
            value=8.0, 
            step=0.5,
            help="既存の予測手法の誤差率"
        )
        
        ai_error = st.slider(
            "AI導入後の予測誤差率 (%)", 
            min_value=1.0, 
            max_value=10.0, 
            value=4.0, 
            step=0.5,
            help="AI予測システム導入後の誤差率"
        )
        
        unit_cost = st.number_input(
            "予測誤差による追加コスト (円/kWh)", 
            min_value=1.0, 
            max_value=50.0, 
            value=15.0, 
            step=1.0,
            help="インバランス料金やピーク調整コスト"
        )
    
    with col2:
        st.subheader("💼 投資パラメータ")
        
        implementation_cost = st.number_input(
            "初期導入コスト (百万円)", 
            min_value=10, 
            max_value=200, 
            value=30, 
            step=10,
            help="システム開発・導入費用（小規模:10-20百万円、中規模:20-50百万円、大規模:50-200百万円）"
        )
        
        annual_operation_cost = st.number_input(
            "年間運用コスト (百万円)", 
            min_value=1, 
            max_value=20, 
            value=5, 
            step=1,
            help="保守・運用・ライセンス費用（小規模:1-3百万円、中規模:3-8百万円、大規模:8-20百万円）"
        )
        
        analysis_years = st.slider(
            "分析期間 (年)", 
            min_value=3, 
            max_value=15, 
            value=10,
            help="投資効果を分析する期間"
        )
    
    # 計算実行
    st.subheader("📊 計算結果")
    
    # 修正版計算（より現実的なモデル）
    # インバランス料金は予測誤差率に比例（需給調整市場の実態に即した計算）
    # 予測誤差1%あたりのコスト影響を年間電力需要の0.1%と仮定
    cost_impact_rate = 0.001  # 予測誤差1%あたりのコスト率
    
    current_cost = annual_demand * current_error * cost_impact_rate * unit_cost
    ai_cost = annual_demand * ai_error * cost_impact_rate * unit_cost
    annual_saving = current_cost - ai_cost - annual_operation_cost
    
    if annual_saving > 0:
        roi_years = implementation_cost / annual_saving
    else:
        roi_years = float('inf')
    
    # 結果表示
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "現在の年間誤差コスト",
            f"{current_cost:.0f}百万円",
            help="現在の予測誤差による年間インバランス料金等の追加コスト（誤差率の2乗に比例）"
        )
    
    with col2:
        st.metric(
            "AI導入後の年間誤差コスト",
            f"{ai_cost:.0f}百万円",
            f"-{current_cost - ai_cost:.0f}百万円",
            help="AI導入後の年間誤差コスト（削減効果は運用コスト差引前）"
        )
    
    with col3:
        st.metric(
            "年間純削減額",
            f"{annual_saving:.0f}百万円",
            help="運用コスト控除後の年間純削減額（この金額で投資回収を計算）"
        )
    
    with col4:
        if roi_years < 100:
            st.metric(
                "投資回収期間",
                f"{roi_years:.1f}年",
                help="初期投資を回収するまでの期間"
            )
        else:
            st.metric(
                "投資回収期間",
                "回収困難",
                help="現在のパラメータでは投資回収が困難"
            )
    
    # 累積効果グラフ
    st.subheader("📈 累積コスト削減効果")
    
    years = np.arange(0, analysis_years + 1)
    cumulative_saving = [-implementation_cost + annual_saving * y for y in years]
    
    fig = go.Figure()
    
    # 累積削減額
    fig.add_trace(go.Scatter(
        x=years,
        y=cumulative_saving,
        mode='lines+markers',
        name='累積削減額',
        line=dict(width=3),
        fill='tonexty'
    ))
    
    # ゼロライン
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="損益分岐点")
    
    # 投資回収ポイント
    if roi_years < analysis_years:
        fig.add_vline(
            x=roi_years, 
            line_dash="dash", 
            line_color="green",
            annotation_text=f"投資回収: {roi_years:.1f}年"
        )
    
    fig.update_layout(
        title="投資対効果の推移",
        xaxis_title="導入後の年数",
        yaxis_title="累積削減額（百万円）",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 詳細分析
    with st.expander("📋 詳細分析結果"):
        st.write("### 💡 主要指標")
        
        total_saving = annual_saving * analysis_years
        roi_percentage = (total_saving - implementation_cost) / implementation_cost * 100 if implementation_cost > 0 else 0
        
        analysis_data = {
            "指標": [
                "年間削減率",
                "総削減額（期間内）",
                "ROI",
                "NPV（10年、割引率5%）"
            ],
            "値": [
                f"{((current_cost - ai_cost) / current_cost * 100):.1f}%",
                f"{total_saving:.0f}百万円",
                f"{roi_percentage:.1f}%",
                f"{calculate_npv(annual_saving, implementation_cost, 10, 0.05):.0f}百万円"
            ]
        }
        
        st.table(pd.DataFrame(analysis_data))

def show_realtime_prediction(demo_data, predictor):
    """リアルタイム予測モード"""
    st.header("🔮 リアルタイム予測デモ")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📅 予測条件設定")
        
        # 予測日時設定
        pred_date = st.date_input(
            "予測日",
            value=datetime.now().date() + timedelta(days=1),
            help="予測を実行する日付"
        )
        
        pred_hour = st.slider(
            "予測時刻",
            min_value=0,
            max_value=23,
            value=12,
            help="予測を実行する時刻"
        )
        
        # 気象条件設定
        pred_temp = st.slider(
            "予想気温 (℃)",
            min_value=-10,
            max_value=45,
            value=25,
            help="予測時点の気温"
        )
        
        is_holiday = st.checkbox(
            "祝日・休日",
            help="祝日や休日の場合はチェック"
        )
        
        # 予測実行ボタン
        if st.button("🚀 予測実行", type="primary"):
            st.session_state.prediction_executed = True
            st.session_state.pred_params = {
                'date': pred_date,
                'hour': pred_hour,
                'temp': pred_temp,
                'holiday': is_holiday
            }
    
    with col2:
        st.subheader("📊 予測結果")
        
        if hasattr(st.session_state, 'prediction_executed') and st.session_state.prediction_executed:
            params = st.session_state.pred_params
            
            # 予測データ作成
            pred_datetime = datetime.combine(params['date'], datetime.min.time().replace(hour=params['hour']))
            
            single_forecast = pd.DataFrame({
                'datetime': [pred_datetime],
                'temperature_forecast': [params['temp']],
                'hour': [params['hour']],
                'weekday': [pred_datetime.weekday()],
                'month': [pred_datetime.month],
                'is_holiday': [1 if params['holiday'] else 0]
            })
            
            # 各モデルで予測
            predictions = {}
            for model_name in predictor.models.keys():
                try:
                    pred = predictor.predict(model_name, single_forecast)
                    predictions[model_name] = pred.iloc[0]['predicted_demand']
                except Exception as e:
                    st.warning(f"⚠️ {model_name} 予測エラー: {str(e)}")
            
            if predictions:
                # アンサンブル予測
                ensemble_pred = np.average(list(predictions.values()))
                
                # 結果表示
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric(
                        "🎯 アンサンブル予測",
                        f"{ensemble_pred:,.0f} MW",
                        help="複数モデルの統合予測結果"
                    )
                    
                    confidence_interval = ensemble_pred * 0.05
                    st.caption(f"信頼区間: ±{confidence_interval:,.0f} MW")
                
                with col_b:
                    # 個別モデル結果（他のページと同じ順序で表示）
                    st.write("**個別モデル結果:**")
                    model_order = ['LinearRegression', 'Prophet', 'RandomForest']
                    model_display_names = {
                        'LinearRegression': '📊 線形回帰（従来手法）',
                        'Prophet': '📈 Prophet（時系列AI）', 
                        'RandomForest': '🚀 Random Forest（高性能ML）'
                    }
                    
                    for model_name in model_order:
                        if model_name in predictions:
                            display_name = model_display_names.get(model_name, model_name)
                            pred_value = predictions[model_name]
                            st.write(f"- {display_name}: {pred_value:,.0f} MW")
        else:
            st.info("👈 左側で条件を設定し、予測実行ボタンを押してください")
    
    # 24時間予測
    st.subheader("📈 今後24時間の需要予測")
    
    if st.button("24時間予測実行"):
        with st.spinner("予測計算中..."):
            # 24時間予測データ作成
            forecast_24h = demo_data['weather_forecast_24h'].copy()
            
            # 現在の設定を反映
            if hasattr(st.session_state, 'pred_params'):
                base_temp = st.session_state.pred_params['temp']
                # 気温の日変動を追加
                for i in range(len(forecast_24h)):
                    hour_variation = 5 * np.sin((forecast_24h.iloc[i]['hour'] - 6) * np.pi / 12)
                    forecast_24h.iloc[i, forecast_24h.columns.get_loc('temperature_forecast')] = base_temp + hour_variation
            
            # 予測実行
            model_predictions = {}
            for model_name in predictor.models.keys():
                try:
                    pred = predictor.predict(model_name, forecast_24h)
                    model_predictions[model_name] = pred
                except Exception as e:
                    st.warning(f"⚠️ {model_name} 24時間予測エラー: {str(e)}")
            
            if model_predictions:
                # アンサンブル予測
                ensemble_24h = create_ensemble_prediction(model_predictions)
                
                # 可視化
                fig = go.Figure()
                
                # 各モデルの予測（薄い線）
                colors = ['lightcoral', 'lightblue', 'lightgreen']
                for i, (model_name, pred_df) in enumerate(model_predictions.items()):
                    fig.add_trace(go.Scatter(
                        x=pred_df['datetime'],
                        y=pred_df['predicted_demand'],
                        mode='lines',
                        name=model_name,
                        line=dict(color=colors[i % len(colors)], width=1),
                        opacity=0.7
                    ))
                
                # アンサンブル予測（太い線）
                fig.add_trace(go.Scatter(
                    x=ensemble_24h['datetime'],
                    y=ensemble_24h['predicted_demand'],
                    mode='lines+markers',
                    name='アンサンブル予測',
                    line=dict(color='darkblue', width=3)
                ))
                
                fig.update_layout(
                    title="今後24時間の電力需要予測",
                    xaxis_title="時刻",
                    yaxis_title="予測需要 (MW)",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)

def show_model_comparison(model_results):
    """モデル比較分析モード"""
    st.header("📈 モデル比較分析")
    
    # 性能比較表
    st.subheader("🏆 モデル性能比較")
    
    # 技術進歩順に表示するための順序定義
    model_order = ['LinearRegression', 'Prophet', 'RandomForest']
    model_display_names = {
        'LinearRegression': '線形回帰（従来手法）',
        'Prophet': 'Prophet（時系列AI）',
        'RandomForest': 'Random Forest（高性能ML）'
    }
    
    comparison_data = []
    # 定義した順序でモデルを処理
    for model_name in model_order:
        if model_name in model_results:
            metrics = model_results[model_name]
            if isinstance(metrics, dict) and 'MAPE' in metrics:
                display_name = model_display_names.get(model_name, model_name)
                comparison_data.append({
                    'モデル': display_name,
                    'MAPE (%)': f"{metrics['MAPE']:.2f}",
                    'R²': f"{metrics['R2']:.3f}",
                    'RMSE': f"{metrics['RMSE']:.0f}",
                    'MAE': f"{metrics['MAE']:.0f}"
                })
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        
        # スタイル適用
        def highlight_best(s):
            if s.name == 'MAPE (%)':
                min_val = min([float(x) for x in s])
                return ['background-color: lightgreen' if float(x) == min_val else '' for x in s]
            elif s.name in ['R²']:
                max_val = max([float(x) for x in s])
                return ['background-color: lightgreen' if float(x) == max_val else '' for x in s]
            else:
                return ['' for _ in s]
        
        styled_df = df_comparison.style.apply(highlight_best)
        st.dataframe(styled_df, use_container_width=True)
        
        # 性能可視化
        col1, col2 = st.columns(2)
        
        with col1:
            # MAPE比較
            mape_values = [float(row['MAPE (%)']) for row in comparison_data]
            model_names = [row['モデル'] for row in comparison_data]
            
            fig = go.Figure(data=[
                go.Bar(x=model_names, y=mape_values, name='MAPE (%)', marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ])
            fig.update_layout(
                title="予測精度比較（MAPE）",
                yaxis_title="MAPE (%)",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # R²比較
            r2_values = [float(row['R²']) for row in comparison_data]
            
            fig = go.Figure(data=[
                go.Bar(x=model_names, y=r2_values, name='R²', marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ])
            fig.update_layout(
                title="説明力比較（R²）",
                yaxis_title="R²",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # 特徴量重要度（Random Forestの場合）
    if 'RandomForest' in model_results and 'feature_importance' in model_results['RandomForest']:
        st.subheader("🔍 特徴量重要度分析 (Random Forest)")
        
        importance = model_results['RandomForest']['feature_importance']
        
        # 重要度でソート
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        feature_names = [item[0] for item in sorted_features]
        importance_values = [item[1] for item in sorted_features]
        
        fig = go.Figure(data=[
            go.Bar(x=importance_values, y=feature_names, orientation='h')
        ])
        fig.update_layout(
            title="特徴量重要度",
            xaxis_title="重要度",
            yaxis_title="特徴量"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # 推奨アクション
    st.subheader("💡 推奨アクション")
    if comparison_data:
        # 従来手法とAI手法の比較
        linear_data = next((row for row in comparison_data if '従来手法' in row['モデル']), None)
        best_ai_data = min([row for row in comparison_data if '従来手法' not in row['モデル']], 
                          key=lambda x: float(x['MAPE (%)']))
        
        if linear_data and best_ai_data:
            linear_mape = float(linear_data['MAPE (%)'])
            best_ai_mape = float(best_ai_data['MAPE (%)'])
            improvement = ((linear_mape - best_ai_mape) / linear_mape) * 100 if linear_mape > 0 else 0
            
            st.success(f"✅ **{best_ai_data['モデル']}** が最も高い予測精度を示しています。")
            st.info(f"📈 従来手法と比較して **{improvement:.1f}%** の精度改善が期待できます。")
            
            # 具体的な導入メリット
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("予測精度改善", f"{improvement:.1f}%", "↑")
            with col2:
                st.metric("最高精度", f"MAPE {best_ai_mape:.2f}%", "↓")
            with col3:
                if linear_mape > 0:
                    cost_reduction = improvement * 0.5  # 簡易計算
                    st.metric("期待削減率", f"約{cost_reduction:.0f}%", "コスト")

def calculate_npv(annual_cash_flow, initial_investment, years, discount_rate):
    """NPV計算"""
    npv = -initial_investment
    for year in range(1, years + 1):
        npv += annual_cash_flow / (1 + discount_rate) ** year
    return npv

# フッター
def show_footer():
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📋 機能概要")
        st.markdown("""
        - 3種類のAIモデル比較
        - リアルタイム予測
        - コスト削減シミュレーション
        - 投資対効果分析
        """)
    
    with col2:
        st.markdown("### 🎯 予測精度")
        st.markdown("""
        - MAPE: 3-5% (目標)
        - 複数モデルのアンサンブル
        - 信頼区間付き予測
        - 継続的な学習更新
        """)
    
    with col3:
        st.markdown("### 💼 ビジネス価値")
        st.markdown("""
        - 電力調達コストの最適化を実現
        - 計画的な投資回収
        - 予測精度の継続改善
        - スケーラブルなシステム
        """)
    
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: gray;">⚡ 電力需給予測システム - AI実証デモ | '
        '🤖 Powered by Prophet + scikit-learn</p>',
        unsafe_allow_html=True
    )

# メイン実行
if __name__ == "__main__":
    main()
    show_footer()
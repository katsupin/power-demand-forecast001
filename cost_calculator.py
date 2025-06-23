"""
コスト削減計算機能
電力需給予測システムの投資対効果計算
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PowerCostCalculator:
    """電力コスト削減計算クラス"""
    
    def __init__(self):
        # デフォルト値
        self.default_params = {
            'annual_demand_gwh': 10000,
            'current_error_rate': 0.08,
            'ai_error_rate': 0.04,
            'imbalance_cost_per_kwh': 15.0,
            'spot_price_per_kwh': 12.0,
            'implementation_cost_million': 100,
            'annual_operation_cost_million': 20,
            'discount_rate': 0.05,
            'analysis_years': 10
        }
    
    def calculate_cost_reduction(self, params: Dict) -> Dict:
        """
        コスト削減効果の計算
        
        Args:
            params: 計算パラメータ
            
        Returns:
            計算結果辞書
        """
        # パラメータ取得
        annual_demand = params.get('annual_demand_gwh', self.default_params['annual_demand_gwh'])
        current_error = params.get('current_error_rate', self.default_params['current_error_rate'])
        ai_error = params.get('ai_error_rate', self.default_params['ai_error_rate'])
        unit_cost = params.get('imbalance_cost_per_kwh', self.default_params['imbalance_cost_per_kwh'])
        impl_cost = params.get('implementation_cost_million', self.default_params['implementation_cost_million'])
        op_cost = params.get('annual_operation_cost_million', self.default_params['annual_operation_cost_million'])
        
        # 基本計算
        annual_demand_kwh = annual_demand * 1_000_000_000  # GWh to kWh
        
        # 現状のコスト
        current_error_kwh = annual_demand_kwh * current_error
        current_cost_million = (current_error_kwh * unit_cost) / 1_000_000
        
        # AI導入後のコスト
        ai_error_kwh = annual_demand_kwh * ai_error
        ai_cost_million = (ai_error_kwh * unit_cost) / 1_000_000
        
        # 削減効果
        gross_saving = current_cost_million - ai_cost_million
        net_saving = gross_saving - op_cost
        
        # 投資回収期間
        if net_saving > 0:
            payback_years = impl_cost / net_saving
        else:
            payback_years = float('inf')
        
        # 改善率
        error_improvement = ((current_error - ai_error) / current_error) * 100
        cost_improvement = (gross_saving / current_cost_million) * 100
        
        return {
            'current_cost_million': current_cost_million,
            'ai_cost_million': ai_cost_million,
            'gross_saving_million': gross_saving,
            'net_saving_million': net_saving,
            'payback_years': payback_years,
            'error_improvement_percent': error_improvement,
            'cost_improvement_percent': cost_improvement,
            'current_error_kwh': current_error_kwh,
            'ai_error_kwh': ai_error_kwh,
            'error_reduction_kwh': current_error_kwh - ai_error_kwh
        }
    
    def calculate_npv(self, params: Dict) -> Dict:
        """
        NPV（正味現在価値）計算
        
        Args:
            params: 計算パラメータ
            
        Returns:
            NPV関連の計算結果
        """
        basic_results = self.calculate_cost_reduction(params)
        
        impl_cost = params.get('implementation_cost_million', self.default_params['implementation_cost_million'])
        annual_saving = basic_results['net_saving_million']
        discount_rate = params.get('discount_rate', self.default_params['discount_rate'])
        years = params.get('analysis_years', self.default_params['analysis_years'])
        
        # NPV計算
        npv = -impl_cost  # 初期投資（マイナス）
        
        for year in range(1, years + 1):
            discounted_saving = annual_saving / ((1 + discount_rate) ** year)
            npv += discounted_saving
        
        # IRR計算（近似）
        irr = self._calculate_irr(impl_cost, annual_saving, years)
        
        # その他の指標
        total_undiscounted_saving = annual_saving * years
        roi_percent = ((total_undiscounted_saving - impl_cost) / impl_cost) * 100 if impl_cost > 0 else 0
        
        return {
            'npv_million': npv,
            'irr_percent': irr,
            'total_saving_million': total_undiscounted_saving,
            'roi_percent': roi_percent
        }
    
    def _calculate_irr(self, initial_investment: float, annual_cash_flow: float, years: int) -> float:
        """IRR（内部収益率）の近似計算"""
        if annual_cash_flow <= 0:
            return 0
        
        # 単純な近似式
        irr = (annual_cash_flow / initial_investment) * 100
        return min(irr, 999.9)  # 上限設定
    
    def create_cumulative_chart(self, params: Dict) -> go.Figure:
        """
        累積効果チャートの作成
        
        Args:
            params: 計算パラメータ
            
        Returns:
            Plotlyチャート
        """
        basic_results = self.calculate_cost_reduction(params)
        
        impl_cost = params.get('implementation_cost_million', self.default_params['implementation_cost_million'])
        annual_saving = basic_results['net_saving_million']
        years = params.get('analysis_years', self.default_params['analysis_years'])
        
        # 累積効果計算
        year_range = np.arange(0, years + 1)
        cumulative_effect = [-impl_cost + annual_saving * year for year in year_range]
        
        # チャート作成
        fig = go.Figure()
        
        # 累積効果線
        fig.add_trace(go.Scatter(
            x=year_range,
            y=cumulative_effect,
            mode='lines+markers',
            name='累積効果',
            line=dict(width=3, color='blue'),
            fill='tonexty'
        ))
        
        # ゼロライン
        fig.add_hline(
            y=0, 
            line_dash="dash", 
            line_color="red",
            annotation_text="損益分岐点",
            annotation_position="top left"
        )
        
        # 投資回収ポイント
        payback = basic_results['payback_years']
        if payback < years:
            fig.add_vline(
                x=payback,
                line_dash="dash",
                line_color="green",
                annotation_text=f"投資回収: {payback:.1f}年",
                annotation_position="top"
            )
        
        # レイアウト設定
        fig.update_layout(
            title="投資対効果の推移",
            xaxis_title="導入後の年数",
            yaxis_title="累積効果（百万円）",
            hovermode='x unified',
            showlegend=True,
            height=400
        )
        
        return fig
    
    def create_sensitivity_analysis(self, base_params: Dict, 
                                  sensitivity_param: str, 
                                  range_percent: float = 0.3) -> go.Figure:
        """
        感度分析チャートの作成
        
        Args:
            base_params: ベースパラメータ
            sensitivity_param: 感度分析対象パラメータ
            range_percent: 変動幅（±30%など）
            
        Returns:
            感度分析チャート
        """
        base_value = base_params[sensitivity_param]
        
        # 変動範囲設定
        min_value = base_value * (1 - range_percent)
        max_value = base_value * (1 + range_percent)
        test_values = np.linspace(min_value, max_value, 11)
        
        results = []
        for value in test_values:
            test_params = base_params.copy()
            test_params[sensitivity_param] = value
            
            calc_result = self.calculate_cost_reduction(test_params)
            results.append({
                'parameter_value': value,
                'net_saving': calc_result['net_saving_million'],
                'payback_years': min(calc_result['payback_years'], 10)  # 上限設定
            })
        
        results_df = pd.DataFrame(results)
        
        # チャート作成
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('年間削減額への影響', '投資回収期間への影響'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 年間削減額
        fig.add_trace(
            go.Scatter(
                x=results_df['parameter_value'],
                y=results_df['net_saving'],
                mode='lines+markers',
                name='年間削減額',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # 投資回収期間
        fig.add_trace(
            go.Scatter(
                x=results_df['parameter_value'],
                y=results_df['payback_years'],
                mode='lines+markers',
                name='投資回収期間',
                line=dict(color='red')
            ),
            row=1, col=2
        )
        
        # ベース値の線
        fig.add_vline(
            x=base_value,
            line_dash="dash",
            line_color="gray",
            annotation_text="ベース値",
            row=1, col=1
        )
        fig.add_vline(
            x=base_value,
            line_dash="dash",
            line_color="gray",
            annotation_text="ベース値",
            row=1, col=2
        )
        
        fig.update_layout(
            title=f"{sensitivity_param} の感度分析",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def generate_report(self, params: Dict) -> str:
        """
        レポート生成
        
        Args:
            params: 計算パラメータ
            
        Returns:
            レポート文字列
        """
        basic_results = self.calculate_cost_reduction(params)
        npv_results = self.calculate_npv(params)
        
        report = f"""
# 電力需給予測システム - 投資対効果分析レポート

## 入力パラメータ
- 年間電力需要: {params.get('annual_demand_gwh', 'N/A'):,} GWh
- 現在の予測誤差率: {params.get('current_error_rate', 0) * 100:.1f}%
- AI導入後誤差率: {params.get('ai_error_rate', 0) * 100:.1f}%
- インバランス単価: {params.get('imbalance_cost_per_kwh', 'N/A')} 円/kWh
- 初期投資: {params.get('implementation_cost_million', 'N/A')} 百万円
- 年間運用コスト: {params.get('annual_operation_cost_million', 'N/A')} 百万円

## 計算結果

### 基本指標
- 現在の年間誤差コスト: {basic_results['current_cost_million']:.0f} 百万円
- AI導入後の年間誤差コスト: {basic_results['ai_cost_million']:.0f} 百万円
- 年間総削減額: {basic_results['gross_saving_million']:.0f} 百万円
- 年間純削減額: {basic_results['net_saving_million']:.0f} 百万円
- 投資回収期間: {basic_results['payback_years']:.1f} 年

### 改善効果
- 予測誤差改善率: {basic_results['error_improvement_percent']:.1f}%
- コスト削減率: {basic_results['cost_improvement_percent']:.1f}%
- 誤差削減量: {basic_results['error_reduction_kwh']/1_000_000:.0f} MWh

### 財務指標
- NPV（{params.get('analysis_years', 10)}年）: {npv_results['npv_million']:.0f} 百万円
- IRR: {npv_results['irr_percent']:.1f}%
- ROI: {npv_results['roi_percent']:.1f}%
- 総削減額（期間内）: {npv_results['total_saving_million']:.0f} 百万円

## 結論
{'非常に有望な投資案件です。' if basic_results['payback_years'] < 2 else '検討価値のある投資案件です。' if basic_results['payback_years'] < 5 else '慎重な検討が必要です。'}
投資回収期間が{basic_results['payback_years']:.1f}年と短く、{params.get('analysis_years', 10)}年間で{npv_results['npv_million']:.0f}百万円の正味現在価値を創出します。
"""
        return report

# 使用例
if __name__ == "__main__":
    calculator = PowerCostCalculator()
    
    # テスト計算
    test_params = {
        'annual_demand_gwh': 10000,
        'current_error_rate': 0.08,
        'ai_error_rate': 0.04,
        'imbalance_cost_per_kwh': 15.0,
        'implementation_cost_million': 100,
        'annual_operation_cost_million': 20
    }
    
    # 基本計算
    results = calculator.calculate_cost_reduction(test_params)
    print("基本計算結果:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    # NPV計算
    npv_results = calculator.calculate_npv(test_params)
    print("\nNPV計算結果:")
    for key, value in npv_results.items():
        print(f"  {key}: {value}")
    
    # レポート生成
    report = calculator.generate_report(test_params)
    print("\n" + "="*50)
    print(report)
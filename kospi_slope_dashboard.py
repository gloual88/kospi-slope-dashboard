"""
KOSPI 기울기 분석 대시보드
- 고객이 임의로 분석 구간 선택 가능
- 현재 시점까지 데이터 자동 업데이트
- 역사적 기록과 비교 분석
- 기울기 설명 탭 추가 (중학생 버전)
- Peter님 제공 코드 기반
"""

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import FinanceDataReader as fdr
import pytz

# ============ 페이지 설정 ============
st.set_page_config(
    page_title="KOSPI 기울기 분석",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ 역사적 기록 데이터 ============
HISTORICAL_RECORDS = [
    (1, "2020-2021 COVID", "3구간", 84.17, 33.67, 0.898, 13.44, 12, "연말 기관 매수세"),
    (2, "2025년 10월", "2구간", 83.38, 38.91, 0.963, 15.44, 14, "물량성 상승세"),
    (3, "2025년 6월", "1구간", 90.68, 18.36, 0.920, 15.50, 28, "상승 트렌드 복귀"),
    (4, "2025-2026", "3구간", 81.42, 27.85, 0.951, 15.26, 13, "12월~1월 강세장"),
    (5, "2020-2021 COVID", "2구간", 17.43, 15.68, 0.955, 21.10, 27, "백신 개발 기대감"),
    (6, "1998-1999 기술주붐", "3구간", 7.61, 8.12, 0.900, 41.46, 32, "IT 버블 초기"),
    (7, "2009-2011 금융위기", "3구간", 6.29, 7.55, 0.914, 13.07, 36, "일드 헌팅 본격화"),
    (8, "2020-2021 COVID", "1구간", 5.22, 9.57, 0.890, 37.27, 55, "팬데믹 충격 후 V자 회복"),
    (9, "2009-2011 금융위기", "1구간", 4.91, 7.53, 0.928, 33.31, 46, "글로벌 금융위기 바닥 반등"),
    (10, "1998-1999 기술주붐", "2구간", 4.80, 5.76, 0.931, 53.58, 36, "기저에서 기대감"),
    (11, "1998-1999 기술주붐", "1구간", 1.59, 3.44, 0.809, 71.50, 65, "IMF 위기 직격탄 이후 반등"),
    (12, "2009-2011 금융위기", "2구간", 0.64, 2.37, 0.888, 19.48, 112, "QE2 기대감 상승"),
]

def get_historical_df():
    return pd.DataFrame(HISTORICAL_RECORDS, columns=[
        '원래순위', '시대', '구간', '정규화기울기', '원본기울기', '신뢰도', '수익률', '기간(일)', '설명'
    ])


# ============ 데이터 로드 함수 ============
@st.cache_data(ttl=60)
def load_kospi_data():
    kst = pytz.timezone('Asia/Seoul')
    end = datetime.now(kst).strftime('%Y-%m-%d')
    start = "2020-01-01"
    
    df = fdr.DataReader('KS11', start, end)
    df = df.reset_index()
    df = df[['Date', 'Open', 'High', 'Low', 'Close']].copy()
    
    for col in ['Open', 'High', 'Low']:
        df[col] = df[col].fillna(df['Close'])
        df.loc[df[col] == 0.0, col] = df.loc[df[col] == 0.0, 'Close']
    
    return df


# ============ 분석 함수 ============
def analyze_period_slope(df, start_date, end_date, period_name):
    df_work = df.copy()
    
    if 'Date' not in df_work.columns and '날짜' in df_work.columns:
        df_work['Date'] = pd.to_datetime(df_work['날짜'])
    elif 'Date' in df_work.columns:
        df_work['Date'] = pd.to_datetime(df_work['Date'])
    else:
        df_work = df_work.reset_index()
        df_work['Date'] = pd.to_datetime(df_work['Date'])
    
    if 'Close' not in df_work.columns:
        if '종가' in df_work.columns:
            df_work['Close'] = df_work['종가']
        elif 'close' in df_work.columns:
            df_work['Close'] = df_work['close']
        else:
            raise ValueError(f"Close price 컬럼을 찾을 수 없습니다.")
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    period_mask = (df_work['Date'] >= start_date) & (df_work['Date'] <= end_date)
    period_data = df_work[period_mask].copy()
    
    if len(period_data) < 2:
        return None, None, None, None
    
    period_data['days'] = (period_data['Date'] - period_data['Date'].min()).dt.days
    
    X = period_data['days'].values.reshape(-1, 1)
    y = period_data['Close'].values
    
    lr_model = LinearRegression()
    lr_model.fit(X, y)
    
    regression_slope = lr_model.coef_[0]
    r_squared = lr_model.score(X, y)
    predicted_prices = lr_model.predict(X)
    
    pct_change = ((period_data['Close'].iloc[-1] / period_data['Close'].iloc[0]) - 1) * 100
    days_diff = period_data['days'].iloc[-1] - period_data['days'].iloc[0]
    daily_avg_pct = pct_change / days_diff if days_diff != 0 else 0
    
    trading_days = len(period_data)
    normalized_slope = (regression_slope / period_data['Close'].iloc[0]) * 1000
    
    return period_data, lr_model, predicted_prices, {
        'slope': regression_slope,
        'normalized_slope': normalized_slope,
        'r_squared': r_squared,
        'pct_change': pct_change,
        'daily_avg_pct': daily_avg_pct,
        'period_name': period_name,
        'start_price': period_data['Close'].iloc[0],
        'end_price': period_data['Close'].iloc[-1],
        'start_date': period_data['Date'].iloc[0],
        'end_date': period_data['Date'].iloc[-1],
        'trading_days': trading_days
    }


# ============ 시각화 함수 ============
def create_period_visualization(period_data, lr_model, predicted_prices, stats, chart_title):
    slope = stats['slope']
    if abs(slope) > 20:
        strength = "매우 강한"
        color = "#FF0000" if slope > 0 else "#0000FF"
    elif abs(slope) > 10:
        strength = "강한" 
        color = "#FF4500" if slope > 0 else "#4169E1"
    elif abs(slope) > 5:
        strength = "보통"
        color = "#FFA500" if slope > 0 else "#6495ED"
    else:
        strength = "약한"
        color = "#32CD32" if slope > 0 else "#808080"
    
    direction = "상승" if slope > 0 else "하락"
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.8, 0.2],
        subplot_titles=[chart_title, '일별 변화율'],
        vertical_spacing=0.1
    )
    
    fig.add_trace(go.Scatter(
        x=period_data['Date'],
        y=period_data['Close'],
        mode='lines+markers',
        name='KOSPI',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6, color='#1f77b4'),
        hovertemplate='날짜: %{x}<br>종가: %{y:,.0f}<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=period_data['Date'],
        y=predicted_prices,
        mode='lines',
        name=f'추세선 (기울기: {slope:.1f}p/일)',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        hovertemplate='날짜: %{x}<br>추세가: %{y:,.0f}<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=[period_data['Date'].iloc[0], period_data['Date'].iloc[-1]],
        y=[period_data['Close'].iloc[0], period_data['Close'].iloc[-1]],
        mode='markers',
        name='시작/종료점',
        marker=dict(size=[12, 12], color=['green', 'red'], symbol=['circle', 'diamond']),
        showlegend=False
    ), row=1, col=1)
    
    daily_returns = period_data['Close'].pct_change().fillna(0) * 100
    colors_bar = ['red' if x < 0 else 'green' for x in daily_returns]
    
    fig.add_trace(go.Bar(
        x=period_data['Date'],
        y=daily_returns,
        name='일별 변화율(%)',
        marker_color=colors_bar,
        opacity=0.7,
        hovertemplate='날짜: %{x}<br>변화율: %{y:.2f}%<extra></extra>'
    ), row=2, col=1)
    
    data_min = period_data['Close'].min()
    data_max = period_data['Close'].max()
    data_range = data_max - data_min
    
    text_y_position = data_max + data_range * 0.15
    mid_date = period_data['Date'].iloc[len(period_data)//2]
    
    analysis_text = f"""
    <b>📊 분석 결과</b><br>
    <b>기간:</b> {period_data['Date'].iloc[0].strftime('%Y-%m-%d')} ~ {period_data['Date'].iloc[-1].strftime('%Y-%m-%d')} ({len(period_data)}거래일)<br>
    <b>변화:</b> {period_data['Close'].iloc[0]:,.0f} → {period_data['Close'].iloc[-1]:,.0f} (<span style="color:{color}"><b>{stats['pct_change']:+.2f}%</b></span>)<br>
    <b>기울기:</b> <span style="color:{color}"><b>{slope:.2f}</b></span> 포인트/일<br>
    <b>신뢰도:</b> R² = {stats['r_squared']:.3f} ({stats['r_squared']:.1%})<br>
    <b>추세:</b> <span style="color:{color}"><b>{strength} {direction}세</b></span>
    """
    
    fig.add_annotation(
        x=mid_date,
        y=text_y_position,
        text=analysis_text,
        showarrow=False,
        font=dict(size=12, color="black"),
        bgcolor="rgba(255, 255, 255, 0.95)",
        bordercolor="gray",
        borderwidth=1,
        borderpad=10,
        align="left",
        xanchor="center",
        yanchor="bottom"
    )
    
    fig.update_layout(
        title=dict(text=f'<b>{chart_title}</b>', x=0.5, xanchor='center', font=dict(size=18)),
        height=650,
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    y_range_bottom = data_min - data_range * 0.05
    y_range_top = text_y_position + data_range * 0.25
    
    fig.update_yaxes(range=[y_range_bottom, y_range_top], row=1, col=1)
    fig.update_yaxes(title_text="KOSPI 지수", row=1, col=1)
    fig.update_yaxes(title_text="변화율(%)", row=2, col=1)
    fig.update_xaxes(title_text="날짜", row=2, col=1)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig


# ============ 역사적 비교 함수 ============
def create_historical_comparison_chart(hist_df, current_stats, metric, metric_label, ascending=False):
    if metric == '원본기울기':
        current_value = current_stats['slope']
    elif metric == '정규화기울기':
        current_value = current_stats['normalized_slope']
    elif metric == '신뢰도':
        current_value = current_stats['r_squared']
    elif metric == '수익률':
        current_value = current_stats['pct_change']
    elif metric == '기간(일)':
        current_value = current_stats['trading_days']
    else:
        current_value = current_stats['slope']
    
    hist_values = hist_df[metric].tolist()
    hist_labels = [f"{row['시대']} {row['구간']}" for _, row in hist_df.iterrows()]
    
    all_values = hist_values + [current_value]
    all_labels = hist_labels + [f"🔴 현재 분석\n({current_stats['period_name']})"]
    all_descriptions = hist_df['설명'].tolist() + [current_stats['period_name']]
    
    sorted_data = sorted(zip(all_values, all_labels, all_descriptions), 
                        key=lambda x: x[0], reverse=not ascending)
    
    sorted_values = [x[0] for x in sorted_data]
    sorted_labels = [x[1] for x in sorted_data]
    sorted_descriptions = [x[2] for x in sorted_data]
    
    current_rank = sorted_labels.index(f"🔴 현재 분석\n({current_stats['period_name']})") + 1
    
    colors = ['#FF4136' if '현재 분석' in label else '#1f77b4' for label in sorted_labels]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=sorted_labels,
        x=sorted_values,
        orientation='h',
        marker_color=colors,
        text=[f"{v:.2f}" for v in sorted_values],
        textposition='outside',
        hovertemplate='%{y}<br>' + metric_label + ': %{x:.2f}<br>설명: %{customdata}<extra></extra>',
        customdata=sorted_descriptions
    ))
    
    fig.update_layout(
        title=dict(
            text=f'<b>{metric_label} 기준 역사적 순위</b><br><sub>🔴 현재 분석: {current_rank}위 / {len(all_values)}개</sub>',
            x=0.5,
            xanchor='center'
        ),
        xaxis_title=metric_label,
        yaxis_title="",
        height=max(400, len(all_values) * 35),
        showlegend=False,
        plot_bgcolor='white',
        margin=dict(l=200)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig, current_rank, len(all_values)


def create_radar_chart(current_stats, hist_df):
    categories = ['원본기울기', '정규화기울기', '신뢰도', '수익률']
    
    current_values = []
    hist_avg_values = []
    
    for cat in categories:
        if cat == '원본기울기':
            curr = current_stats['slope']
        elif cat == '정규화기울기':
            curr = current_stats['normalized_slope']
        elif cat == '신뢰도':
            curr = current_stats['r_squared']
        elif cat == '수익률':
            curr = current_stats['pct_change']
        
        hist_max = hist_df[cat].max()
        hist_min = hist_df[cat].min()
        hist_avg = hist_df[cat].mean()
        
        if hist_max != hist_min:
            curr_normalized = ((curr - hist_min) / (hist_max - hist_min)) * 100
            avg_normalized = ((hist_avg - hist_min) / (hist_max - hist_min)) * 100
        else:
            curr_normalized = 50
            avg_normalized = 50
        
        current_values.append(max(0, min(100, curr_normalized)))
        hist_avg_values.append(max(0, min(100, avg_normalized)))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=hist_avg_values + [hist_avg_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.2)',
        line=dict(color='#1f77b4', width=2),
        name='역사적 평균'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=current_values + [current_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(255, 65, 54, 0.3)',
        line=dict(color='#FF4136', width=3),
        name='현재 분석'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title=dict(text='<b>역사적 평균 대비 현재 분석</b>', x=0.5, xanchor='center'),
        height=450
    )
    
    return fig


# ============ 기울기 설명 탭 (중학생 버전) ============
def render_slope_explanation_tab():
    """기울기 설명 탭 - 중학생도 이해할 수 있는 쉬운 설명"""
    
    st.header("🌡️ 숫자로 보는 시장 온도계")
    st.subheader("핵심 지표 완전 해부 (중학생 버전)")
    
    st.info("""
    ⚠️ **먼저 한 가지 약속부터요.**  
    여기서 말하는 "급등 구간(엄청 빨리 오른 구간)"은 제가 임의로 정한 겁니다.  
    "정확한 저점·고점"은 사람마다 다르게 잡을 수 있으니, 그 부분은 **분석 기준의 차이**로 이해해주면 됩니다.
    """)
    
    st.markdown("---")
    
    # ============ 1) 상승 속도 (정규화 기울기) ============
    st.markdown("## 1️⃣ 상승 속도 (정규화 기울기)")
    
    st.success('💡 **한마디로:** "서로 다른 시장을 **공정하게 비교**하는 속도계"')
    
    st.markdown("""
    #### 📌 예를 들어볼게요
    
    | 시대 | KOSPI 수준 |
    |------|-----------|
    | 1998년 | 약 500 |
    | 2020년 | 약 2,500 |
    
    이 둘을 그냥 **"포인트가 몇 올랐냐"**로 비교하면 **불공정**합니다.  
    가격 수준이 다르니까요.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background-color: #fff3e0; padding: 20px; border-radius: 10px;">
            <h4>🏠 비유하자면...</h4>
            <p>서울 아파트가 <b>1억 오르는 것</b>과<br>
            지방 아파트가 <b>1억 오르는 것</b>은<br>
            느낌이 완전 다르죠?</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #e3f2fd; padding: 20px; border-radius: 10px;">
            <h4>📊 그래서 이렇게 해요</h4>
            <p><b>시작 값을 100으로 맞춰놓고</b> (표준화)<br>
            같은 기간 동안 얼마나 빨리 올랐는지 계산합니다.</p>
            <p><b>이 값이 클수록</b> → 시장이 더 빠른 속도로 급등!</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============ 2) 하루 평균 (원본 기울기) ============
    st.markdown("## 2️⃣ 하루 평균 (원본 기울기)")
    
    st.success('💡 **한마디로:** "내 계좌가 **매일 얼마나 늘어나는 느낌**인지 보여주는 숫자"')
    
    st.markdown("""
    원본 기울기는 **아주 현실적인 지표**예요.  
    KOSPI가 **하루에 평균 몇 포인트씩 올랐는지**를 그대로 봅니다.
    """)
    
    st.markdown("""
    <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px; margin: 20px 0;">
        <h4>📈 예를 들어 "하루 평균 39포인트 상승"이라면?</h4>
        <ul>
            <li>진짜로 매일 평균 <b>39포인트씩</b> 오른 것에 가깝다는 뜻이고</li>
            <li>투자자는 <b>"요즘 시장 진짜 미쳤다!"</b> (좋은 의미든 나쁜 의미든)라고 체감하기 쉬워요</li>
        </ul>
        <p style="margin-top: 15px; font-size: 18px;">
            👉 즉, 원본 기울기는 <b>사람들이 몸으로 느끼는 상승 체감</b>을 보여줍니다.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============ 3) 직선도 (R², 신뢰도) ============
    st.markdown("## 3️⃣ 직선도 (R², 신뢰도)")
    
    st.success('💡 **한마디로:** "그래프가 얼마나 **거의 직선처럼** 올라갔는지"')
    
    st.markdown("""
    R²는 **0~1 사이 숫자**고, **1에 가까울수록** 그래프가 거의 **일직선**으로 쭉 상승했다는 뜻입니다.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background-color: #ffebee; padding: 20px; border-radius: 10px;">
            <h4>⚠️ 직선도가 너무 높다면?</h4>
            <p>"거의 매일 오르고, 중간에 쉬는 날(조정)이 별로 없다"</p>
            <p>겉보기엔 멋져 보이지만...<br>
            오히려 <b>위험 신호</b>가 될 수 있어요!</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px;">
            <h4>✅ 정상적인 시장은?</h4>
            <p>오르는 중간중간<br>
            <b>"잠깐 쉬었다가(조정) 다시 오르는"</b><br>
            과정이 반복됩니다.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.warning("""
    🔥 **직선도가 지나치게 높다는 건?**
    - 팔 사람이 거의 없고 (매도 압력 약함)
    - 다들 **"더 오를 거야!"** 하면서 달려드는 분위기
    
    → 이런 때가 종종 **과열 구간**으로 해석됩니다.
    """)
    
    st.markdown("---")
    
    # ============ 4) 같은 수익률인데 더 빨리? ============
    st.markdown("## 4️⃣ 같은 15%인데, 더 빨리 벌었다면?")
    
    st.success('💡 **한마디로:** "같은 수익률이라도 **너무 빨리 나오면 과열**일 수 있다"')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background-color: #e3f2fd; padding: 25px; border-radius: 10px; text-align: center;">
            <h2>🐢</h2>
            <h4>15% 수익</h4>
            <p style="font-size: 24px; font-weight: bold;">28일 걸림</p>
            <p style="color: green;">→ 비교적 안전한 상승</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #ffebee; padding: 25px; border-radius: 10px; text-align: center;">
            <h2>🚀</h2>
            <h4>15% 수익</h4>
            <p style="font-size: 24px; font-weight: bold;">14일 걸림</p>
            <p style="color: red;">→ ⚠️ 과열 가능성!</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    #### 🤔 왜 14일짜리가 더 위험할까요?
    
    너무 짧은 시간에 급등하면 사람들의 심리가 이렇게 변해요:
    """)
    
    st.markdown("""
    <div style="background-color: #fff8e1; padding: 20px; border-radius: 10px; margin: 20px 0;">
        <p style="font-size: 20px;">😰 "나만 못 벌면 어떡하지?"</p>
        <p style="font-size: 20px;">😱 "지금 안 사면 영영 못 사!"</p>
        <p style="margin-top: 15px;"><b>👉 FOMO</b> (Fear Of Missing Out): 놓칠까 봐 두려움</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.error("""
    💥 **결론:**  
    이런 분위기가 강해지면 시장이 **과열**되고,  
    어느 순간 **작은 악재에도 크게 흔들릴 수 있습니다.**
    """)
    
    st.markdown("---")
    
    # ============ 요약 테이블 ============
    st.markdown("## 📋 한눈에 정리")
    
    summary_df = pd.DataFrame({
        "지표": ["🚀 상승 속도", "📊 하루 평균", "📏 직선도"],
        "원래 이름": ["정규화 기울기", "원본 기울기", "R² (신뢰도)"],
        "의미": [
            "시대가 달라도 공정 비교 가능한 속도",
            "매일 체감하는 상승폭",
            "얼마나 일직선으로 올랐나"
        ],
        "높으면?": [
            "역사적으로 빠른 급등",
            "피부로 느껴지는 강한 상승",
            "⚠️ 쉬지 않고 올라서 과열 가능"
        ]
    })
    
    st.table(summary_df)
    
    st.success("""
    ✅ **이제 이해되셨나요?**  
    👉 **'기울기 분석'** 탭에서 직접 기간을 선택해 분석해보세요!
    """)


# ============ 메인 대시보드 ============
def main():
    st.title("📈 KOSPI 기울기 분석 대시보드")
    
    # 탭 구성 (설명 → 분석 → 비교)
    tab1, tab2, tab3 = st.tabs(["📖 기울기 설명", "📊 기울기 분석", "🏆 역사적 비교"])
    
    # 데이터 로드
    with st.spinner("📥 KOSPI 데이터 로딩 중..."):
        df = load_kospi_data()
    
    kst = pytz.timezone('Asia/Seoul')
    current_time = datetime.now(kst).strftime('%Y-%m-%d %H:%M:%S KST')
    latest_date = df['Date'].max().strftime('%Y-%m-%d')
    latest_close = df['Close'].iloc[-1]
    
    st.info(f"🕐 현재 시간: {current_time} | 📊 최신 데이터: {latest_date} | 💹 KOSPI: {latest_close:,.2f}")
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 분석 기간 설정")
        
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        st.caption(f"📅 데이터 범위: {min_date} ~ {max_date}")
        
        st.markdown("---")
        
        st.subheader("📌 분석 기간")
        col1, col2 = st.columns(2)
        with col1:
            start_date_1 = st.date_input("시작일", value=max_date - timedelta(days=30),
                                         min_value=min_date, max_value=max_date, key="start_1")
        with col2:
            end_date_1 = st.date_input("종료일", value=max_date,
                                       min_value=min_date, max_value=max_date, key="end_1")
        period_name_1 = st.text_input("기간 이름", value="현재 분석 기간", key="name_1")
        
        st.markdown("---")
        
        compare_enabled = st.checkbox("📊 추가 기간 비교", value=False)
        
        if compare_enabled:
            st.subheader("📌 비교 기간")
            col3, col4 = st.columns(2)
            with col3:
                start_date_2 = st.date_input("시작일", value=max_date - timedelta(days=90),
                                             min_value=min_date, max_value=max_date, key="start_2")
            with col4:
                end_date_2 = st.date_input("종료일", value=max_date - timedelta(days=60),
                                           min_value=min_date, max_value=max_date, key="end_2")
            period_name_2 = st.text_input("기간 이름", value="비교 기간", key="name_2")
        
        st.markdown("---")
        
        analyze_button = st.button("🔍 분석 실행", type="primary", use_container_width=True)
        
        if st.button("🔄 데이터 새로고침", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # TAB 1: 기울기 설명
    with tab1:
        render_slope_explanation_tab()
    
    # TAB 2: 기울기 분석
    with tab2:
        if analyze_button:
            st.markdown("---")
            st.header(f"📊 {period_name_1}")
            
            if start_date_1 >= end_date_1:
                st.error("❌ 시작일이 종료일보다 같거나 늦습니다.")
            else:
                period1_data, period1_model, period1_predicted, period1_stats = analyze_period_slope(
                    df, start_date_1, end_date_1, period_name_1
                )
                
                if period1_data is not None:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(label="📈 총 수익률", value=f"{period1_stats['pct_change']:+.2f}%",
                                  delta=f"{period1_stats['trading_days']}거래일",
                                  help="선택한 기간 동안 KOSPI가 얼마나 올랐는지")
                    with col2:
                        st.metric(label="📊 하루 평균", value=f"{period1_stats['slope']:+.2f}p",
                                  help="하루 평균 몇 포인트씩 변했는지 (체감 상승폭)")
                    with col3:
                        st.metric(label="📏 직선도", value=f"{period1_stats['r_squared']:.1%}",
                                  help="얼마나 꾸준히 한 방향으로 움직였는지 (너무 높으면 과열 가능)")
                    with col4:
                        st.metric(label="🚀 상승 속도", value=f"{period1_stats['normalized_slope']:.1f}점",
                                  help="다른 시대와 비교 가능한 점수 (높을수록 빠른 급등)")
                    
                    fig1 = create_period_visualization(
                        period1_data, period1_model, period1_predicted, period1_stats,
                        f'KOSPI 기울기 분석 - {period_name_1}'
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    st.session_state['current_stats'] = period1_stats
                else:
                    st.warning("⚠️ 선택한 기간에 충분한 데이터가 없습니다.")
            
            if compare_enabled:
                st.markdown("---")
                st.header(f"📊 {period_name_2}")
                
                if start_date_2 >= end_date_2:
                    st.error("❌ 시작일이 종료일보다 같거나 늦습니다.")
                else:
                    period2_data, period2_model, period2_predicted, period2_stats = analyze_period_slope(
                        df, start_date_2, end_date_2, period_name_2
                    )
                    
                    if period2_data is not None:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric(label="📈 총 수익률", value=f"{period2_stats['pct_change']:+.2f}%",
                                      delta=f"{period2_stats['trading_days']}거래일")
                        with col2:
                            st.metric(label="📊 하루 평균", value=f"{period2_stats['slope']:+.2f}p")
                        with col3:
                            st.metric(label="📏 직선도", value=f"{period2_stats['r_squared']:.1%}")
                        with col4:
                            st.metric(label="🚀 상승 속도", value=f"{period2_stats['normalized_slope']:.1f}점")
                        
                        fig2 = create_period_visualization(
                            period2_data, period2_model, period2_predicted, period2_stats,
                            f'KOSPI 기울기 분석 - {period_name_2}'
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.warning("⚠️ 선택한 기간에 충분한 데이터가 없습니다.")
                
                if 'period1_stats' in dir() and period1_stats and 'period2_stats' in dir() and period2_stats:
                    st.markdown("---")
                    st.header("📋 기간별 비교 요약")
                    
                    comparison_df = pd.DataFrame([
                        {
                            "기간": s['period_name'],
                            "시작일": s['start_date'].strftime('%Y-%m-%d'),
                            "종료일": s['end_date'].strftime('%Y-%m-%d'),
                            "거래일": s['trading_days'],
                            "하루 평균": f"{s['slope']:+.2f}p",
                            "총 수익률": f"{s['pct_change']:+.2f}%",
                            "직선도": f"{s['r_squared']:.1%}"
                        }
                        for s in [period1_stats, period2_stats]
                    ])
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        else:
            st.markdown("---")
            st.markdown("""
            ### 📌 사용 방법
            1. **왼쪽 사이드바**에서 분석할 기간의 시작일과 종료일을 선택하세요.
            2. **"🔍 분석 실행"** 버튼을 클릭하면 결과가 표시됩니다.
            
            ### 💡 처음이신가요?
            👉 먼저 **"📖 기울기 설명"** 탭에서 용어를 이해하고 오세요!
            """)
    
    # TAB 3: 역사적 비교
    with tab3:
        st.header("🏆 역사적 기록 비교")
        
        st.subheader("📜 KOSPI 역사적 상승 기록 TOP 12")
        hist_df = get_historical_df()
        
        display_df = hist_df.copy()
        display_df = display_df.rename(columns={
            '원래순위': '순위', '정규화기울기': '상승 속도',
            '원본기울기': '하루 평균', '신뢰도': '직선도'
        })
        
        st.dataframe(
            display_df.style.format({
                '상승 속도': '{:.2f}점', '하루 평균': '{:.2f}p',
                '직선도': '{:.1%}', '수익률': '{:.2f}%'
            }),
            use_container_width=True, hide_index=True
        )
        
        st.markdown("---")
        
        if 'current_stats' in st.session_state:
            current_stats = st.session_state['current_stats']
            
            st.subheader(f"🔴 현재 분석 결과: {current_stats['period_name']}")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("📊 하루 평균", f"{current_stats['slope']:.2f}p", help="체감 상승폭")
            with col2:
                st.metric("🚀 상승 속도", f"{current_stats['normalized_slope']:.2f}점", help="시대 비교 점수")
            with col3:
                st.metric("📏 직선도", f"{current_stats['r_squared']:.1%}", help="과열 지표")
            with col4:
                st.metric("📈 총 수익률", f"{current_stats['pct_change']:.2f}%")
            with col5:
                st.metric("📅 기간", f"{current_stats['trading_days']}일")
            
            st.markdown("---")
            st.subheader("📊 비교 기준 선택")
            
            metric_options = {
                "📊 하루 평균 (체감 상승폭)": ("원본기울기", False),
                "🚀 상승 속도 (시대 비교 점수)": ("정규화기울기", False),
                "📏 직선도 (과열 지표)": ("신뢰도", False),
                "📈 수익률 (%)": ("수익률", False),
                "📅 기간 (일) - 짧은순": ("기간(일)", True),
            }
            
            selected_metric = st.selectbox("어떤 기준으로 순위를 볼까요?",
                                           options=list(metric_options.keys()), index=0)
            
            metric_col, ascending = metric_options[selected_metric]
            
            fig_compare, rank, total = create_historical_comparison_chart(
                hist_df, current_stats, metric_col, selected_metric, ascending
            )
            
            if rank <= 3:
                rank_emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
                st.success(f"{rank_emoji} **현재 분석 순위: {rank}위 / {total}개** - 역사적 TOP 3!")
            elif rank <= 6:
                st.info(f"🏅 **현재 분석 순위: {rank}위 / {total}개** - 상위권!")
            else:
                st.warning(f"📊 **현재 분석 순위: {rank}위 / {total}개**")
            
            st.plotly_chart(fig_compare, use_container_width=True)
            
            st.markdown("---")
            st.subheader("🎯 종합 비교 (레이더 차트)")
            fig_radar = create_radar_chart(current_stats, hist_df)
            st.plotly_chart(fig_radar, use_container_width=True)
            st.caption("※ 레이더 차트는 역사적 최대/최소값을 기준으로 0-100 스케일로 정규화한 값입니다.")
            
            st.markdown("---")
            st.subheader("📋 전체 지표별 순위 요약")
            
            summary_data = []
            for metric_name, (metric_col, asc) in metric_options.items():
                _, rank, total = create_historical_comparison_chart(
                    hist_df, current_stats, metric_col, metric_name, asc
                )
                
                if metric_col == '원본기울기':
                    current_val = f"{current_stats['slope']:.2f}p"
                elif metric_col == '정규화기울기':
                    current_val = f"{current_stats['normalized_slope']:.2f}점"
                elif metric_col == '신뢰도':
                    current_val = f"{current_stats['r_squared']:.1%}"
                elif metric_col == '수익률':
                    current_val = f"{current_stats['pct_change']:.2f}%"
                elif metric_col == '기간(일)':
                    current_val = f"{current_stats['trading_days']}일"
                
                summary_data.append({
                    "지표": metric_name,
                    "현재 값": current_val,
                    "순위": f"{rank}위 / {total}개",
                    "백분위": f"상위 {(rank/total)*100:.1f}%"
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ 먼저 **'📊 기울기 분석'** 탭에서 분석을 실행해주세요.")
    
    st.markdown("---")
    st.caption("💡 데이터 출처: FinanceDataReader (KRX)")


if __name__ == "__main__":
    main()

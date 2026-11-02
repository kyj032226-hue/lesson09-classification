import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------
# 1. 페이지 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="데이터 탐색 - 뇌졸중 예측 실습실 🧠",
    page_icon="🔍",
    layout="wide"
)

# ---------------------------------------------------------
# 2. 데이터 로드 (캐싱 활용)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# ---------------------------------------------------------
# 3. 타이틀 및 안내
# ---------------------------------------------------------
st.title("🔍 뇌졸중 데이터 탐색 (EDA)")
st.caption("주요 변수들의 분포와 뇌졸중(Stroke)과의 연관성을 상세히 분석합니다.")
st.markdown("---")

# ---------------------------------------------------------
# 4. 나이와 평균 혈당의 분포 (히스토그램 두 개 나란히)
# ---------------------------------------------------------
st.subheader("1️⃣ 나이 및 평균 혈당 수준 분포")
hist_col1, hist_col2 = st.columns(2)

with hist_col1:
    fig_age = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이(Age) 분포 히스토그램",
        labels={"age": "나이 (Age)", "count": "인원수"},
        color_discrete_sequence=["#2B6CB0"]
    )
    fig_age.update_layout(template="plotly_white")
    st.plotly_chart(fig_age, use_container_width=True)

with hist_col2:
    fig_glucose = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 수준(avg_glucose_level) 분포 히스토그램",
        labels={"avg_glucose_level": "평균 혈당 수준", "count": "인원수"},
        color_discrete_sequence=["#DD6B20"]
    )
    fig_glucose.update_layout(template="plotly_white")
    st.plotly_chart(fig_glucose, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 5. 뇌졸중 여부에 따른 나이 & 평균 혈당 비교 (상자그림 & 평균 표)
# ---------------------------------------------------------
st.subheader("2️⃣ 뇌졸중 여부별 나이 및 평균 혈당 비교")

# 데이터 처리: stroke 값 직관적 레이블화
df_stroke_label = df.copy()
df_stroke_label['뇌졸중 여부'] = df_stroke_label['stroke'].map({0: '정상 (0)', 1: '뇌졸중 (1)'})

box_col1, box_col2 = st.columns(2)

with box_col1:
    fig_box_age = px.box(
        df_stroke_label,
        x="뇌졸중 여부",
        y="age",
        color="뇌졸중 여부",
        title="뇌졸중 여부별 나이 분포 (Box Plot)",
        labels={"age": "나이 (Age)"},
        color_discrete_map={'정상 (0)': '#3182CE', '뇌졸중 (1)': '#E53E3E'}
    )
    fig_box_age.update_layout(template="plotly_white")
    st.plotly_chart(fig_box_age, use_container_width=True)

with box_col2:
    fig_box_glucose = px.box(
        df_stroke_label,
        x="뇌졸중 여부",
        y="avg_glucose_level",
        color="뇌졸중 여부",
        title="뇌졸중 여부별 평균 혈당 분포 (Box Plot)",
        labels={"avg_glucose_level": "평균 혈당 수준"},
        color_discrete_map={'정상 (0)': '#3182CE', '뇌졸중 (1)': '#E53E3E'}
    )
    fig_box_glucose.update_layout(template="plotly_white")
    st.plotly_chart(fig_box_glucose, use_container_width=True)

# 두 그룹의 평균값 표 작성
st.markdown("##### 📌 뇌졸중 여부별 나이 및 평균 혈당 평균값 비교")
mean_table = df_stroke_label.groupby('뇌졸중 여부')[['age', 'avg_glucose_level']].mean().reset_index()
mean_table.columns = ['구분', '평균 나이 (세)', '평균 혈당 (mg/dL)']
st.dataframe(mean_table, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 6. 고혈압 및 심장병 유무에 따른 뇌졸중 비율 (막대그래프)
# ---------------------------------------------------------
st.subheader("3️⃣ 기저질환(고혈압 · 심장병) 유무에 따른 뇌졸중 발생 비율")

bar_col1, bar_col2 = st.columns(2)

with bar_col1:
    ht_df = df.groupby('hypertension')['stroke'].mean().reset_index()
    ht_df['hypertension_label'] = ht_df['hypertension'].map({0: '고혈압 없음 (0)', 1: '고혈압 있음 (1)'})
    ht_df['stroke_ratio'] = ht_df['stroke'] * 100

    fig_ht = px.bar(
        ht_df,
        x='hypertension_label',
        y='stroke_ratio',
        color='hypertension_label',
        title='고혈압 유무별 뇌졸중 발생 비율 (%)',
        labels={'hypertension_label': '고혈압 유무', 'stroke_ratio': '뇌졸중 비율 (%)'},
        text_auto='.2f',
        color_discrete_sequence=['#4299E1', '#E53E3E']
    )
    fig_ht.update_layout(template="plotly_white", yaxis_title="비율 (%)")
    st.plotly_chart(fig_ht, use_container_width=True)

with bar_col2:
    hd_df = df.groupby('heart_disease')['stroke'].mean().reset_index()
    hd_df['heart_disease_label'] = hd_df['heart_disease'].map({0: '심장병 없음 (0)', 1: '심장병 있음 (1)'})
    hd_df['stroke_ratio'] = hd_df['stroke'] * 100

    fig_hd = px.bar(
        hd_df,
        x='heart_disease_label',
        y='stroke_ratio',
        color='heart_disease_label',
        title='심장병 유무별 뇌졸중 발생 비율 (%)',
        labels={'heart_disease_label': '심장병 유무', 'stroke_ratio': '뇌졸중 비율 (%)'},
        text_auto='.2f',
        color_discrete_sequence=['#4299E1', '#E53E3E']
    )
    fig_hd.update_layout(template="plotly_white", yaxis_title="비율 (%)")
    st.plotly_chart(fig_hd, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 7. BMI 결측치(빈 값) 집단의 뇌졸중 비율 비교
# ---------------------------------------------------------
st.subheader("4️⃣ BMI 결측 대상자 vs 전체 뇌졸중 비율 비교")

bmi_null_df = df[df['bmi'].isnull()]

bmi_null_total = len(bmi_null_df)
bmi_null_stroke_count = int(bmi_null_df['stroke'].sum()) if bmi_null_total > 0 else 0
bmi_null_stroke_ratio = (bmi_null_stroke_count / bmi_null_total * 100) if bmi_null_total > 0 else 0.0

overall_total = len(df)
overall_stroke_count = int(df['stroke'].sum())
overall_stroke_ratio = (overall_stroke_count / overall_total * 100)

comparison_data = [
    {
        "구분": "BMI 누락(결측) 대상자",
        "전체 사람 수 (명)": bmi_null_total,
        "뇌졸중 환자 수 (명)": bmi_null_stroke_count,
        "뇌졸중 발생 비율 (%)": f"{bmi_null_stroke_ratio:.2f}%"
    },
    {
        "구분": "전체 데이터셋",
        "전체 사람 수 (명)": overall_total,
        "뇌졸중 환자 수 (명)": overall_stroke_count,
        "뇌졸중 발생 비율 (%)": f"{overall_stroke_ratio:.2f}%"
    }
]

comparison_df = pd.DataFrame(comparison_data)
st.dataframe(comparison_df, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 8. 흡연 상태(smoking_status)별 인원수 표
# ---------------------------------------------------------
st.subheader("5️⃣ 흡연 상태(smoking_status)별 인원수 현황")

smoke_counts = df['smoking_status'].value_counts().reset_index()
smoke_counts.columns = ['흡연 상태 (smoking_status)', '인원수 (명)']

# 전체 인원 대비 비율 계산 추가
smoke_counts['비율 (%)'] = (smoke_counts['인원수 (명)'] / len(df) * 100).round(2)

st.dataframe(smoke_counts, use_container_width=True)

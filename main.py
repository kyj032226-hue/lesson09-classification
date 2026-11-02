import os
import json
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------
# 1. 페이지 및 브라우저 탭 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실 🧠",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# JSON 데이터 저장용 폴더 경로 생성
SAVE_DIR = "내_기록"
os.makedirs(SAVE_DIR, exist_ok=True)
COLUMNS_JSON_PATH = os.path.join(SAVE_DIR, "column_meanings.json")
SOURCE_JSON_PATH = os.path.join(SAVE_DIR, "data_source.json")

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
# 3. 앱 타이틀 및 헤더
# ---------------------------------------------------------
st.title("🧠 뇌졸중 예측 실습실")
st.caption("Stroke Prediction Data Analysis & Machine Learning Workspace")
st.markdown("---")

# ---------------------------------------------------------
# 4. 주요 데이터 지표 (상단 카드 4개)
# ---------------------------------------------------------
total_people = len(df)
total_cols = len(df.columns)
stroke_count = int(df['stroke'].sum()) if 'stroke' in df.columns else 0
stroke_ratio = (stroke_count / total_people * 100) if total_people > 0 else 0.0

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="👥 전체 사람 수", value=f"{total_people:,} 명")

with col2:
    st.metric(label="📊 열(Column) 개수", value=f"{total_cols} 개")

with col3:
    st.metric(label="⚠️ 뇌졸중 환자 (Stroke=1)", value=f"{stroke_count:,} 명")

with col4:
    st.metric(label="📈 뇌졸중 발생 비율", value=f"{stroke_ratio:.2f} %")

st.markdown("---")

# ---------------------------------------------------------
# 5. 열(Column) 정보 및 우리말 뜻 편집 기능 (JSON 저장/복원)
# ---------------------------------------------------------
st.subheader("📋 데이터 컬럼 명세서 및 우리말 뜻 입력")
st.markdown("교재를 참고하여 **'우리말 뜻'** 칸을 직접 수정하고 하단의 **'저장하기'** 버튼을 누르세요. 저장된 내용이 자동으로 보존됩니다.")

# 저장된 우리말 뜻 불러오기
saved_meanings = {}
if os.path.exists(COLUMNS_JSON_PATH):
    try:
        with open(COLUMNS_JSON_PATH, "r", encoding="utf-8") as f:
            saved_meanings = json.load(f)
    except Exception:
        saved_meanings = {}

# 데이터 프레임 열 정보 정리
column_info_list = []
for col in df.columns:
    # 값의 종류 (유니크한 수 또는 대표 타입)
    dtype_str = str(df[col].dtype)
    null_count = int(df[col].isnull().sum())
    
    # 이전에 저장된 한국어 설명이 있으면 로드, 없으면 빈값
    korean_meaning = saved_meanings.get(col, "")
    
    column_info_list.append({
        "열 이름": col,
        "우리말 뜻": korean_meaning,
        "값의 종류 (데이터 타입)": dtype_str,
        "빈 값(Null) 개수": null_count
    })

info_df = pd.DataFrame(column_info_list)

# Data Editor를 통한 사용자 직접 편집
edited_info_df = st.data_editor(
    info_df,
    num_rows="fixed",
    disabled=["열 이름", "값의 종류 (데이터 타입)", "빈 값(Null) 개수"],
    use_container_width=True,
    key="column_editor"
)

# 한국어 설명 저장 버튼
if st.button("💾 우리말 뜻 저장하기", use_container_width=True):
    meanings_to_save = {}
    for idx, row in edited_info_df.iterrows():
        meanings_to_save[row["열 이름"]] = str(row["우리말 뜻"])
    
    with open(COLUMNS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(meanings_to_save, f, ensure_ascii=False, indent=4)
    st.success("우리말 뜻이 '내_기록/column_meanings.json' 파일에 성공적으로 저장되었습니다!")

st.markdown("---")

# ---------------------------------------------------------
# 6. 미리보기 & Plotly 데이터 시각화
# ---------------------------------------------------------
st.subheader("🔍 데이터 상위 5개 행 (Head Preview)")
st.dataframe(df.head(5), use_container_width=True)

# Plotly 시각화 영역
st.subheader("📊 연령별 뇌졸중 발생 현황 (Plotly Chart)")
if 'age' in df.columns and 'stroke' in df.columns:
    fig = px.histogram(
        df, 
        x="age", 
        color="stroke", 
        barmode="overlay",
        title="나이(Age)별 뇌졸중(Stroke) 환자 분포",
        labels={"age": "나이 (Age)", "stroke": "뇌졸중 여부"},
        color_discrete_map={0: "#3366CC", 1: "#FF4B4B"}
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 7. 데이터 출처 메모 입력 기능 (JSON 저장/복원)
# ---------------------------------------------------------
st.subheader("📚 데이터 출처 메모")

# 저장된 출처 메모 불러오기
default_source_text = ""
if os.path.exists(SOURCE_JSON_PATH):
    try:
        with open(SOURCE_JSON_PATH, "r", encoding="utf-8") as f:
            data_source = json.load(f)
            default_source_text = data_source.get("source", "")
    except Exception:
        default_source_text = ""

# 사용자가 입력할 수 있는 텍스트 에어리어
source_text_input = st.text_area(
    "교재에 기재된 데이터 출처 정보를 아래에 적어주세요.",
    value=default_source_text,
    height=120,
    placeholder="예: Kaggle Stroke Prediction Dataset / 출처 교재 페이지 등..."
)

if st.button("💾 출처 정보 저장하기"):
    with open(SOURCE_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({"source": source_text_input}, f, ensure_ascii=False, indent=4)
    st.success("출처 정보가 '내_기록/data_source.json' 파일에 저장되었습니다!")

import os
import json
import pandas as pd
import streamlit as st

# 1. 페이지 기본 설정 (페이지 제목 및 브라우저 탭 아이콘)
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

# 2. 내_기록 폴더 및 JSON 저장 경로 설정
SAVE_DIR = "내_기록"
SAVE_FILE = os.path.join(SAVE_DIR, "user_notes.json")

def ensure_directory_exists():
    """저장 폴더가 없으면 생성합니다."""
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

def load_user_notes():
    """저장된 사용자 메모(우리말 뜻 및 출처)를 불러옵니다."""
    ensure_directory_exists()
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_user_notes(notes):
    """사용자 메모를 JSON 파일로 저장합니다."""
    ensure_directory_exists()
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=4)

# 사용자 기록 불러오기 및 세션 상태 초기화
user_notes = load_user_notes()

# 3. 데이터 로드 (캐싱 사용)
DATA_URL = "https://raw.githubusercontent.com/kyj032226-hue/data/remain/stroke.csv"

@st.cache_data
def load_data():
    # UTF-8 인코딩으로 데이터 불러오기
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 4. 앱 헤더 및 타이틀
st.title("🧠 뇌졸중 예측 실습실")
st.markdown("본 실습실은 뇌졸중 관련 데이터를 탐색하고 모델을 학습시키는 공간입니다.")
st.divider()

# 5. 상단 지표 카드 (4개)
total_count = len(df)
total_cols = len(df.columns)
stroke_count = int(df['stroke'].sum()) if 'stroke' in df.columns else 0
stroke_ratio = (stroke_count / total_count * 100) if total_count > 0 else 0.0

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="👥 전체 사람 수", value=f"{total_count:,} 명")

with col2:
    st.metric(label="📊 열(Feature) 개수", value=f"{total_cols} 개")

with col3:
    st.metric(label="⚠️ 뇌졸중(stroke=1) 환자 수", value=f"{stroke_count:,} 명")

with col4:
    st.metric(label="📈 뇌졸중 환자 비율", value=f"{stroke_ratio:.2f} %")

st.divider()

# 6. 데이터 세부 정보 및 우리말 뜻 작성 섹션
st.subheader("📋 데이터 열(Column) 정보 및 우리말 뜻 정리")
st.caption("교재를 참고하여 각 열의 우리말 뜻을 입력하고 저장해 보세요. 저장된 내용은 새로고침해도 유지됩니다.")

# 데이터 열별 정보 생성
col_info_list = []
for col in df.columns:
    null_cnt = df[col].isnull().sum()
    dtype_str = str(df[col].dtype)
    col_info_list.append({
        "열 이름": col,
        "값의 종류 (타입)": dtype_str,
        "빈 값(결측치) 개수": f"{null_cnt}개" if null_cnt > 0 else "0개"
    })

info_df = pd.DataFrame(col_info_list)

# 열의 우리말 뜻을 입력할 수 있는 Form 생성
saved_meanings = user_notes.get("meanings", {})

with st.form(key="meanings_form"):
    updated_meanings = {}
    st.write("각 열의 **우리말 뜻**을 입력해 주세요:")
    
    # 2컬럼 레이아웃으로 입력 필드 배치
    input_cols = st.columns(2)
    for idx, row in info_df.iterrows():
        col_name = row["열 이름"]
        default_val = saved_meanings.get(col_name, "")
        
        target_col = input_cols[idx % 2]
        with target_col:
            user_input = st.text_input(
                label=f"**{col_name}** ({row['값의 종류 (타입)']}, 결측치: {row['빈 값(결측치) 개수']})",
                value=default_val,
                key=f"meaning_{col_name}"
            )
            updated_meanings[col_name] = user_input

    submit_meanings = st.form_submit_button(label="💾 우리말 뜻 저장하기")

if submit_meanings:
    user_notes["meanings"] = updated_meanings
    save_user_notes(user_notes)
    st.success("우리말 뜻이成功적으로 저장되었습니다!")

# 종합 정보 표 표시
info_df["우리말 뜻"] = info_df["열 이름"].map(lambda x: updated_meanings.get(x, saved_meanings.get(x, "")))
# 열 순서 조정: 열 이름 -> 우리말 뜻 -> 값의 종류 -> 빈 값 개수
info_df = info_df[["열 이름", "우리말 뜻", "값의 종류 (타입)", "빈 값(결측치) 개수"]]
st.dataframe(info_df, use_container_width=True)

st.divider()

# 7. 데이터 상위 5줄 미리보기
st.subheader("🔍 데이터 미리보기 (처음 5줄)")
st.dataframe(df.head(5), use_container_width=True)

st.divider()

# 8. 데이터 출처 및 메모 작성 (하단)
st.subheader("📚 데이터 출처 및 메모")
saved_source = user_notes.get("source", "")

with st.form(key="source_form"):
    source_text = st.text_area(
        label="교재에 나와 있는 데이터 출처나 추가 설명 내용을 기록하세요:",
        value=saved_source,
        height=120,
        placeholder="예: Kaggle Stroke Prediction Dataset..."
    )
    submit_source = st.form_submit_button(label="💾 출처 내용 저장하기")

if submit_source:
    user_notes["source"] = source_text
    save_user_notes(user_notes)
    st.success("출처 및 메모가 저장되었습니다!")

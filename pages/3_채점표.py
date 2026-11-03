port pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------
# 1. 페이지 기본 설정import numpy as np
im
# ---------------------------------------------------------
st.set_page_config(
    page_title="채점표 - 뇌졸중 예측 실습실 🧠",
    page_icon="📝",
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

st.title("📝 뇌졸중 모델 채점표")
st.caption("테스트 데이터셋을 바탕으로 혼동행렬(Confusion Matrix) 및 평가 지표(정확도·재현율·정밀도·F1)를 상세 비교합니다.")
st.markdown("---")

# ---------------------------------------------------------
# 3. 데이터 분할 (10명 묶음 중 앞 3명 테스트, 나머지 7명 훈련)
# ---------------------------------------------------------
df_sorted = df.copy().sort_index()

test_mask = (df_sorted.index % 10) < 3
train_mask = ~test_mask

train_df = df_sorted[train_mask].copy()
test_df = df_sorted[test_mask].copy()

# 기본 4개 속성 (bmi 제외)
selected_features = ["age", "avg_glucose_level", "hypertension", "heart_disease"]

X_train = train_df[selected_features]
y_train = train_df["stroke"]
X_test = test_df[selected_features]
y_test = test_df["stroke"]

# ---------------------------------------------------------
# 4. 모델 학습
# ---------------------------------------------------------
# 1) 기준 모델 (훈련용 다수 클래스로 전건 예측)
majority_class = int(y_train.mode()[0])
base_test_pred = np.full(len(y_test), majority_class)

# 2) 로지스틱 회귀
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)
lr_test_pred = log_reg.predict(X_test_scaled)

# 3) 의사결정트리
tree_clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
tree_clf.fit(X_train, y_train)
dt_test_pred = tree_clf.predict(X_test)

# ---------------------------------------------------------
# 5. 테스트 데이터 현황 요약
# ---------------------------------------------------------
test_total_count = len(test_df)
test_stroke_count = int(y_test.sum())

st.info(f"📌 **테스트용 사람 수:** 총 `{test_total_count:,}`명 | **그중 실제 뇌졸중인 사람 수:** `{test_stroke_count:,}`명")
st.markdown("---")

# ---------------------------------------------------------
# 6. 혼동행렬 계산 및 2x2 표 계산 함수
# ---------------------------------------------------------
def compute_cm_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    
    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    
    # 재현율 (Recall = TP / (TP + FN)) -> 실제 뇌졸중인 사람 분모
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # 정밀도 (Precision = TP / (TP + FP)) -> 예측 뇌졸중인 사람 분모
    if (tp + fp) == 0:
        precision_str = "계산할 수 없음"
        f1_str = "계산할 수 없음"
    else:
        precision = tp / (tp + fp)
        precision_str = f"{precision:.4f}"
        if (precision + recall) == 0:
            f1_str = "계산할 수 없음"
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
            f1_str = f"{f1:.4f}"
            
    recall_str = f"{recall:.4f}"
    acc_str = f"{acc:.4f}"
    
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "acc_str": acc_str,
        "recall_str": recall_str,
        "precision_str": precision_str,
        "f1_str": f1_str
    }

base_metrics = compute_cm_metrics(y_test, base_test_pred)
lr_metrics = compute_cm_metrics(y_test, lr_test_pred)
dt_metrics = compute_cm_metrics(y_test, dt_test_pred)

# ---------------------------------------------------------
# 7. 모델별 2x2 혼동행렬 표시
# ---------------------------------------------------------
st.subheader("🧩 모델별 혼동행렬 (2×2 표)")

col_cm1, col_cm2 = st.columns(2)

def render_cm_table(metrics, title_name):
    tp, fp, fn, tn = metrics["TP"], metrics["FP"], metrics["FN"], metrics["TN"]
    row_act_0 = tn + fp
    row_act_1 = fn + tp
    col_pred_0 = tn + fn
    col_pred_1 = fp + tp
    total = tn + fp + fn + tp
    
    cm_df = pd.DataFrame({
        "예측: 아님(0)": [
            f"TN (진음성): {tn:,}명",
            f"FN (위음성): {fn:,}명",
            f"합계: {col_pred_0:,}명"
        ],
        "예측: 뇌졸중(1)": [
            f"FP (위양성): {fp:,}명",
            f"TP (진양성): {tp:,}명",
            f"합계: {col_pred_1:,}명"
        ],
        "행 합계": [
            f"실제 아님 합계: {row_act_0:,}명",
            f"실제 뇌졸중 합계: {row_act_1:,}명",
            f"전체: {total:,}명"
        ]
    }, index=["실제: 아님(0)", "실제: 뇌졸중(1)", "열 합계"])
    
    st.markdown(f"##### 📌 {title_name}")
    st.dataframe(cm_df, use_container_width=True)

with col_cm1:
    render_cm_table(lr_metrics, "로지스틱 회귀(확률로 답하는 모델)")

with col_cm2:
    render_cm_table(dt_metrics, "의사결정트리(질문으로 답하는 모델)")

st.markdown("---")

# ---------------------------------------------------------
# 8. 네 가지 지표 종합 비교 표
# ---------------------------------------------------------
st.subheader("📊 성능 평가 지표 비교표")

eval_summary_df = pd.DataFrame([
    {
        "모델 이름": "훈련용에서 많은 쪽으로만 답하는 모델",
        "정확도 (Accuracy)": base_metrics["acc_str"],
        "재현율 (Recall)": base_metrics["recall_str"],
        "정밀도 (Precision)": base_metrics["precision_str"],
        "F1 스코어 (F1-Score)": base_metrics["f1_str"]
    },
    {
        "모델 이름": "로지스틱 회귀(확률로 답하는 모델)",
        "정확도 (Accuracy)": lr_metrics["acc_str"],
        "재현율 (Recall)": lr_metrics["recall_str"],
        "정밀도 (Precision)": lr_metrics["precision_str"],
        "F1 스코어 (F1-Score)": lr_metrics["f1_str"]
    },
    {
        "모델 이름": "의사결정트리(질문으로 답하는 모델)",
        "정확도 (Accuracy)": dt_metrics["acc_str"],
        "재현율 (Recall)": dt_metrics["recall_str"],
        "정밀도 (Precision)": dt_metrics["precision_str"],
        "F1 스코어 (F1-Score)": dt_metrics["f1_str"]
    }
])

st.dataframe(eval_summary_df, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 9. 예측 오류 샘플 상세 목록 (놓친 환자 vs 헛짚은 사람)
# ---------------------------------------------------------
st.subheader("🔍 예측 오류 상세 분석 목록")

selected_model_name = st.selectbox(
    "오류 목록을 확인할 모델을 선택하세요:",
    options=["로지스틱 회귀(확률로 답하는 모델)", "의사결정트리(질문으로 답하는 모델)"]
)

# 선택된 모델의 예측 결과 연결
if selected_model_name == "로지스틱 회귀(확률로 답하는 모델)":
    pred_vals = lr_test_pred
    current_metrics = lr_metrics
else:
    pred_vals = dt_test_pred
    current_metrics = dt_metrics

test_df_eval = test_df.copy()
test_df_eval["pred"] = pred_vals

# 1) 놓친 환자 (FN: 실제 stroke=1, 예측 pred=0)
fn_df = test_df_eval[(test_df_eval["stroke"] == 1) & (test_df_eval["pred"] == 0)].copy()
# 2) 헛짚은 사람 (FP: 실제 stroke=0, 예측 pred=1)
fp_df = test_df_eval[(test_df_eval["stroke"] == 0) & (test_df_eval["pred"] == 1)].copy()

# 공통 정렬 및 열 구성 함수
def prepare_error_list(df_sub):
    if df_sub.empty:
        return pd.DataFrame(columns=["번호(id)", "나이", "평균 혈당", "고혈압 여부", "심장병 여부"])
    
    # 나이 기준 내림차순 정렬
    df_sorted_sub = df_sub.sort_values(by="age", ascending=False)
    
    # 보기 쉬운 열 이름 구성
    result_df = pd.DataFrame({
        "번호(id)": df_sorted_sub["id"],
        "나이": df_sorted_sub["age"],
        "평균 혈당": df_sorted_sub["avg_glucose_level"],
        "고혈압 여부": df_sorted_sub["hypertension"].map({0: "없음(0)", 1: "있음(1)"}),
        "심장병 여부": df_sorted_sub["heart_disease"].map({0: "없음(0)", 1: "있음(1)"})
    })
    return result_df

fn_prepared = prepare_error_list(fn_df)
fp_prepared = prepare_error_list(fp_df)

fn_total_count = len(fn_prepared)
fp_total_count = len(fp_prepared)

err_col1, err_col2 = st.columns(2)

with err_col1:
    st.markdown(f"##### ⚠️ 놓친 환자 (FN) - 총 `{fn_total_count:,}`명")
    st.caption("실제 뇌졸중인데 모델이 아님(0)으로 예측한 사례 (나이 내림차순)")
    if fn_total_count > 20:
        st.write("*(스무 줄을 넘어가므로 상위 20개 행만 표시합니다)*")
    st.dataframe(fn_prepared.head(20), use_container_width=True)

with err_col2:
    st.markdown(f"##### ❓ 헛짚은 사람 (FP) - 총 `{fp_total_count:,}`명")
    st.caption("실제 아님(0)인데 모델이 뇌졸중(1)으로 예측한 사례 (나이 내림차순)")
    if fp_total_count > 20:
        st.write("*(스무 줄을 넘어가므로 상위 20개 행만 표시합니다)*")
    st.dataframe(fp_prepared.head(20), use_container_width=True)

st.info(
    f"💡 **안내:** 위 두 목록의 사람 수(**놓친 환자 {fn_total_count}명**, **헛짚은 사람 {fp_total_count}명**)는 "
    f"선택한 모델 혼동행렬의 **FN({current_metrics['FN']}명)** 및 **FP({current_metrics['FP']}명)** 칸에 적힌 사람 수와 정확히 일치합니다."
)

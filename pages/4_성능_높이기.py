import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

# ---------------------------------------------------------
# 1. 페이지 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="성능 높이기 - 뇌졸중 예측 실습실 🧠",
    page_icon="⚡",
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

st.title("⚡ 성능 높이기 (클래스 가중치 불균형 조정)")
st.caption("뇌졸중 데이터의 클래스 불균형을 해소하기 위해 클래스 가중치를 조정하기 전과 후의 성능 및 결정 경계 변화를 비교합니다.")
st.markdown("---")

# ---------------------------------------------------------
# 3. 속성 매핑 및 선택
# ---------------------------------------------------------
FEATURE_MAP = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}
REVERSE_FEATURE_MAP = {v: k for k, v in FEATURE_MAP.items()}

all_korean_features = list(FEATURE_MAP.values())
default_korean_features = ["나이", "평균 혈당", "고혈압", "심장병"]  # bmi 제외 넷

selected_korean_features = st.multiselect(
    "입력으로 사용할 속성을 고르세요 (최소 2개 이상):",
    options=all_korean_features,
    default=default_korean_features
)

if len(selected_korean_features) < 2:
    st.warning("⚠️ 최소 2개 이상의 속성을 선택해야 모델 학습 및 시각화를 진행할 수 있습니다.")
    st.stop()

selected_features = [REVERSE_FEATURE_MAP[f] for f in selected_korean_features]

# ---------------------------------------------------------
# 4. 데이터 분할 (10명 묶음 중 앞 3명 테스트, 나머지 7명 훈련)
# ---------------------------------------------------------
df_sorted = df.copy().sort_index()

test_mask = (df_sorted.index % 10) < 3
train_mask = ~test_mask

train_df = df_sorted[train_mask].copy()
test_df = df_sorted[test_mask].copy()

# BMI 결측치 처리 (선택된 경우 훈련용 중앙값으로 채움)
if "bmi" in selected_features:
    bmi_median_train = train_df["bmi"].median()
    train_df["bmi"] = train_df["bmi"].fillna(bmi_median_train)
    test_df["bmi"] = test_df["bmi"].fillna(bmi_median_train)

X_train = train_df[selected_features]
y_train = train_df["stroke"]
X_test = test_df[selected_features]
y_test = test_df["stroke"]

# 스케일링 (로지스틱 회귀용)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 5. 모델 학습 (가중치 전 / 후)
# ---------------------------------------------------------
# 1) 로지스틱 회귀
lr_before = LogisticRegression(random_state=42)
lr_before.fit(X_train_scaled, y_train)

lr_after = LogisticRegression(class_weight="balanced", random_state=42)
lr_after.fit(X_train_scaled, y_train)

# 2) 의사결정트리
dt_before = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_before.fit(X_train, y_train)

dt_after = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, class_weight="balanced", random_state=42)
dt_after.fit(X_train, y_train)

# 예측 결과
lr_pred_before = lr_before.predict(X_test_scaled)
lr_pred_after = lr_after.predict(X_test_scaled)

dt_pred_before = dt_before.predict(X_test)
dt_pred_after = dt_after.predict(X_test)

# ---------------------------------------------------------
# 6. 혼동행렬 및 성능 지표 계산 함수
# ---------------------------------------------------------
def calc_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    
    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if (tp + fp) == 0:
        prec_str, f1_str = "계산할 수 없음", "계산할 수 없음"
    else:
        prec = tp / (tp + fp)
        prec_str = f"{prec:.4f}"
        if (prec + recall) == 0:
            f1_str = "계산할 수 없음"
        else:
            f1_str = f"{2 * prec * recall / (prec + recall):.4f}"
            
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "acc": f"{acc:.4f}", "recall": f"{recall:.4f}",
        "precision": prec_str, "f1": f1_str
    }

m_lr_bef = calc_metrics(y_test, lr_pred_before)
m_lr_aft = calc_metrics(y_test, lr_pred_after)

m_dt_bef = calc_metrics(y_test, dt_pred_before)
m_dt_aft = calc_metrics(y_test, dt_pred_after)

# ---------------------------------------------------------
# 7. 전·후 비교 표 출력
# ---------------------------------------------------------
st.subheader("📊 클래스 가중치 조정 전·후 성능 지표 및 혼동행렬 비교")

def create_comparison_df(m_bef, m_aft):
    return pd.DataFrame({
        "구분": ["TP (진양성)", "FP (위양성)", "FN (위음성)", "TN (진음성)", "정확도 (Accuracy)", "재현율 (Recall)", "정밀도 (Precision)", "F1 스코어"],
        "가중치 조정 전": [
            f"{m_bef['TP']:,}명", f"{m_bef['FP']:,}명", f"{m_bef['FN']:,}명", f"{m_bef['TN']:,}명",
            m_bef['acc'], m_bef['recall'], m_bef['precision'], m_bef['f1']
        ],
        "가중치 조정 후": [
            f"{m_aft['TP']:,}명", f"{m_aft['FP']:,}명", f"{m_aft['FN']:,}명", f"{m_aft['TN']:,}명",
            m_aft['acc'], m_aft['recall'], m_aft['precision'], m_aft['f1']
        ]
    })

col_tbl1, col_tbl2 = st.columns(2)

with col_tbl1:
    st.markdown("##### 📈 로지스틱 회귀(확률로 답하는 모델)")
    df_lr_comp = create_comparison_df(m_lr_bef, m_lr_aft)
    st.dataframe(df_lr_comp, use_container_width=True)
    st.info(
        f"💡 **변화 요약:** 찾아낸 환자 수(TP) `{m_lr_bef['TP']}명 ➔ {m_lr_aft['TP']}명`, "
        f"놓친 환자 수(FN) `{m_lr_bef['FN']}명 ➔ {m_lr_aft['FN']}명`, "
        f"헛짚은 사람 수(FP) `{m_lr_bef['FP']}명 ➔ {m_lr_aft['FP']}명`으로 변경되었습니다."
    )

with col_tbl2:
    st.markdown("##### 🌴 의사결정트리(질문으로 답하는 모델)")
    df_dt_comp = create_comparison_df(m_dt_bef, m_dt_aft)
    st.dataframe(df_dt_comp, use_container_width=True)
    st.info(
        f"💡 **변화 요약:** 찾아낸 환자 수(TP) `{m_dt_bef['TP']}명 ➔ {m_dt_aft['TP']}명`, "
        f"놓친 환자 수(FN) `{m_dt_bef['FN']}명 ➔ {m_dt_aft['FN']}명`, "
        f"헛짚은 사람 수(FP) `{m_dt_bef['FP']}명 ➔ {m_dt_aft['FP']}명`으로 변경되었습니다."
    )

st.markdown("---")

# ---------------------------------------------------------
# 8. 2차원 산점도 및 가중치 조정 전/후 결정 경계 시각화
# ---------------------------------------------------------
st.subheader("🗺️ 2차원 산점도 및 결정 경계선 (0.5) 비교")

axis_c1, axis_c2 = st.columns(2)
with axis_c1:
    x_korean = st.selectbox("가로축(X축) 선택:", selected_korean_features, index=0)
with axis_c2:
    y_opts = [f for f in selected_korean_features if f != x_korean]
    y_korean = st.selectbox("세로축(Y축) 선택:", y_opts, index=0)

x_var, y_var = REVERSE_FEATURE_MAP[x_korean], REVERSE_FEATURE_MAP[y_korean]
other_vars = [f for f in selected_features if f not in [x_var, y_var]]

fixed_vals = {v: test_df[v].median() for v in other_vars}

x_min, x_max = test_df[x_var].min(), test_df[x_var].max()
y_min, y_max = test_df[y_var].min(), test_df[y_var].max()

x_pad = (x_max - x_min) * 0.05 if x_max != x_min else 1.0
y_pad = (y_max - y_min) * 0.05 if y_max != y_min else 1.0

x_grid = np.linspace(x_min - x_pad, x_max + x_pad, 100)
y_grid = np.linspace(y_min - y_pad, y_max + y_pad, 100)
xx, yy = np.meshgrid(x_grid, y_grid)

grid_p = pd.DataFrame({x_var: xx.ravel(), y_var: yy.ravel()})
for v in other_vars:
    grid_p[v] = fixed_vals[v]
grid_p = grid_p[selected_features]

# 확률 예측
grid_scaled = scaler.transform(grid_p)
prob_lr_bef = lr_before.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)
prob_lr_aft = lr_after.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)

fig2d = go.Figure()

# 산점도 점
test_df_p = test_df.copy()
test_df_p['뇌졸중 여부'] = test_df_p['stroke'].map({0: '정상 (0)', 1: '뇌졸중 (1)'})

for label, color, symbol in [('정상 (0)', '#2B6CB0', 'circle'), ('뇌졸중 (1)', '#E53E3E', 'x')]:
    sub = test_df_p[test_df_p['뇌졸중 여부'] == label]
    fig2d.add_trace(go.Scatter(
        x=sub[x_var], y=sub[y_var], mode='markers', name=f"실제 {label}",
        marker=dict(color=color, symbol=symbol, size=7, opacity=0.7)
    ))

# 조정 전 0.5 경계
has_bef = (prob_lr_bef.min() <= 0.5 <= prob_lr_bef.max())
if has_bef:
    fig2d.add_trace(go.Contour(
        x=x_grid, y=y_grid, z=prob_lr_bef, contours_start=0.5, contours_end=0.5,
        contours_coloring='lines', line=dict(color='gray', width=3, dash='dash'),
        showscale=False, name='로지스틱 (가중치 전 0.5)'
    ))

# 조정 후 0.5 경계
has_aft = (prob_lr_aft.min() <= 0.5 <= prob_lr_aft.max())
if has_aft:
    fig2d.add_trace(go.Contour(
        x=x_grid, y=y_grid, z=prob_lr_aft, contours_start=0.5, contours_end=0.5,
        contours_coloring='lines', line=dict(color='red', width=3),
        showscale=False, name='로지스틱 (가중치 후 0.5)'
    ))

fig2d.update_layout(
    title=f"2D 결정 경계선 비교 ({x_korean} vs {y_korean})",
    xaxis_title=x_korean, yaxis_title=y_korean, template="plotly_white"
)

st.plotly_chart(fig2d, use_container_width=True)

if not has_bef:
    st.info("ℹ️ **가중치 조정 전** 0.5 결정 경계선이 현재 화면 범위 밖에 있습니다.")
if not has_aft:
    st.info("ℹ️ **가중치 조정 후** 0.5 결정 경계선이 현재 화면 범위 밖에 있습니다.")

st.markdown("---")

# ---------------------------------------------------------
# 9. 3차원 산점도 및 전/후 0.5 결정 평면 (속성 3개 이상 시)
# ---------------------------------------------------------
if len(selected_korean_features) >= 3:
    st.subheader("🧊 3차원 공간 산점도 및 전·후 결정 평면 (0.5)")
    st.caption("마우스를 드래그하여 3D 공간을 자유롭게 회전시켜 볼 수 있습니다.")

    c3d_1, c3d_2, c3d_3 = st.columns(3)
    with c3d_1:
        x3_kr = st.selectbox("3D X축 선택:", selected_korean_features, index=0)
    with c3d_2:
        y3_opts = [f for f in selected_korean_features if f != x3_kr]
        y3_kr = st.selectbox("3D Y축 선택:", y3_opts, index=0)
    with c3d_3:
        z3_opts = [f for f in selected_korean_features if f not in [x3_kr, y3_kr]]
        z3_kr = st.selectbox("3D Z축 선택:", z3_opts, index=0)

    x3_var, y3_var, z3_var = REVERSE_FEATURE_MAP[x3_kr], REVERSE_FEATURE_MAP[y3_kr], REVERSE_FEATURE_MAP[z3_kr]
    other3_vars = [f for f in selected_features if f not in [x3_var, y3_var, z3_var]]
    fixed3_vals = {v: test_df[v].median() for v in other3_vars}

    x3_grid = np.linspace(test_df[x3_var].min(), test_df[x3_var].max(), 30)
    y3_grid = np.linspace(test_df[y3_var].min(), test_df[y3_var].max(), 30)
    xx3, yy3 = np.meshgrid(x3_grid, y3_grid)

    z3_grid = np.linspace(test_df[z3_var].min(), test_df[z3_var].max(), 30)

    # 3D Meshgrid
    xxx, yyy, zzz = np.meshgrid(x3_grid, y3_grid, z3_grid)
    grid3_p = pd.DataFrame({x3_var: xxx.ravel(), y3_var: yyy.ravel(), z3_var: zzz.ravel()})
    for v in other3_vars:
        grid3_p[v] = fixed3_vals[v]
    grid3_p = grid3_p[selected_features]

    grid3_scaled = scaler.transform(grid3_p)
    prob3_bef = lr_before.predict_proba(grid3_scaled)[:, 1].reshape(xxx.shape)
    prob3_aft = lr_after.predict_proba(grid3_scaled)[:, 1].reshape(xxx.shape)

    fig3d = go.Figure()

    # 3D 산점도 점
    for label, color, symbol in [('정상 (0)', '#2B6CB0', 'circle'), ('뇌졸중 (1)', '#E53E3E', 'diamond')]:
        sub3 = test_df_p[test_df_p['뇌졸중 여부'] == label]
        fig3d.add_trace(go.Scatter3d(
            x=sub3[x3_var], y=sub3[y3_var], z=sub3[z3_var],
            mode='markers', name=f"실제 {label}",
            marker=dict(color=color, symbol=symbol, size=4, opacity=0.8)
        ))

    # 3D Isosurface (0.5 확률 평면)
    fig3d.add_trace(go.Isosurface(
        x=xxx.ravel(), y=yyy.ravel(), z=zzz.ravel(), value=prob3_bef.ravel(),
        isomin=0.5, isomax=0.5, surface_count=1, opacity=0.4,
        colorscale=[[0, 'gray'], [1, 'gray']], showscale=False, name='가중치 전 0.5 평면'
    ))

    fig3d.add_trace(go.Isosurface(
        x=xxx.ravel(), y=yyy.ravel(), z=zzz.ravel(), value=prob3_aft.ravel(),
        isomin=0.5, isomax=0.5, surface_count=1, opacity=0.5,
        colorscale=[[0, 'red'], [1, 'red']], showscale=False, name='가중치 후 0.5 평면'
    ))

    fig3d.update_layout(
        title="3D 공간 산점도 및 전·후 0.5 확률 결정 평면",
        scene=dict(xaxis_title=x3_kr, yaxis_title=y3_kr, zaxis_title=z3_kr),
        template="plotly_white"
    )

    st.plotly_chart(fig3d, use_container_width=True)

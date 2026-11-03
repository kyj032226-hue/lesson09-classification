import numpy as np
import pandas as pd
import plotly.graph_objects as px_go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
import streamlit as st

# ---------------------------------------------------------
# 1. 페이지 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="분류 모델 - 뇌졸중 예측 실습실 🧠",
    page_icon="🤖",
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

st.title("🤖 뇌졸중 분류 모델 실습")
st.caption("로지스틱 회귀와 의사결정트리 모델을 학습시키고 결정 경계 및 트리 구조를 시각화합니다.")
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
    "학습 및 예측에 사용할 속성을 선택하세요 (최소 2개 이상):",
    options=all_korean_features,
    default=default_korean_features
)

if len(selected_korean_features) < 2:
    st.warning("⚠️ 최소 2개 이상의 속성을 선택해야 모델 학습 및 2차원 시각화를 진행할 수 있습니다.")
    st.stop()

selected_features = [REVERSE_FEATURE_MAP[f] for f in selected_korean_features]

# ---------------------------------------------------------
# 4. 데이터 분할 (10명 묶음 중 앞 3명 테스트, 나머지 7명 훈련)
# ---------------------------------------------------------
df_sorted = df.copy()
# 번호(index) 순으로 정렬 보장
df_sorted = df_sorted.sort_index()

# 10명씩 묶었을 때 인덱스 % 10 이 0, 1, 2 이면 테스트용 (3명), 3~9 이면 훈련용 (7명)
test_mask = (df_sorted.index % 10) < 3
train_mask = ~test_mask

train_df = df_sorted[train_mask].copy()
test_df = df_sorted[test_mask].copy()

# ---------------------------------------------------------
# 5. BMI 결측치 처리 (선택된 경우 훈련용 중앙값으로 채움)
# ---------------------------------------------------------
if "bmi" in selected_features:
    bmi_median_train = train_df["bmi"].median()
    train_df["bmi"] = train_df["bmi"].fillna(bmi_median_train)
    test_df["bmi"] = test_df["bmi"].fillna(bmi_median_train)

X_train = train_df[selected_features]
y_train = train_df["stroke"]
X_test = test_df[selected_features]
y_test = test_df["stroke"]

# ---------------------------------------------------------
# 6. 모델 학습
# ---------------------------------------------------------
# 1) 기준 모델 (Baseline: 훈련용에서 다수 클래스로 항상 답하는 모델)
majority_class = y_train.mode()[0]
base_train_pred = np.full(len(y_train), majority_class)
base_test_pred = np.full(len(y_test), majority_class)
base_train_acc = (base_train_pred == y_train).mean()
base_test_acc = (base_test_pred == y_test).mean()

# 2) 로지스틱 회귀 (크기 맞추기 포함)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)

lr_train_acc = log_reg.score(X_train_scaled, y_train)
lr_test_acc = log_reg.score(X_test_scaled, y_test)

# 3) 의사결정트리 (질문 3번 제한, 리프 최소 샘플 5, 난수 고정)
tree_clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
tree_clf.fit(X_train, y_train)

dt_train_acc = tree_clf.score(X_train, y_train)
dt_test_acc = tree_clf.score(X_test, y_test)

# ---------------------------------------------------------
# 7. 정확도 카드 출력
# ---------------------------------------------------------
st.subheader("📊 모델 평가 및 정확도 비교")
col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    st.markdown("### 📌 다수결 기준 모델")
    st.caption("입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델")
    st.metric(
        label="테스트 데이터 정확도",
        value=f"{base_test_acc:.4f} ({base_test_acc*100:.2f}%)",
        delta=None
    )
    st.caption(f"훈련 데이터 정확도: `{base_train_acc:.4f}` | 테스트 정확도: `{base_test_acc:.4f}`")

with col_m2:
    st.markdown("### 📈 로지스틱 회귀")
    st.caption("확률로 답하는 모델")
    st.metric(
        label="테스트 데이터 정확도",
        value=f"{lr_test_acc:.4f} ({lr_test_acc*100:.2f}%)",
        delta=f"{(lr_test_acc - base_test_acc)*100:+.2f}%p vs 기준"
    )
    st.caption(f"훈련 데이터 정확도: `{lr_train_acc:.4f}` | 테스트 정확도: `{lr_test_acc:.4f}`")

with col_m3:
    st.markdown("### 🌴 의사결정트리")
    st.caption("질문으로 답하는 모델")
    st.metric(
        label="테스트 데이터 정확도",
        value=f"{dt_test_acc:.4f} ({dt_test_acc*100:.2f}%)",
        delta=f"{(dt_test_acc - base_test_acc)*100:+.2f}%p vs 기준"
    )
    st.caption(f"훈련 데이터 정확도: `{dt_train_acc:.4f}` | 테스트 정확도: `{dt_test_acc:.4f}`")

st.markdown("---")

# ---------------------------------------------------------
# 8. Plotly 산점도 & 결정 경계 시각화
# ---------------------------------------------------------
st.subheader("🗺️ 2차원 공간 결정 경계 시각화")

axis_col1, axis_col2 = st.columns(2)
with axis_col1:
    x_korean = st.selectbox("가로축(X축)으로 사용할 속성을 선택하세요:", selected_korean_features, index=0)
with axis_col2:
    y_options = [f for f in selected_korean_features if f != x_korean]
    y_korean = st.selectbox("세로축(Y축)으로 사용할 속성을 선택하세요:", y_options, index=0)

x_var = REVERSE_FEATURE_MAP[x_korean]
y_var = REVERSE_FEATURE_MAP[y_korean]

# 두 축이 아닌 고른 속성들의 테스트 데이터 중앙값 고정 정보
other_vars = [f for f in selected_features if f not in [x_var, y_var]]
fixed_values = {}

if other_vars:
    st.markdown("##### 📌 고정된 나머지 속성값 (테스트 데이터의 중앙값 기준)")
    fixed_text_list = []
    for var in other_vars:
        med_val = test_df[var].median()
        fixed_values[var] = med_val
        fixed_text_list.append(f"**{FEATURE_MAP[var]}**: `{med_val:.2f}`")
    st.write(", ".join(fixed_text_list))

# 격자 생성
x_min, x_max = test_df[x_var].min(), test_df[x_var].max()
y_min, y_max = test_df[y_var].min(), test_df[y_var].max()

# 패딩 추가
x_pad = (x_max - x_min) * 0.05 if x_max != x_min else 1.0
y_pad = (y_max - y_min) * 0.05 if y_max != y_min else 1.0

x_grid = np.linspace(x_min - x_pad, x_max + x_pad, 100)
y_grid = np.linspace(y_min - y_pad, y_max + y_pad, 100)
xx, yy = np.meshgrid(x_grid, y_grid)

# Grid 데이터 프레임 준비
grid_points = pd.DataFrame({
    x_var: xx.ravel(),
    y_var: yy.ravel()
})

for var in other_vars:
    grid_points[var] = fixed_values[var]

# feature 순서 일치시키기
grid_points = grid_points[selected_features]

# 의사결정트리 2D 예측 (영역 채우기용)
dt_grid_pred = tree_clf.predict(grid_points).reshape(xx.shape)

# 로지스틱 회귀 2D 확률 예측 (0.5 결정 선용)
grid_points_scaled = scaler.transform(grid_points)
lr_grid_prob = log_reg.predict_proba(grid_points_scaled)[:, 1].reshape(xx.shape)

# Plotly 시각화 생성
fig_scatter = px_go.Figure()

# 1) 의사결정트리 분할 영역 (옅은 색 채우기)
fig_scatter.add_trace(px_go.Heatmap(
    x=x_grid,
    y=y_grid,
    z=dt_grid_pred,
    colorscale=[[0, 'rgba(144, 238, 144, 0.25)'], [1, 'rgba(255, 182, 193, 0.35)']],
    showscale=False,
    hoverinfo='skip',
    name='의사결정트리 구분 영역'
))

# 2) 로지스틱 회귀 0.5 결정 경계선
has_lr_boundary = (lr_grid_prob.min() <= 0.5 <= lr_grid_prob.max())
if has_lr_boundary:
    fig_scatter.add_trace(px_go.Contour(
        x=x_grid,
        y=y_grid,
        z=lr_grid_prob,
        contours_start=0.5,
        contours_end=0.5,
        contours_coloring='lines',
        line=dict(color='blue', width=3, dash='dash'),
        showscale=False,
        name='로지스틱 회귀 경계 (0.5)'
    ))

# 3) 테스트 데이터 산점도
test_df_plot = test_df.copy()
test_df_plot['뇌졸중 여부'] = test_df_plot['stroke'].map({0: '정상 (0)', 1: '뇌졸중 (1)'})

for label, color, symbol in [('정상 (0)', '#2B6CB0', 'circle'), ('뇌졸중 (1)', '#E53E3E', 'x')]:
    sub_df = test_df_plot[test_df_plot['뇌졸중 여부'] == label]
    fig_scatter.add_trace(px_go.Scatter(
        x=sub_df[x_var],
        y=sub_df[y_var],
        mode='markers',
        name=f"실제 {label}",
        marker=dict(color=color, symbol=symbol, size=7, opacity=0.8)
    ))

fig_scatter.update_layout(
    title=f"테스트 데이터 산점도 및 모델 결정 경계 ({x_korean} vs {y_korean})",
    xaxis_title=x_korean,
    yaxis_title=y_korean,
    template="plotly_white",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

st.plotly_chart(fig_scatter, use_container_width=True)

if not has_lr_boundary:
    st.info("ℹ️ 로지스틱 회귀 모델이 0.5로 가르는 결정 경계선이 현재 선택한 시각화 범위 밖에 위치합니다.")

st.markdown("---")

# ---------------------------------------------------------
# 9. Graphviz 의사결정트리 구조 시각화
# ---------------------------------------------------------
st.subheader("🌳 의사결정트리 구조 (질문 가지)")

def build_dot_tree(tree, feature_names):
    tree_ = tree.tree_
    
    dot_lines = [
        'digraph Tree {',
        'node [shape=box, style="filled, rounded", color="black", fontname="helvetica"] ;',
        'edge [fontname="helvetica"] ;'
    ]
    
    def recurse(node_id):
        is_leaf = (tree_.children_left[node_id] == tree_.children_right[node_id])
        n_samples = tree_.n_node_samples[node_id]
        values = tree_.value[node_id][0]
        stroke_count = int(values[1]) if len(values) > 1 else 0
        ratio = (stroke_count / n_samples * 100) if n_samples > 0 else 0.0
        
        if is_leaf:
            pred_class = int(np.argmax(values))
            if pred_class == 1:
                fillcolor = '"#FFC0CB"'  # 뇌졸중 (양성) -> 옅은 분홍/빨강
                pred_label = "뇌졸중"
            else:
                fillcolor = '"#E0F2FE"'  # 정상 -> 옅은 파랑
                pred_label = "아님"
            
            label_text = f"결론: {pred_label}\\n사람 수: {n_samples:,}명\\n뇌졸중: {stroke_count:,}명 ({ratio:.1f}%)"
            dot_lines.append(f'{node_id} [label="{label_text}", fillcolor={fillcolor}] ;')
        else:
            feature_idx = tree_.feature[node_id]
            feature_name = FEATURE_MAP.get(feature_names[feature_idx], feature_names[feature_idx])
            threshold = tree_.threshold[node_id]
            
            label_text = f"{feature_name} <= {threshold:.2f}?\\n사람 수: {n_samples:,}명\\n뇌졸중: {stroke_count:,}명 ({ratio:.1f}%)"
            dot_lines.append(f'{node_id} [label="{label_text}", fillcolor="#FFFFFF"] ;')
            
            left_child = tree_.children_left[node_id]
            right_child = tree_.children_right[node_id]
            
            dot_lines.append(f'{node_id} -> {left_child} [label="예", angle=45, labeldistance=2.5] ;')
            dot_lines.append(f'{node_id} -> {right_child} [label="아니오", angle=-45, labeldistance=2.5] ;')
            
            recurse(left_child)
            recurse(right_child)
            
    recurse(0)
    dot_lines.append('}')
    return "\n".join(dot_lines)

dot_graph = build_dot_tree(tree_clf, selected_features)
st.graphviz_chart(dot_graph, use_container_width=True)

# 트리에 대한 요약 분석
tree_structure = tree_clf.tree_
leaf_nodes = [i for i in range(tree_structure.node_count) if tree_structure.children_left[i] == tree_structure.children_right[i]]
total_leaf_count = len(leaf_nodes)

no_stroke_leaf_count = 0
for leaf_id in leaf_nodes:
    vals = tree_structure.value[leaf_id][0]
    pred = int(np.argmax(vals))
    if pred == 0:
        no_stroke_leaf_count += 1

used_features_indices = set(tree_structure.feature[tree_structure.feature >= 0])
used_korean_features = [FEATURE_MAP[selected_features[i]] for i in used_features_indices]

st.markdown("##### 📌 의사결정트리 분석 요약")
st.write(f"- 답을 내는 최종 마디(리프 노드)는 총 **{total_leaf_count}칸** 이며, 그 중 **{no_stroke_leaf_count}칸** 에서 **'아님(정상)'**으로 답합니다.")
st.write(f"- 선택된 속성 중 나무가 실제로 분할 질문으로 사용한 속성은 **{', '.join(used_korean_features)}** 입니다.")

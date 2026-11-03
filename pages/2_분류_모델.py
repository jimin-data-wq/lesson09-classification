import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

APP_TITLE = "뇌졸중 예측 실습실"
PAGE_TITLE = "분류 모델"
PAGE_ICON = "🤖"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

# 브라우저 탭 제목과 아이콘 (반드시 첫 번째 streamlit 명령이어야 함)
st.set_page_config(page_title=f"{PAGE_TITLE} - {APP_TITLE}", page_icon=PAGE_ICON, layout="wide")

# 열 이름 -> 우리말 이름
KOR = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
ALL_FEATURES = list(KOR.keys())
DEFAULT_FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]  # bmi는 뺌
BINARY_FEATURES = {"hypertension", "heart_disease"}

# 모델 이름 (교과서의 이름을 앞에)
LOGI_NAME = "로지스틱 회귀"
LOGI_DESC = "확률로 답하는 모델"
TREE_NAME = "의사결정트리"
TREE_DESC = "질문으로 답하는 모델"
BASE_NAME = "기준 모델"
BASE_DESC = "입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델"

COLOR_NEG = "#4C78A8"  # 뇌졸중 아님
COLOR_POS = "#E45756"  # 뇌졸중
REGION_NEG = "rgba(76,120,168,0.15)"
REGION_POS = "rgba(228,87,86,0.20)"
LEAF_FILL_NEG = "#D0DEEF"
LEAF_FILL_POS = "#F8D0CF"

GRID_N = 100  # 그림을 칠할 격자 크기


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def pct(x):
    return f"{x * 100:.2f}%"


def accuracy_card(title, desc, train_acc, test_acc):
    st.markdown(
        f"""
        <div style="border:1px solid rgba(128,128,128,0.35); border-radius:14px;
                    padding:18px 12px; text-align:center;
                    background:rgba(128,128,128,0.08); min-height:230px;">
            <div style="font-size:1.1rem; font-weight:700;">{title}</div>
            <div style="font-size:0.8rem; opacity:0.75; min-height:2.4em;">({desc})</div>
            <div style="font-size:2.8rem; font-weight:700; line-height:1.3;">{pct(test_acc)}</div>
            <div style="font-size:0.85rem; opacity:0.75;">테스트 데이터 정확도</div>
            <div style="display:flex; justify-content:space-around; margin-top:12px;
                        font-size:0.8rem; opacity:0.85;">
                <div>훈련 데이터<br><b>{pct(train_acc)}</b></div>
                <div>테스트 데이터<br><b>{pct(test_acc)}</b></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_tree_dot(clf, features, x_train, y_train):
    """의사결정트리를 DOT 문자열로 바꾼다. 마디마다 훈련용 사람 수, 뇌졸중 수, 비율을 적는다."""
    t = clf.tree_
    path = clf.decision_path(x_train)
    member = pd.DataFrame(path.toarray(), index=x_train.index).astype(bool)
    n_all = member.sum()
    n_pos = member[y_train == 1].sum()

    lines = [
        "digraph Tree {",
        '  graph [rankdir=TB];',
        '  node [shape=box, style="rounded,filled", fillcolor="#FFFFFF", color="#888888", fontname="Helvetica"];',
        '  edge [fontname="Helvetica"];',
    ]
    leaf_answers = {}
    for i in range(t.node_count):
        n = int(n_all.loc[i])
        k = int(n_pos.loc[i])
        rate = k / n * 100 if n else 0.0
        stats = f"훈련용 {n:,}명\\n실제 뇌졸중 {k:,}명 ({rate:.1f}%)"
        is_leaf = int(t.children_left[i]) == -1
        if is_leaf:
            answer_pos = k * 2 > n  # 같으면 0(아님)
            leaf_answers[i] = answer_pos
            answer = "뇌졸중" if answer_pos else "아님"
            fill = LEAF_FILL_POS if answer_pos else LEAF_FILL_NEG
            label = f"답: {answer}\\n{stats}"
            lines.append(f'  n{i} [label="{label}", fillcolor="{fill}"];')
        else:
            feat = features[int(t.feature[i])]
            name = KOR[feat]
            if feat in BINARY_FEATURES:
                question = f"{name} 없음?"
            else:
                question = f"{name} ≤ {float(t.threshold[i]):.4g}?"
            label = f"{question}\\n{stats}"
            lines.append(f'  n{i} [label="{label}"];')
    for i in range(t.node_count):
        if int(t.children_left[i]) != -1:
            lines.append(f'  n{i} -> n{int(t.children_left[i])} [label="예"];')
            lines.append(f'  n{i} -> n{int(t.children_right[i])} [label="아니요"];')
    lines.append("}")
    return "\n".join(lines), leaf_answers


# ---------- 화면 ----------
st.title(f"{PAGE_ICON} {PAGE_TITLE}")
st.caption(f"{APP_TITLE}: 뇌졸중(stroke=1)을 양성으로 두고, 두 가지 모델로 뇌졸중을 맞혀 봐요.")

try:
    raw = load_data()
except Exception as e:  # 네트워크 오류 등
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n{e}")
    st.stop()

# ---------- 입력 속성 고르기 ----------
selected = st.multiselect(
    "입력으로 사용할 속성",
    options=ALL_FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda c: KOR[c],
)
features = [f for f in ALL_FEATURES if f in selected]

if len(features) < 2:
    st.info("입력으로 사용할 속성을 두 개 이상 골라 주세요. 그래야 모델을 만들고 그림을 그릴 수 있어요.")
    st.stop()

# ---------- 데이터 나누기 ----------
# 번호(id) 순으로 정렬 -> 열 명씩 묶기 -> 각 묶음의 앞 세 명은 테스트용으로 고정
data = raw.sort_values("id").reset_index(drop=True)
is_test = (data.index % 10) < 3
train_df = data[~is_test].copy()
test_df = data[is_test].copy()

x_train = train_df[features].copy()
x_test = test_df[features].copy()
y_train = train_df["stroke"]
y_test = test_df["stroke"]

st.caption(
    f"번호(id) 순으로 정렬해 열 명씩 묶고, 각 묶음의 앞 세 명을 테스트용으로 고정했어요. "
    f"테스트용 {len(test_df):,}명, 학습용 {len(train_df):,}명이에요. "
    f"크기 맞추기와 학습은 학습용 사람들로만 했어요."
)

# 체질량지수는 고른 경우에만, 훈련용의 중앙값으로 채움
if "bmi" in features:
    bmi_median = float(x_train["bmi"].median())
    n_fill_train = int(x_train["bmi"].isna().sum())
    n_fill_test = int(x_test["bmi"].isna().sum())
    x_train["bmi"] = x_train["bmi"].fillna(bmi_median)
    x_test["bmi"] = x_test["bmi"].fillna(bmi_median)
    st.caption(
        f"체질량지수의 빈 값은 훈련용의 중앙값 {bmi_median:.2f}로 채웠어요 "
        f"(학습용 {n_fill_train:,}명, 테스트용 {n_fill_test:,}명)."
    )

# ---------- 모델 만들기 ----------
logi = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ]
).fit(x_train, y_train)

tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(x_train, y_train)

base = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)

# ---------- 1) 정확도 카드 ----------
st.subheader("1. 모델 정확도")
c1, c2, c3 = st.columns(3)
with c1:
    accuracy_card(LOGI_NAME, LOGI_DESC, logi.score(x_train, y_train), logi.score(x_test, y_test))
with c2:
    accuracy_card(TREE_NAME, TREE_DESC, tree.score(x_train, y_train), tree.score(x_test, y_test))
with c3:
    accuracy_card(BASE_NAME, BASE_DESC, base.score(x_train, y_train), base.score(x_test, y_test))
st.caption(
    "정확도는 전체 사람 중에서 맞힌 사람의 비율이에요. "
    "뇌졸중인 사람이 드물기 때문에, 기준 모델의 정확도와 견주어 두 모델이 더 나은지 살펴보세요."
)

# ---------- 2) 산점도 + 경계 ----------
st.subheader("2. 두 속성으로 본 테스트 데이터와 모델의 경계")
s1, s2 = st.columns(2)
with s1:
    x_key = st.selectbox("가로축", options=features, index=0, format_func=lambda c: KOR[c])
with s2:
    y_key = st.selectbox("세로축", options=features, index=1, format_func=lambda c: KOR[c])

if x_key == y_key:
    st.warning("가로축과 세로축에는 서로 다른 속성을 골라 주세요.")
else:
    # 두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 둠
    medians = x_test.median()
    others = [f for f in features if f not in (x_key, y_key)]

    def padded_range(col):
        lo, hi = float(x_test[col].min()), float(x_test[col].max())
        span = hi - lo
        pad = span * 0.05 if span > 0 else 1.0
        return lo - pad, hi + pad

    x0, x1 = padded_range(x_key)
    y0, y1 = padded_range(y_key)
    xs = [x0 + (x1 - x0) * i / (GRID_N - 1) for i in range(GRID_N)]
    ys = [y0 + (y1 - y0) * i / (GRID_N - 1) for i in range(GRID_N)]

    grid = pd.MultiIndex.from_product([ys, xs], names=[y_key, x_key]).to_frame(index=False)
    for f in others:
        grid[f] = float(medians[f])
    grid = grid[features]

    prob = logi.predict_proba(grid)[:, 1]  # 뇌졸중(양성)일 확률
    tree_pred = tree.predict(grid)
    prob_min, prob_max = float(prob.min()), float(prob.max())
    line_visible = prob_min <= 0.5 <= prob_max

    fig = go.Figure()

    # 의사결정트리가 나눈 칸 (옅은 색)
    fig.add_trace(
        go.Heatmap(
            x=xs,
            y=ys,
            z=tree_pred.reshape(GRID_N, GRID_N),
            zmin=0,
            zmax=1,
            colorscale=[[0, REGION_NEG], [1, REGION_POS]],
            showscale=False,
            hoverinfo="skip",
        )
    )

    # 로지스틱 회귀가 0.5로 가르는 선
    if line_visible:
        fig.add_trace(
            go.Contour(
                x=xs,
                y=ys,
                z=prob.reshape(GRID_N, GRID_N),
                contours=dict(start=0.5, end=0.5, size=0.1, coloring="lines"),
                line=dict(width=3),
                colorscale=[[0, "black"], [1, "black"]],
                showscale=False,
                hoverinfo="skip",
            )
        )

    # 테스트 데이터 점 (아님을 먼저, 뇌졸중을 위에)
    for value, label, color in [(0, "실제: 뇌졸중 아님", COLOR_NEG), (1, "실제: 뇌졸중", COLOR_POS)]:
        part = x_test[y_test == value]
        fig.add_trace(
            go.Scatter(
                x=part[x_key],
                y=part[y_key],
                mode="markers",
                name=label,
                marker=dict(color=color, size=7, opacity=0.65, line=dict(width=0.5, color="white")),
            )
        )

    # 범례용 안내 항목 (선과 칸)
    if line_visible:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name=f"{LOGI_NAME}이 0.5로 가르는 선",
                line=dict(color="black", width=3),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=f"{TREE_NAME}: '아님'이라 답하는 칸",
            marker=dict(symbol="square", size=14, color=REGION_NEG, line=dict(width=1, color=COLOR_NEG)),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=f"{TREE_NAME}: '뇌졸중'이라 답하는 칸",
            marker=dict(symbol="square", size=14, color=REGION_POS, line=dict(width=1, color=COLOR_POS)),
        )
    )

    fig.update_xaxes(title=KOR[x_key], range=[x0, x1])
    fig.update_yaxes(title=KOR[y_key], range=[y0, y1])
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=560,
        margin=dict(t=90),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 설명 글
    if others:
        fixed = ", ".join(f"{KOR[f]} {round(float(medians[f]), 2):g}" for f in others)
        st.write(f"두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: **{fixed}**")
    else:
        st.write("고른 속성이 두 개뿐이라, 따로 세워 둘 속성은 없어요.")

    if line_visible:
        st.write(
            f"검은 선은 {LOGI_NAME}({LOGI_DESC})이 뇌졸중일 확률을 0.5로 보는 자리예요."
        )
    elif prob_max < 0.5:
        st.write(
            f"{LOGI_NAME}이 0.5로 가르는 선은 이 그림 밖에 있어요. "
            f"그림 안 어디에서나 뇌졸중일 확률이 0.5보다 낮아서(가장 높은 곳이 {prob_max:.3f}) "
            f"모두 '아님'이라고 답해요."
        )
    else:
        st.write(
            f"{LOGI_NAME}이 0.5로 가르는 선은 이 그림 밖에 있어요. "
            f"그림 안 어디에서나 뇌졸중일 확률이 0.5보다 높아서(가장 낮은 곳이 {prob_min:.3f}) "
            f"모두 '뇌졸중'이라고 답해요."
        )
    st.caption(
        "옅게 칠한 칸은 의사결정트리의 답이에요. 파란 칸은 '아님', 붉은 칸은 '뇌졸중'이라고 답하는 곳이에요."
    )

# ---------- 3) 의사결정트리 가지 그림 ----------
st.subheader(f"3. {TREE_NAME}({TREE_DESC})가 던진 질문")
dot, leaf_answers = build_tree_dot(tree, features, x_train, y_train)
st.graphviz_chart(dot, use_container_width=True)
st.caption(
    "각 마디에는 그 자리에 온 훈련용 사람 수와, 그중 실제 뇌졸중인 사람 수·비율을 적었어요. "
    "붉은 마디는 '뇌졸중', 푸른 마디는 '아님'이라고 답하는 마디예요. "
    "의사결정트리는 크기를 맞추지 않은 원래 값으로 학습해서, 질문의 기준값을 그대로 읽을 수 있어요."
)

n_leaf = len(leaf_answers)
n_leaf_neg = sum(1 for v in leaf_answers.values() if not v)
st.write(f"답을 내는 마디(칸)는 모두 **{n_leaf}칸**이고, 그중 **{n_leaf_neg}칸**이 '아님'이라고 답해요.")

t = tree.tree_
asked_idx = {int(t.feature[i]) for i in range(t.node_count) if int(t.children_left[i]) != -1}
asked = [KOR[f] for j, f in enumerate(features) if j in asked_idx]
not_asked = [KOR[f] for j, f in enumerate(features) if j not in asked_idx]
asked_text = ", ".join(asked) if asked else "아무것도 묻지 않았어요"
not_asked_text = ", ".join(not_asked) if not_asked else "없음"
st.write(f"고른 속성 가운데 이 나무가 실제로 물은 것: **{asked_text}** (묻지 않은 것: {not_asked_text})")

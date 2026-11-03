import math
from itertools import product

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

APP_TITLE = "뇌졸중 예측 실습실"
PAGE_TITLE = "성능 높이기"
PAGE_ICON = "🚀"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

# 브라우저 탭 제목과 아이콘 (반드시 첫 번째 streamlit 명령이어야 함)
st.set_page_config(page_title=f"{PAGE_TITLE} - {APP_TITLE}", page_icon=PAGE_ICON, layout="wide")

# 열 이름 -> 우리말 이름 ('분류 모델' 페이지와 같음)
KOR = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
ALL_FEATURES = list(KOR.keys())
DEFAULT_FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]  # bmi는 뺌

# 모델 이름 (교과서의 이름을 앞에)
LOGI_LABEL = "로지스틱 회귀(확률로 답하는 모델)"
TREE_LABEL = "의사결정트리(질문으로 답하는 모델)"

BEFORE = "가중치 맞추기 전"
AFTER = "가중치 맞춘 뒤"

COLOR_NEG = "#4C78A8"  # 실제 뇌졸중 아님
COLOR_POS = "#E45756"  # 실제 뇌졸중
LINE_BEFORE = "#F28E2B"  # 맞추기 전 0.5 경계선
LINE_AFTER = "#59A14F"  # 맞춘 뒤 0.5 경계선

NA_TEXT = "계산할 수 없음"


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


# ---------- 채점 도우미 ('채점표' 페이지와 같음) ----------
def confusion_counts(y_true, y_pred):
    """뇌졸중(1)을 양성으로 두고 TP, FN, FP, TN 사람 수를 센다."""
    actual_pos = y_true == 1
    pred_pos = y_pred == 1
    return {
        "TP": int((actual_pos & pred_pos).sum()),
        "FN": int((actual_pos & ~pred_pos).sum()),
        "FP": int((~actual_pos & pred_pos).sum()),
        "TN": int((~actual_pos & ~pred_pos).sum()),
    }


def metrics_from(c):
    """네 칸에서 정확도·재현율·정밀도·F1을 구한다. 계산할 수 없으면 None."""
    tp, fn, fp, tn = c["TP"], c["FN"], c["FP"], c["TN"]
    total = tp + fn + fp + tn
    accuracy = (tp + tn) / total if total else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None  # 분모: 실제 뇌졸중인 사람 수
    precision = tp / (tp + fp) if (tp + fp) > 0 else None  # 분모: 뇌졸중이라 예측한 사람 수
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return accuracy, recall, precision, f1


def show_pct(x):
    return NA_TEXT if x is None else f"{x * 100:.2f}%"


def compare_table(before_counts, after_counts):
    """네 칸과 네 지표를 전·후로 나란히 놓은 표."""
    b_acc, b_rec, b_pre, b_f1 = metrics_from(before_counts)
    a_acc, a_rec, a_pre, a_f1 = metrics_from(after_counts)
    rows = [
        ("찾아낸 환자(TP)", f"{before_counts['TP']:,}명", f"{after_counts['TP']:,}명"),
        ("놓친 환자(FN)", f"{before_counts['FN']:,}명", f"{after_counts['FN']:,}명"),
        ("헛짚은 사람(FP)", f"{before_counts['FP']:,}명", f"{after_counts['FP']:,}명"),
        ("아니라고 맞힌 사람(TN)", f"{before_counts['TN']:,}명", f"{after_counts['TN']:,}명"),
        ("정확도", show_pct(b_acc), show_pct(a_acc)),
        ("재현율", show_pct(b_rec), show_pct(a_rec)),
        ("정밀도", show_pct(b_pre), show_pct(a_pre)),
        ("F1", show_pct(b_f1), show_pct(a_f1)),
    ]
    return pd.DataFrame(rows, columns=["", BEFORE, AFTER])


# ---------- 0.5 경계선(면) 도우미 ----------
def logit_form(pipe, features):
    """로지스틱 회귀의 '확률 0.5' 조건을 원래 값에 대한 식 (계수들 · x + 상수 = 0) 으로 바꾼다."""
    scaler = pipe.named_steps["scaler"]
    model = pipe.named_steps["model"]
    coefs = {}
    const = float(model.intercept_[0])
    for j, f in enumerate(features):
        a = float(model.coef_[0][j]) / float(scaler.scale_[j])
        coefs[f] = a
        const -= a * float(scaler.mean_[j])
    return coefs, const


def box_crossing(coefs, const, lows, highs):
    """식 (계수들 · x + 상수 = 0) 이 상자 [lows, highs] 의 모서리와 만나는 점들을 구한다."""
    d = len(coefs)
    found = {}
    for corner in product([0, 1], repeat=d):
        for dim in range(d):
            if corner[dim] != 0:
                continue
            p1 = [lows[m] if corner[m] == 0 else highs[m] for m in range(d)]
            p2 = list(p1)
            p2[dim] = highs[dim]
            f1 = const + sum(c * v for c, v in zip(coefs, p1))
            f2 = const + sum(c * v for c, v in zip(coefs, p2))
            candidates = []
            if f1 == 0:
                candidates.append(p1)
            if f2 == 0:
                candidates.append(p2)
            if f1 * f2 < 0:
                t = f1 / (f1 - f2)
                candidates.append([p1[m] + t * (p2[m] - p1[m]) for m in range(d)])
            for p in candidates:
                found[tuple(round(v, 9) for v in p)] = p
    return list(found.values())


def outside_side(coefs, const, lows, highs):
    """경계가 상자 밖일 때, 상자 안 모든 곳이 0.5보다 높은지 낮은지."""
    center = [(lo + hi) / 2 for lo, hi in zip(lows, highs)]
    value = const + sum(c * v for c, v in zip(coefs, center))
    return "높아서" if value > 0 else "낮아서"


def cross(u, v):
    return [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]


def dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def plane_mesh(points, normal, color):
    """3차원 점들로 이루어진 볼록 다각형을 반투명한 면으로 만든다."""
    center = [sum(p[m] for p in points) / len(points) for m in range(3)]
    smallest = min(range(3), key=lambda m: abs(normal[m]))
    helper = [0.0, 0.0, 0.0]
    helper[smallest] = 1.0
    u = cross(normal, helper)
    norm_u = math.sqrt(dot(u, u))
    u = [a / norm_u for a in u]
    v = cross(normal, u)
    ordered = sorted(
        points,
        key=lambda p: math.atan2(
            dot([p[m] - center[m] for m in range(3)], v),
            dot([p[m] - center[m] for m in range(3)], u),
        ),
    )
    n = len(ordered)
    return go.Mesh3d(
        x=[p[0] for p in ordered],
        y=[p[1] for p in ordered],
        z=[p[2] for p in ordered],
        i=[0] * (n - 2),
        j=list(range(1, n - 1)),
        k=list(range(2, n)),
        color=color,
        opacity=0.45,
        flatshading=True,
        hoverinfo="skip",
        showlegend=False,
    )


def padded_range(series):
    lo, hi = float(series.min()), float(series.max())
    span = hi - lo
    pad = span * 0.05 if span > 0 else 1.0
    return lo - pad, hi + pad


def fmt_value(v):
    return f"{round(float(v), 2):g}"


# ---------- 화면 ----------
st.title(f"{PAGE_ICON} {PAGE_TITLE}")
st.caption(
    f"{APP_TITLE}: 뇌졸중인 사람이 아주 드물어서, 양쪽의 가중치를 맞추기 전과 맞춘 뒤의 결과를 견주어 봐요. "
    "뇌졸중(stroke=1)을 양성으로 둬요."
)

try:
    raw = load_data()
except Exception as e:  # 네트워크 오류 등
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n{e}")
    st.stop()

# ---------- 입력 속성 고르기 ('분류 모델' 페이지와 같은 방법) ----------
selected = st.multiselect(
    "입력으로 사용할 속성",
    options=ALL_FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda c: KOR[c],
)
features = [f for f in ALL_FEATURES if f in selected]

if len(features) < 2:
    st.info("입력으로 사용할 속성을 두 개 이상 골라 주세요. 그래야 모델을 만들 수 있어요.")
    st.stop()

# ---------- 데이터 나누기 ('채점표' 페이지와 같은 방법) ----------
# 번호(id) 순으로 정렬 -> 열 명씩 묶기 -> 각 묶음의 앞 세 명은 테스트용으로 고정
data = raw.sort_values("id").reset_index(drop=True)
is_test = (data.index % 10) < 3
train_df = data[~is_test].copy()
test_df = data[is_test].copy()

x_train = train_df[features].copy()
x_test = test_df[features].copy()
y_train = train_df["stroke"]
y_test = test_df["stroke"]

# 체질량지수는 고른 경우에만, 훈련용의 중앙값으로 채움
if "bmi" in features:
    bmi_median = float(x_train["bmi"].median())
    x_train["bmi"] = x_train["bmi"].fillna(bmi_median)
    x_test["bmi"] = x_test["bmi"].fillna(bmi_median)

n_train_pos = int((y_train == 1).sum())
n_test_pos = int((y_test == 1).sum())
st.write(
    f"학습용 {len(train_df):,}명 중 뇌졸중인 사람은 **{n_train_pos:,}명**"
    f"({n_train_pos / len(train_df) * 100:.2f}%), "
    f"테스트용 {len(test_df):,}명 중 뇌졸중인 사람은 **{n_test_pos:,}명**이에요."
)
st.caption(
    "가중치를 맞춘다는 것은, 훈련할 때 뇌졸중인 사람 한 명을 더 무겁게 세어서 "
    "뇌졸중인 쪽과 아닌 쪽의 무게 합이 같아지게 하는 방법이에요 (class_weight='balanced')."
)


# ---------- 모델 만들기 (가중치 맞추기 전·후) ----------
def make_logistic(class_weight):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weight)),
        ]
    ).fit(x_train, y_train)


def make_tree(class_weight):
    return DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=5, random_state=42, class_weight=class_weight
    ).fit(x_train, y_train)


logi_before = make_logistic(None)
logi_after = make_logistic("balanced")
tree_before = make_tree(None)
tree_after = make_tree("balanced")

# ---------- 1) 전·후 비교표 ----------
st.subheader("1. 가중치를 맞추기 전과 맞춘 뒤")


def counts_of(model):
    return confusion_counts(y_test, pd.Series(model.predict(x_test), index=x_test.index))


def change_line(before, after):
    return (
        f"찾아낸 환자 {before['TP']:,}명 → {after['TP']:,}명, "
        f"놓친 환자 {before['FN']:,}명 → {after['FN']:,}명, "
        f"헛짚은 사람 {before['FP']:,}명 → {after['FP']:,}명으로 바뀌었어요."
    )


left, right = st.columns(2)
for col, label, m_before, m_after in [
    (left, LOGI_LABEL, logi_before, logi_after),
    (right, TREE_LABEL, tree_before, tree_after),
]:
    cb, ca = counts_of(m_before), counts_of(m_after)
    with col:
        st.markdown(f"**{label}**")
        st.dataframe(compare_table(cb, ca), hide_index=True, use_container_width=True)
        st.write(change_line(cb, ca))

st.caption(
    f"나누는 수가 0이면 '{NA_TEXT}'이라고 적어요. "
    "정확도 = (TP+TN) ÷ 전체 · 재현율 = TP ÷ (TP+FN) · 정밀도 = TP ÷ (TP+FP) · "
    "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율)"
)

# 로지스틱 회귀의 0.5 경계를 식으로 꺼내 둠
coefs_before, const_before = logit_form(logi_before, features)
coefs_after, const_after = logit_form(logi_after, features)
medians = x_test.median()

# ---------- 2) 2차원 산점도 + 0.5 경계선 ----------
st.subheader(f"2. 두 속성으로 본 {LOGI_LABEL}의 0.5 경계선")
s1, s2 = st.columns(2)
with s1:
    x_key = st.selectbox("가로축", options=features, index=0, format_func=lambda c: KOR[c], key="axis2d_x")
with s2:
    y_key = st.selectbox("세로축", options=features, index=1, format_func=lambda c: KOR[c], key="axis2d_y")

if x_key == y_key:
    st.warning("가로축과 세로축에는 서로 다른 속성을 골라 주세요.")
else:
    others = [f for f in features if f not in (x_key, y_key)]
    x0, x1 = padded_range(x_test[x_key])
    y0, y1 = padded_range(x_test[y_key])
    lows, highs = [x0, y0], [x1, y1]

    fig = go.Figure()
    for value, label, color in [(0, "실제: 뇌졸중 아님", COLOR_NEG), (1, "실제: 뇌졸중", COLOR_POS)]:
        part = x_test[y_test == value]
        fig.add_trace(
            go.Scatter(
                x=part[x_key],
                y=part[y_key],
                mode="markers",
                name=label,
                marker=dict(color=color, size=7, opacity=0.6, line=dict(width=0.5, color="white")),
            )
        )

    messages = []
    for name, coefs, const, color in [
        (BEFORE, coefs_before, const_before, LINE_BEFORE),
        (AFTER, coefs_after, const_after, LINE_AFTER),
    ]:
        two = [coefs[x_key], coefs[y_key]]
        c_fixed = const + sum(coefs[m] * float(medians[m]) for m in others)
        pts = box_crossing(two, c_fixed, lows, highs)
        if len(pts) >= 2:
            pts = sorted(pts, key=lambda p: (p[0], p[1]))
            fig.add_trace(
                go.Scatter(
                    x=[pts[0][0], pts[-1][0]],
                    y=[pts[0][1], pts[-1][1]],
                    mode="lines",
                    name=f"{name} 0.5 경계선",
                    line=dict(color=color, width=4),
                )
            )
        else:
            side = outside_side(two, c_fixed, lows, highs)
            answer = "뇌졸중" if side == "높아서" else "아님"
            messages.append(
                f"**{name}**의 0.5 경계선은 이 그림 밖에 있어요. "
                f"그림 안 어디에서나 뇌졸중일 확률이 0.5보다 {side} 모두 '{answer}'이라고 답해요."
            )

    fig.update_xaxes(title=KOR[x_key], range=[x0, x1])
    fig.update_yaxes(title=KOR[y_key], range=[y0, y1])
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=560,
        margin=dict(t=90),
    )
    st.plotly_chart(fig, use_container_width=True)

    if others:
        fixed = ", ".join(f"{KOR[f]} {fmt_value(medians[f])}" for f in others)
        st.write(f"두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: **{fixed}**")
    else:
        st.write("고른 속성이 두 개뿐이라, 따로 세워 둘 속성은 없어요.")
    for m in messages:
        st.write(m)
    if not messages:
        st.write("주황색 선은 가중치를 맞추기 전, 초록색 선은 맞춘 뒤의 0.5 경계선이에요.")

# ---------- 3) 3차원 산점도 + 0.5 평면 ----------
if len(features) >= 3:
    st.subheader(f"3. 세 속성으로 본 {LOGI_LABEL}의 0.5 평면")
    t1, t2, t3 = st.columns(3)
    with t1:
        ax_key = st.selectbox("가로축", options=features, index=0, format_func=lambda c: KOR[c], key="axis3d_x")
    with t2:
        ay_key = st.selectbox("세로축", options=features, index=1, format_func=lambda c: KOR[c], key="axis3d_y")
    with t3:
        az_key = st.selectbox("높이축", options=features, index=2, format_func=lambda c: KOR[c], key="axis3d_z")

    axes3 = [ax_key, ay_key, az_key]
    if len(set(axes3)) < 3:
        st.warning("세 축에는 서로 다른 속성을 골라 주세요.")
    else:
        others3 = [f for f in features if f not in axes3]
        ranges = [padded_range(x_test[k]) for k in axes3]
        lows3 = [r[0] for r in ranges]
        highs3 = [r[1] for r in ranges]

        fig3 = go.Figure()
        for value, label, color in [(0, "실제: 뇌졸중 아님", COLOR_NEG), (1, "실제: 뇌졸중", COLOR_POS)]:
            part = x_test[y_test == value]
            fig3.add_trace(
                go.Scatter3d(
                    x=part[ax_key],
                    y=part[ay_key],
                    z=part[az_key],
                    mode="markers",
                    name=label,
                    marker=dict(color=color, size=3, opacity=0.7),
                )
            )

        messages3 = []
        for name, coefs, const, color in [
            (BEFORE, coefs_before, const_before, LINE_BEFORE),
            (AFTER, coefs_after, const_after, LINE_AFTER),
        ]:
            three = [coefs[k] for k in axes3]
            c_fixed = const + sum(coefs[m] * float(medians[m]) for m in others3)
            pts = box_crossing(three, c_fixed, lows3, highs3)
            if len(pts) >= 3:
                fig3.add_trace(plane_mesh(pts, three, color))
                # 범례용 안내 항목
                fig3.add_trace(
                    go.Scatter3d(
                        x=[None],
                        y=[None],
                        z=[None],
                        mode="lines",
                        name=f"{name} 0.5 평면",
                        line=dict(color=color, width=8),
                    )
                )
            else:
                side = outside_side(three, c_fixed, lows3, highs3)
                answer = "뇌졸중" if side == "높아서" else "아님"
                messages3.append(
                    f"**{name}**의 0.5 평면은 이 그림 밖에 있어요. "
                    f"그림 안 어디에서나 뇌졸중일 확률이 0.5보다 {side} 모두 '{answer}'이라고 답해요."
                )

        fig3.update_layout(
            scene=dict(
                xaxis=dict(title=KOR[ax_key], range=[lows3[0], highs3[0]]),
                yaxis=dict(title=KOR[ay_key], range=[lows3[1], highs3[1]]),
                zaxis=dict(title=KOR[az_key], range=[lows3[2], highs3[2]]),
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            height=640,
            margin=dict(t=60, l=0, r=0, b=0),
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("마우스로 끌어서 돌려 보고, 휠로 크기를 바꿀 수 있어요.")

        if others3:
            fixed3 = ", ".join(f"{KOR[f]} {fmt_value(medians[f])}" for f in others3)
            st.write(f"세 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: **{fixed3}**")
        for m in messages3:
            st.write(m)
        if not messages3:
            st.write("주황색 면은 가중치를 맞추기 전, 초록색 면은 맞춘 뒤의 0.5 평면이에요.")
else:
    st.caption("속성을 셋 이상 고르면 3차원 그림과 0.5 평면도 볼 수 있어요.")

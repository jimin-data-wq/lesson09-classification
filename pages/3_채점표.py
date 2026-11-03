import pandas as pd
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

APP_TITLE = "뇌졸중 예측 실습실"
PAGE_TITLE = "채점표"
PAGE_ICON = "📝"
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
BASE_LABEL = "기준 모델(훈련용에서 많은 쪽으로만 답하는 모델)"

NA_TEXT = "계산할 수 없음"
MAX_ROWS = 20


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


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


def confusion_html(c):
    """행은 실제, 열은 예측인 2×2 혼동행렬 (합계 포함)."""
    tp, fn, fp, tn = c["TP"], c["FN"], c["FP"], c["TN"]
    row_pos, row_neg = tp + fn, fp + tn
    col_pos, col_neg = tp + fp, fn + tn
    total = tp + fn + fp + tn

    good = "background:rgba(46,160,67,0.18);"
    bad = "background:rgba(228,87,86,0.18);"
    head = "background:rgba(128,128,128,0.15); font-weight:700;"
    sums = "background:rgba(128,128,128,0.08);"

    def cell(abbr, value, style):
        return (
            f'<td style="{style} text-align:center; padding:10px 8px;">'
            f'<div style="font-size:0.8rem; opacity:0.75;">{abbr}</div>'
            f'<div style="font-size:1.5rem; font-weight:700;">{value:,}명</div></td>'
        )

    def plain(value, style):
        return f'<td style="{style} text-align:center; padding:10px 8px; font-weight:600;">{value:,}명</td>'

    th = "text-align:center; padding:10px 8px;"
    return f"""
    <table style="width:100%; border-collapse:collapse; border:1px solid rgba(128,128,128,0.4);">
      <tr>
        <td style="{head} {th}">실제 ＼ 예측</td>
        <td style="{head} {th}">뇌졸중이라 예측</td>
        <td style="{head} {th}">아니라고 예측</td>
        <td style="{head} {th}">합계</td>
      </tr>
      <tr>
        <td style="{head} {th}">실제 뇌졸중</td>
        {cell("TP", tp, good)}
        {cell("FN", fn, bad)}
        {plain(row_pos, sums)}
      </tr>
      <tr>
        <td style="{head} {th}">실제 아님</td>
        {cell("FP", fp, bad)}
        {cell("TN", tn, good)}
        {plain(row_neg, sums)}
      </tr>
      <tr>
        <td style="{head} {th}">합계</td>
        {plain(col_pos, sums)}
        {plain(col_neg, sums)}
        {plain(total, sums)}
      </tr>
    </table>
    """


def person_list(rows):
    """번호, 나이, 평균 혈당, 고혈압 여부, 심장병 여부를 나이가 많은 순으로 정리한다."""
    out = rows[["id", "age", "avg_glucose_level", "hypertension", "heart_disease"]].copy()
    out = out.sort_values(["age", "id"], ascending=[False, True])
    out["hypertension"] = out["hypertension"].map({0: "없음", 1: "있음"})
    out["heart_disease"] = out["heart_disease"].map({0: "없음", 1: "있음"})
    out.columns = ["번호", "나이", "평균 혈당", "고혈압 여부", "심장병 여부"]
    return out


# ---------- 화면 ----------
st.title(f"{PAGE_ICON} {PAGE_TITLE}")
st.caption(f"{APP_TITLE}: 두 모델이 테스트 데이터에서 얼마나 맞혔는지 채점해 봐요. 뇌졸중(stroke=1)을 양성으로 둬요.")

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

# 체질량지수는 고른 경우에만, 훈련용의 중앙값으로 채움
if "bmi" in features:
    bmi_median = float(x_train["bmi"].median())
    x_train["bmi"] = x_train["bmi"].fillna(bmi_median)
    x_test["bmi"] = x_test["bmi"].fillna(bmi_median)

# ---------- 모델 만들기 ----------
logi = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ]
).fit(x_train, y_train)

tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(x_train, y_train)

base = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)

preds = {
    LOGI_LABEL: pd.Series(logi.predict(x_test), index=x_test.index),
    TREE_LABEL: pd.Series(tree.predict(x_test), index=x_test.index),
    BASE_LABEL: pd.Series(base.predict(x_test), index=x_test.index),
}
counts = {name: confusion_counts(y_test, pred) for name, pred in preds.items()}

n_test = len(test_df)
n_test_pos = int((y_test == 1).sum())
st.write(f"테스트용은 **{n_test:,}명**이고, 그중 뇌졸중인 사람은 **{n_test_pos:,}명**이에요.")

# ---------- 1) 혼동행렬 ----------
st.subheader("1. 혼동행렬 (행은 실제, 열은 예측)")
m1, m2 = st.columns(2)
with m1:
    st.markdown(f"**{LOGI_LABEL}**")
    st.markdown(confusion_html(counts[LOGI_LABEL]), unsafe_allow_html=True)
with m2:
    st.markdown(f"**{TREE_LABEL}**")
    st.markdown(confusion_html(counts[TREE_LABEL]), unsafe_allow_html=True)
st.caption(
    "TP: 뇌졸중이라고 맞힘 · FN: 뇌졸중인데 아니라고 예측함 · "
    "FP: 아닌데 뇌졸중이라고 예측함 · TN: 아니라고 맞힘"
)

# ---------- 2) 네 가지 지표 ----------
st.subheader("2. 네 가지 지표")
rows = []
for name in [LOGI_LABEL, TREE_LABEL, BASE_LABEL]:
    accuracy, recall, precision, f1 = metrics_from(counts[name])
    rows.append(
        {
            "모델": name,
            "정확도": show_pct(accuracy),
            "재현율": show_pct(recall),
            "정밀도": show_pct(precision),
            "F1": show_pct(f1),
        }
    )
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
st.caption(
    "정확도 = (TP+TN) ÷ 전체 · 재현율 = TP ÷ (TP+FN), 실제 뇌졸중인 사람 수로 나눠요 · "
    "정밀도 = TP ÷ (TP+FP), 뇌졸중이라 예측한 사람 수로 나눠요 · "
    "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율). "
    f"나누는 수가 0이면 '{NA_TEXT}'이라고 적어요."
)

# ---------- 3) 놓친 환자와 헛짚은 사람 ----------
st.subheader("3. 놓친 환자와 헛짚은 사람")
chosen = st.selectbox("어느 모델의 목록을 볼까요?", options=[LOGI_LABEL, TREE_LABEL])
pred = preds[chosen]

missed = person_list(test_df[(y_test == 1) & (pred == 0)])  # 실제 뇌졸중인데 아니라고 예측
wrong = person_list(test_df[(y_test == 0) & (pred == 1)])  # 아닌데 뇌졸중이라고 예측

left, right = st.columns(2)
with left:
    st.markdown(f"**놓친 환자 (전체 {len(missed):,}명)**")
    st.caption("실제 뇌졸중인데 아니라고 예측한 사람" + (f" · 나이가 많은 앞 {MAX_ROWS}줄만 보여 줘요" if len(missed) > MAX_ROWS else " · 나이가 많은 순"))
    st.dataframe(missed.head(MAX_ROWS), hide_index=True, use_container_width=True)
with right:
    st.markdown(f"**헛짚은 사람 (전체 {len(wrong):,}명)**")
    st.caption("아닌데 뇌졸중이라고 예측한 사람" + (f" · 나이가 많은 앞 {MAX_ROWS}줄만 보여 줘요" if len(wrong) > MAX_ROWS else " · 나이가 많은 순"))
    st.dataframe(wrong.head(MAX_ROWS), hide_index=True, use_container_width=True)

c = counts[chosen]
st.write(
    f"놓친 환자 {len(missed):,}명은 위 혼동행렬의 FN 칸({c['FN']:,}명)과, "
    f"헛짚은 사람 {len(wrong):,}명은 FP 칸({c['FP']:,}명)과 같아요."
)

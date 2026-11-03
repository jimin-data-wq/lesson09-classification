import pandas as pd
import plotly.express as px
import streamlit as st

APP_TITLE = "뇌졸중 예측 실습실"
PAGE_TITLE = "탐색"
PAGE_ICON = "🔍"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

# 브라우저 탭 제목과 아이콘 (반드시 첫 번째 streamlit 명령이어야 함)
st.set_page_config(page_title=f"{PAGE_TITLE} - {APP_TITLE}", page_icon=PAGE_ICON, layout="wide")

GROUP_LABELS = {0: "겪지 않은 사람", 1: "겪은 사람"}
YESNO_LABELS = {0: "없음", 1: "있음"}
GROUP_ORDER = ["겪지 않은 사람", "겪은 사람"]


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def stroke_rate_by(df, column):
    """column 값(0/1)마다 뇌졸중 비율(%)과 사람 수를 계산한다."""
    grouped = df.groupby(column)["stroke"].agg(["count", "sum", "mean"]).reset_index()
    grouped.columns = [column, "사람 수", "뇌졸중 사람 수", "뇌졸중 비율(%)"]
    grouped["뇌졸중 비율(%)"] = (grouped["뇌졸중 비율(%)"] * 100).round(2)
    grouped["구분"] = grouped[column].map(YESNO_LABELS)
    return grouped


def rate_bar(df, column, title):
    rates = stroke_rate_by(df, column)
    fig = px.bar(
        rates,
        x="구분",
        y="뇌졸중 비율(%)",
        text="뇌졸중 비율(%)",
        title=title,
        category_orders={"구분": ["없음", "있음"]},
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_yaxes(title="뇌졸중 비율(%)", rangemode="tozero")
    fig.update_xaxes(title="")
    return fig, rates


st.title(f"{PAGE_ICON} {PAGE_TITLE}")
st.caption(f"{APP_TITLE}: 데이터를 그림과 표로 살펴봐요.")

try:
    df = load_data()
except Exception as e:  # 네트워크 오류 등
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n{e}")
    st.stop()

df = df.copy()
df["뇌졸중 여부"] = df["stroke"].map(GROUP_LABELS)

# ---------- 1) 나이와 평균 혈당의 분포 ----------
st.subheader("1. 나이와 평균 혈당의 분포")
h1, h2 = st.columns(2)
with h1:
    fig = px.histogram(df, x="age", nbins=30, title="나이 분포")
    fig.update_xaxes(title="나이")
    fig.update_yaxes(title="사람 수")
    st.plotly_chart(fig, use_container_width=True)
with h2:
    fig = px.histogram(df, x="avg_glucose_level", nbins=30, title="평균 혈당 분포")
    fig.update_xaxes(title="평균 혈당(avg_glucose_level)")
    fig.update_yaxes(title="사람 수")
    st.plotly_chart(fig, use_container_width=True)

# ---------- 2) 뇌졸중 여부에 따른 비교 ----------
st.subheader("2. 뇌졸중을 겪은 사람과 겪지 않은 사람 비교")
b1, b2 = st.columns(2)
with b1:
    fig = px.box(
        df,
        x="뇌졸중 여부",
        y="age",
        color="뇌졸중 여부",
        category_orders={"뇌졸중 여부": GROUP_ORDER},
        title="나이 비교",
    )
    fig.update_xaxes(title="")
    fig.update_yaxes(title="나이")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
with b2:
    fig = px.box(
        df,
        x="뇌졸중 여부",
        y="avg_glucose_level",
        color="뇌졸중 여부",
        category_orders={"뇌졸중 여부": GROUP_ORDER},
        title="평균 혈당 비교",
    )
    fig.update_xaxes(title="")
    fig.update_yaxes(title="평균 혈당(avg_glucose_level)")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

means = (
    df.groupby("뇌졸중 여부")
    .agg(
        **{
            "사람 수": ("stroke", "count"),
            "나이 평균": ("age", "mean"),
            "평균 혈당 평균": ("avg_glucose_level", "mean"),
        }
    )
    .reindex(GROUP_ORDER)
    .round(2)
    .reset_index()
)
st.markdown("**두 그룹의 평균값**")
st.dataframe(means, hide_index=True, use_container_width=True)

# ---------- 3) 고혈압·심장병과 뇌졸중 비율 ----------
st.subheader("3. 고혈압·심장병과 뇌졸중 비율")
r1, r2 = st.columns(2)
with r1:
    fig, hyper_rates = rate_bar(df, "hypertension", "고혈압 여부별 뇌졸중 비율")
    st.plotly_chart(fig, use_container_width=True)
with r2:
    fig, heart_rates = rate_bar(df, "heart_disease", "심장병 여부별 뇌졸중 비율")
    st.plotly_chart(fig, use_container_width=True)

# ---------- 4) bmi가 빈 사람들 ----------
st.subheader("4. bmi가 비어 있는 사람들의 뇌졸중 비율")
bmi_missing = df[df["bmi"].isna()]
total_n = len(df)
total_stroke = int((df["stroke"] == 1).sum())
miss_n = len(bmi_missing)
miss_stroke = int((bmi_missing["stroke"] == 1).sum())

bmi_table = pd.DataFrame(
    {
        "구분": ["bmi가 빈 사람", "전체"],
        "사람 수": [miss_n, total_n],
        "뇌졸중 사람 수": [miss_stroke, total_stroke],
        "뇌졸중 비율(%)": [
            round(miss_stroke / miss_n * 100, 2) if miss_n else None,
            round(total_stroke / total_n * 100, 2) if total_n else None,
        ],
    }
)
st.dataframe(bmi_table, hide_index=True, use_container_width=True)

# ---------- 5) 흡연 상태별 사람 수 ----------
st.subheader("5. 흡연 상태(smoking_status)별 사람 수")
smoking = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)
st.dataframe(smoking, hide_index=True, use_container_width=True)

import json
from pathlib import Path

import pandas as pd
import streamlit as st

APP_TITLE = "뇌졸중 예측 실습실"
APP_ICON = "🧠"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

BASE_DIR = Path(__file__).resolve().parent
RECORD_DIR = BASE_DIR / "내_기록"
RECORD_FILE = RECORD_DIR / "데이터_소개_기록.json"

# 브라우저 탭 제목과 아이콘 (반드시 첫 번째 streamlit 명령이어야 함)
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")


# ---------- 기록 불러오기 / 저장하기 ----------
def load_records():
    """내_기록 폴더의 json 파일에서 우리말 뜻과 출처를 불러온다."""
    default = {"meanings": {}, "source": ""}
    if not RECORD_FILE.exists():
        return default
    try:
        with open(RECORD_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "meanings": dict(data.get("meanings", {})),
            "source": str(data.get("source", "")),
        }
    except (json.JSONDecodeError, OSError, AttributeError, TypeError, ValueError):
        return default


def save_records(records):
    """우리말 뜻과 출처를 내_기록 폴더에 json 파일로 저장한다."""
    try:
        RECORD_DIR.mkdir(parents=True, exist_ok=True)
        with open(RECORD_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        return True
    except OSError:
        return False


# ---------- 데이터 ----------
@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def describe_values(series):
    """값의 종류를 짧은 글로 설명한다."""
    values = series.dropna().unique()
    if len(values) == 0:
        return "값 없음"
    if len(values) <= 10:
        try:
            values = sorted(values)
        except TypeError:
            values = sorted(values, key=str)
        return ", ".join(str(v) for v in values)
    if pd.api.types.is_numeric_dtype(series):
        return f"숫자 ({series.min():g} ~ {series.max():g}), {len(values)}가지"
    return f"문자 {len(values)}가지"


def big_card(label, value):
    st.markdown(
        f"""
        <div style="border:1px solid rgba(128,128,128,0.35); border-radius:14px;
                    padding:22px 12px; text-align:center;
                    background:rgba(128,128,128,0.08);">
            <div style="font-size:1rem; opacity:0.75;">{label}</div>
            <div style="font-size:2.8rem; font-weight:700; line-height:1.3;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


try:
    df = load_data()
except Exception as e:  # 네트워크 오류 등
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n{e}")
    st.stop()

# 새로고침하면 파일에서 다시 불러오도록, 세션 시작 때 한 번만 읽는다.
if "initial_records" not in st.session_state:
    st.session_state["initial_records"] = load_records()
initial = st.session_state["initial_records"]

# ---------- 화면 ----------
st.title(f"{APP_ICON} {APP_TITLE}")
st.caption("첫 화면: 이 데이터가 무엇인지 알아보아요.")

# 1) 큰 숫자 카드 네 개
total = len(df)
n_cols = df.shape[1]
n_stroke = int((df["stroke"] == 1).sum())
ratio = n_stroke / total * 100 if total else 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    big_card("전체 사람 수", f"{total:,}명")
with c2:
    big_card("열 개수", f"{n_cols}개")
with c3:
    big_card("stroke가 1인 사람 수", f"{n_stroke:,}명")
with c4:
    big_card("stroke가 1인 비율", f"{ratio:.2f}%")

# 2) 열 설명 표 (우리말 뜻은 직접 입력)
st.subheader("열 설명 표")
st.caption("‘우리말 뜻’ 칸을 눌러 교재를 보고 직접 적어 보세요. 적은 내용은 자동으로 저장돼요.")

info = pd.DataFrame(
    {
        "열 이름": df.columns,
        "우리말 뜻": [initial["meanings"].get(c, "") for c in df.columns],
        "값의 종류": [describe_values(df[c]) for c in df.columns],
        "빈 값 개수": [int(df[c].isna().sum()) for c in df.columns],
    }
)

edited = st.data_editor(
    info,
    key="info_editor",
    hide_index=True,
    use_container_width=True,
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "우리말 뜻": st.column_config.TextColumn("우리말 뜻", help="교재를 보고 직접 적어 보세요."),
    },
)

# 3) 처음 다섯 줄
st.subheader("데이터 처음 다섯 줄")
st.dataframe(df.head(5), use_container_width=True)

# 4) 데이터 출처
st.subheader("데이터 출처")
source_text = st.text_area(
    "출처",
    value=initial["source"],
    key="source_box",
    height=120,
    placeholder="교재에 나온 데이터 출처를 여기에 적어 보세요.",
    label_visibility="collapsed",
)

# ---------- 자동 저장 ----------
current = {
    "meanings": {
        str(name): ("" if pd.isna(text) else str(text))
        for name, text in zip(edited["열 이름"], edited["우리말 뜻"])
    },
    "source": source_text,
}
if "last_saved" not in st.session_state:
    st.session_state["last_saved"] = {
        "meanings": {c: initial["meanings"].get(c, "") for c in df.columns},
        "source": initial["source"],
    }

if current != st.session_state["last_saved"]:
    if save_records(current):
        st.session_state["last_saved"] = current
        st.toast("내_기록 폴더에 저장했어요.", icon="💾")
    else:
        st.error("기록을 저장하지 못했어요. 폴더 쓰기 권한을 확인해 주세요.")

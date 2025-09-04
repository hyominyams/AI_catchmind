# app.py
# -*- coding: utf-8 -*-

import os
import io
import csv
import json
import random
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# ---------------- Gemini ----------------
try:
    import google.generativeai as genai
except Exception:
    genai = None


def get_gemini_model():
    api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
    if genai is None or not api_key:
        st.session_state["ai_status"] = "unavailable"
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        st.session_state["ai_status"] = "ok"
        return model
    except Exception as e:
        st.session_state["ai_status"] = "error"
        st.session_state["ai_error_msg"] = str(e)
        return None


# ---------------- Utils ----------------
def pil_from_canvas(image_data: Optional[np.ndarray]) -> Optional[Image.Image]:
    if image_data is None:
        return None
    arr = image_data.astype("uint8")  # RGBA
    img = Image.fromarray(arr, mode="RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    composed = Image.alpha_composite(bg, img)
    return composed.convert("RGB")


def image_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def normalize(s: Optional[str]) -> str:
    return (s or "").strip().lower()


# ---------------- 카테고리 (5개) ----------------
CATEGORIES = ["동물", "과일", "채소", "사물", "교통수단"]


# ---------------- 동의어(정답 인정 범위) ----------------
SYNONYMS: Dict[str, set] = {
    "고양이": {"냥이", "야옹이", "캣"},
    "강아지": {"개", "멍멍이", "도그"},
    "자동차": {"차", "승용차", "카"},
    "자전거": {"두바퀴", "바이크"},
    "텔레비전": {"tv", "티비"},
    "전화기": {"휴대폰", "핸드폰", "스마트폰", "폰"},
    "축구공": {"축구 볼", "사커볼"},
    "농구공": {"농구 볼", "바스켓볼"},
    "야구공": {"야구 볼", "베이스볼"},
}


def canon_ko(w: str) -> str:
    w = (w or "").strip().lower()
    for canon, alts in SYNONYMS.items():
        if w == canon or w in alts:
            return canon
    return w


# ---------------- 프롬프트 (후보 단어 절대 제공 금지) ----------------
PROMPT_GUESS_FREE = """
[역할] 너는 초등학생의 스케치를 보고 정답을 추측하는 심판이다.
[지시]
- 카테고리: {category}
- 이미지를 보고 한국어 단어 1개만 출력하라. (설명/문장/기호/영문 금지)
- 거친 선·부정확한 비율 허용. 윤곽/상징 요소(바퀴, 날개 등) 중시.
"""


# ---------------- 세션 ----------------
def init_state():
    ss = st.session_state
    ss.setdefault("page", "Home")
    ss.setdefault("category", CATEGORIES[0])
    ss.setdefault("max_rounds", 5)

    ss.setdefault("game_started", False)
    ss.setdefault("score", 0)
    ss.setdefault("round", 0)

    ss.setdefault("targets_queue", [])  # keyword.csv에서 사전 생성된 제시어들
    ss.setdefault("target", None)

    ss.setdefault("submitted", False)
    ss.setdefault("last_guess", "")
    ss.setdefault("round_end_time", None)
    ss.setdefault("auto_submit_triggered", False)

    ss.setdefault("ai_status", "unknown")   # ok | unavailable | error | unknown
    ss.setdefault("ai_error_msg", "")

    # 라벨링(선택): [{name: str, images: List[bytes]}]
    ss.setdefault("label_sets", [])


# ---------------- keyword.csv 로딩 ----------------
@st.cache_data(show_spinner=False)
def load_keywords_from_csv(path: str = "keyword.csv") -> Tuple[Dict[str, List[str]], Optional[str]]:
    """
    CSV 형식: category,keyword
    반환: {카테고리: [단어...]}와 에러 메시지(없으면 None)
    """
    data: Dict[str, List[str]] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            # 헤더 유무 관계없이 0=category, 1=keyword로 처리 시도
            # 첫 줄이 헤더가 아닐 수도 있으므로, 재처리
            if header and ("category" in header[0].lower() or "keyword" in header[1].lower()):
                pass
            else:
                # 헤더가 없었다면 첫 줄도 포함해서 다시 처리
                f.seek(0)
                reader = csv.reader(f)

            for row in reader:
                if len(row) < 2:
                    continue
                cat = row[0].strip()
                word = row[1].strip()
                if not cat or not word:
                    continue
                data.setdefault(cat, [])
                if word not in data[cat]:
                    data[cat].append(word)
        return data, None
    except FileNotFoundError:
        return {}, "keyword.csv 파일이 없습니다. 프로젝트 루트에 배치해 주세요."
    except Exception as e:
        return {}, f"keyword.csv 로딩 오류: {e}"


def pick_targets_from_csv(category: str, n: int, bank: Dict[str, List[str]]) -> List[str]:
    """CSV 단어뱅크에서 카테고리별로 n개 랜덤 추출(중복 없이, 부족하면 중복 허용 보충)."""
    candidates = bank.get(category, [])
    if not candidates:
        return []
    if n <= len(candidates):
        return random.sample(candidates, n)
    return random.sample(candidates, len(candidates)) + random.choices(candidates, k=n - len(candidates))


# ---------------- Gemini 호출 ----------------
def build_reference_parts(label_sets: List[Dict[str, Any]]) -> List[Any]:
    parts: List[Any] = []
    for item in label_sets:
        name = item.get("name")
        imgs: List[bytes] = item.get("images") or []
        if not name or not imgs:
            continue
        parts.append(f"참조: {name}")
        for b in imgs[:10]:
            parts.append({"mime_type": "image/png", "data": b})
    return parts


def guess_from_image(img: Optional[Image.Image], category: str) -> str:
    if img is None:
        return ""
    model = get_gemini_model()
    if model is None:
        return ""  # 상단 배너로 상태 표기

    prompt = PROMPT_GUESS_FREE.format(category=category)
    try:
        parts = [
            prompt,
            {"mime_type": "image/png", "data": image_to_png_bytes(img)},
        ]
        # (선택) 라벨링 참조 이미지 전달
        parts.extend(build_reference_parts(st.session_state.get("label_sets", [])))

        resp = model.generate_content(parts)
        text = (getattr(resp, "text", "") or "").strip()
        text = text.replace("\n", " ").replace("\r", " ").strip().strip('"\'')
        if " " in text:
            text = text.split()[0]  # 단어 1개만
        return text
    except Exception as e:
        st.session_state["ai_status"] = "error"
        st.session_state["ai_error_msg"] = str(e)
        return ""


# ---------------- 게임 플로우 ----------------
def start_game(keyword_bank: Dict[str, List[str]]):
    ss = st.session_state
    targets = pick_targets_from_csv(ss["category"], ss["max_rounds"], keyword_bank) or []
    if len(targets) < ss["max_rounds"]:
        st.warning("제시어를 충분히 생성하지 못했습니다. keyword.csv를 확인하세요.")
        return

    ss["targets_queue"] = targets
    ss["game_started"] = True
    ss["score"] = 0
    ss["round"] = 0
    next_round()
    ss["page"] = "Game"


def next_round():
    ss = st.session_state
    ss["round"] += 1
    ss["submitted"] = False
    ss["last_guess"] = ""
    ss["auto_submit_triggered"] = False

    idx = ss["round"] - 1
    if 0 <= idx < len(ss.get("targets_queue", [])):
        ss["target"] = ss["targets_queue"][idx]
    else:
        ss["target"] = None

    ss["round_end_time"] = datetime.utcnow() + timedelta(seconds=60)


def end_game_if_needed():
    ss = st.session_state
    if ss["round"] >= ss["max_rounds"] and ss["submitted"]:
        ss["game_started"] = False


def submit_answer(img_pil: Optional[Image.Image]):
    ss = st.session_state
    if ss["submitted"]:
        return
    ss["submitted"] = True
    guess = guess_from_image(img_pil, ss["category"]) if img_pil else ""
    ss["last_guess"] = guess
    if canon_ko(guess) == canon_ko(ss.get("target")):
        ss["score"] += 1


# ---------------- 라벨링(선택) ----------------
def add_label():
    st.session_state["label_sets"].append({"name": "", "images": []})


def remove_label(idx: int):
    labels = st.session_state.get("label_sets", [])
    if 0 <= idx < len(labels):
        labels.pop(idx)


def refresh_label_from_inputs(idx: int):
    key_name = f"label_name_{idx}"
    key_files = f"label_files_{idx}"
    name_val = st.session_state.get(key_name, "").strip()
    files_val = st.session_state.get(key_files) or []
    st.session_state["label_sets"][idx]["name"] = name_val

    imgs: List[bytes] = []
    for f in files_val[:10]:
        try:
            img = Image.open(f).convert("RGB")
            imgs.append(image_to_png_bytes(img))
        except Exception:
            pass
    st.session_state["label_sets"][idx]["images"] = imgs


# ---------------- UI ----------------
init_state()
st.set_page_config(page_title="AI 스케치 퀴즈", page_icon="🎨", layout="wide")

st.title("🎨 AI 스케치 퀴즈")

# AI 상태 배너
ai_status = st.session_state.get("ai_status", "unknown")
if ai_status == "unavailable":
    st.warning("⚠️ Gemini API 키가 설정되지 않아 AI를 호출할 수 없습니다. (GEMINI_API_KEY 필요)")
elif ai_status == "error":
    st.error(f"❌ Gemini 호출 오류: {st.session_state.get('ai_error_msg', '')}")
elif ai_status == "ok":
    st.success("✅ Gemini 연결 정상")

# 키워드 CSV 로딩
KEYWORD_BANK, CSV_ERR = load_keywords_from_csv("keyword.csv")
if CSV_ERR:
    st.error(f"❌ {CSV_ERR}")

page = st.session_state.get("page", "Home")

if page == "Home":
    st.subheader("카테고리 선택")
    st.radio("카테고리", CATEGORIES, key="category", horizontal=True)

    st.number_input("문제 수", min_value=1, max_value=20, step=1, key="max_rounds")
    st.button("게임 시작", type="primary", on_click=start_game, args=(KEYWORD_BANK,))

    st.markdown("---")

    # 라벨링(선택) — 하단
    st.subheader("라벨링(선택) · 정확도 보조자료")
    st.caption("필수 아님: 라벨과 참조 이미지를 추가하면 판정 시 참고합니다.")

    for i, item in enumerate(st.session_state["label_sets"]):
        with st.container(border=True):
            cols = st.columns([6, 1])
            with cols[0]:
                st.text_input("라벨 이름", value=item.get("name", ""), key=f"label_name_{i}", placeholder="예: 사과")
            with cols[1]:
                st.button("🗑️ 삭제", key=f"delete_label_{i}", on_click=remove_label, args=(i,), use_container_width=True)

            st.file_uploader(
                "참조 이미지(최대 10장)",
                key=f"label_files_{i}",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
            )
            refresh_label_from_inputs(i)

    st.button("+ 라벨 추가", on_click=add_label)

elif page == "Game":
    # 1초 갱신
    if st.session_state.get("game_started") and not st.session_state.get("submitted"):
        try:
            st.autorefresh(interval=1000, key="__tick__")
        except Exception:
            pass

    status_cols = st.columns([1, 1, 2])
    with status_cols[0]:
        st.metric("라운드", f"{st.session_state['round']}")
    with status_cols[1]:
        st.metric("점수", f"{st.session_state['score']}")
    with status_cols[2]:
        if st.session_state.get("game_started") and st.session_state.get("round_end_time"):
            remaining = int((st.session_state["round_end_time"] - datetime.utcnow()).total_seconds())
            remaining = max(0, remaining)
            st.metric("남은 시간 (초)", f"{remaining}")

    if st.session_state.get("game_started"):
        # 시간 만료 자동 제출
        if st.session_state.get("round_end_time") and datetime.utcnow() >= st.session_state["round_end_time"]:
            if not st.session_state["submitted"]:
                st.session_state["auto_submit_triggered"] = True

        st.subheader(f"제시어: {st.session_state['target']} (그려보세요!)")

        canvas_res = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=6,
            stroke_color="#000000",
            background_color="#FFFFFF",
            update_streamlit=True,
            height=360,
            width=640,
            drawing_mode="freedraw",
            key="canvas",
        )
        canvas_img = pil_from_canvas(canvas_res.image_data) if canvas_res is not None else None

        cols = st.columns([1, 1, 1])
        with cols[0]:
            if st.button("제출", type="primary", use_container_width=True, disabled=st.session_state["submitted"]):
                submit_answer(canvas_img)
        with cols[1]:
            if st.button("지우기", use_container_width=True):
                st.session_state.pop("canvas", None)
                st.experimental_rerun()

        if st.session_state.get("auto_submit_triggered") and not st.session_state["submitted"]:
            submit_answer(canvas_img)

        if st.session_state["submitted"]:
            st.markdown("---")
            st.subheader("결과")
            cols2 = st.columns(3)
            with cols2[0]:
                st.caption("AI 추측")
                guess_text = st.session_state["last_guess"] or "(응답 없음)"
                if st.session_state.get("ai_status") != "ok" and not st.session_state["last_guess"]:
                    guess_text += "  ·  ⚠️ AI 미호출"
                st.success(guess_text)
            with cols2[1]:
                st.caption("정답 제시어")
                st.info(st.session_state["target"] or "(없음)")
            with cols2[2]:
                verdict = (
                    "✅ 성공"
                    if canon_ko(st.session_state["last_guess"]) == canon_ko(st.session_state.get("target"))
                    else "❌ 실패"
                )
                st.metric("판정", verdict)

            nxt = st.button("다음 문제로", use_container_width=True)
            if nxt:
                end_game_if_needed()
                if not st.session_state["game_started"]:
                    st.warning("게임이 종료되었습니다. 홈에서 새 게임을 시작하세요.")
                    st.session_state["page"] = "Home"
                else:
                    next_round()

            end_game_if_needed()
    else:
        st.info("홈에서 카테고리를 고르고 '게임 시작'을 눌러주세요.")

st.markdown("---")
st.caption("제시어는 keyword.csv에서 무작위로 선정됩니다. AI에는 카테고리 외 단어 목록을 절대 제공하지 않습니다.")

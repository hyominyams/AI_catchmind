# app.py
# -*- coding: utf-8 -*-

"""
AI 스케치 퀴즈 — 카테고리 기반 제시어 + Gemini 2.5-Flash
- 홈에서 카테고리 선택 → 게임 시작 버튼으로 자동 이동
- 홈 하단에 '라벨링(선택)' 영역 추가 (정확도 보조자료)
"""

import os
import io
import random
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

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
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.5-flash")
    except Exception:
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


# ---------------- Categories ----------------
CATEGORIES = [
    "동물", "과일", "채소", "사물", "교통수단",
    "자연", "가전제품", "의류/패션", "스포츠/놀이", "건물/장소"
]

# ---------------- Prompts ----------------
PROMPT_TARGET = """
[역할]
너는 초등학생용 그림 퀴즈 출제자다.

[지시]
- 카테고리: {category}
- 초등학생이 쉽게 그릴 수 있는 단어 1개만 한국어로 출력해라.
- 출력은 단어만, 다른 설명이나 문장 없이.
"""

PROMPT_GUESS = """
[역할]
너는 아동의 거친 스케치를 보고, 주어진 정답 후보를 맞추는 분류기다.

[지시]
- 그림은 서툴 수 있으며, 비율과 디테일이 부정확할 수 있다.
- 아래 정답 후보와 그림을 비교해 가장 알맞은 답을 고른다.

[정답 후보]
{answer}

[출력 규칙]
- 반드시 정답 후보 단어와 동일하게 출력한다.
- 다른 단어, 문장, 설명 없이.
"""

# ---------------- Session ----------------
def init_state():
    ss = st.session_state
    ss.setdefault("page", "Home")
    ss.setdefault("category", CATEGORIES[0])
    ss.setdefault("game_started", False)
    ss.setdefault("score", 0)
    ss.setdefault("round", 0)
    ss.setdefault("target", None)
    ss.setdefault("submitted", False)
    ss.setdefault("last_guess", "")
    ss.setdefault("round_end_time", None)
    ss.setdefault("max_rounds", 5)           # 여기서만 기본값 설정
    ss.setdefault("auto_submit_triggered", False)

    # 라벨링(선택: 정확도 보조자료) — [{name: str, images: List[bytes]}]
    ss.setdefault("label_sets", [])


# ---------------- Gemini Calls ----------------
def generate_target_word(category: str) -> str:
    model = get_gemini_model()
    if model is None:
        return random.choice(["사과", "고양이", "자동차"])
    prompt = PROMPT_TARGET.format(category=category)
    try:
        resp = model.generate_content([prompt])
        word = (resp.text or "").strip().split()[0]
        return word
    except Exception:
        return random.choice(["사과", "고양이", "자동차"])


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


def guess_from_image(img: Optional[Image.Image], answer: str) -> str:
    if img is None or not answer:
        return ""
    model = get_gemini_model()
    if model is None:
        return ""
    prompt = PROMPT_GUESS.format(answer=answer)
    try:
        parts = [
            prompt,
            {"mime_type": "image/png", "data": image_to_png_bytes(img)}
        ]
        # 선택: 라벨링 참조 이미지도 함께 전달 (정확도 보조)
        parts.extend(build_reference_parts(st.session_state.get("label_sets", [])))
        resp = model.generate_content(parts)
        text = (resp.text or "").strip().replace("\n", "")
        return text
    except Exception:
        return ""


# ---------------- Game Flow ----------------
def start_game():
    ss = st.session_state
    ss["game_started"] = True
    ss["score"] = 0
    ss["round"] = 0
    next_round()
    ss["page"] = "Game"     # 홈 → 게임 화면으로 자동 이동


def next_round():
    ss = st.session_state
    ss["round"] += 1
    ss["submitted"] = False
    ss["last_guess"] = ""
    ss["auto_submit_triggered"] = False
    ss["target"] = generate_target_word(ss["category"])
    ss["round_end_time"] = datetime.utcnow() + timedelta(seconds=60)


def end_game_if_needed():
    ss = st.session_state
    if ss["round"] > ss["max_rounds"]:
        ss["game_started"] = False


def submit_answer(img_pil: Optional[Image.Image]):
    ss = st.session_state
    if ss["submitted"]:
        return
    ss["submitted"] = True
    guess = guess_from_image(img_pil, ss["target"]) if img_pil else ""
    ss["last_guess"] = guess
    if normalize(guess) == normalize(ss.get("target")):
        ss["score"] += 1


# ---------------- Labeling (Home 하단) ----------------
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

# 현재 페이지에 따라 렌더링
page = st.session_state.get("page", "Home")

if page == "Home":
    # 1) 카테고리 & 게임 설정 (간결하게)
    st.subheader("카테고리 선택")
    st.radio("카테고리", CATEGORIES, key="category", horizontal=True)

    st.number_input("문제 수", min_value=1, max_value=20, step=1, key="max_rounds")
    st.button("게임 시작", type="primary", on_click=start_game)

    st.markdown("---")

    # 2) 라벨링(선택) — 페이지 하단 배치
    st.subheader("라벨링(선택) · 정확도 보조자료")
    st.caption("필수 아님: 필요 시 라벨과 참조 이미지를 추가하면 AI가 판정할 때 함께 참고합니다.")

    # 라벨 목록 렌더링
    for i, item in enumerate(st.session_state["label_sets"]):
        with st.container(border=True):
            cols = st.columns([6, 1])
            with cols[0]:
                st.text_input("라벨 이름", value=item.get("name", ""),
                              key=f"label_name_{i}", placeholder="예: 사과")
            with cols[1]:
                st.button("🗑️ 삭제", key=f"delete_label_{i}",
                          on_click=remove_label, args=(i,), use_container_width=True)

            st.file_uploader(
                "참조 이미지(최대 10장)",
                key=f"label_files_{i}",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
                help="가능하면 라벨의 대표적인 모습을 담은 간단한 이미지로 준비하세요."
            )
            refresh_label_from_inputs(i)

    st.button("+ 라벨 추가", on_click=add_label)

elif page == "Game":
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
            if remaining == 0 and not st.session_state["submitted"] and not st.session_state["auto_submit_triggered"]:
                st.session_state["auto_submit_triggered"] = True
                st.info("⏰ 시간이 종료되어 자동 제출합니다.")

    if st.session_state.get("game_started"):
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
                st.caption("AI 예측")
                st.success(st.session_state["last_guess"] or "(예측 없음)")
            with cols2[1]:
                st.caption("정답 제시어")
                st.info(st.session_state["target"] or "(없음)")
            with cols2[2]:
                verdict = (
                    "✅ 성공" if normalize(st.session_state["last_guess"]) == normalize(st.session_state.get("target")) else "❌ 실패"
                )
                st.metric("판정", verdict)

            nxt = st.button("다음 문제로", use_container_width=True)
            if nxt:
                end_game_if_needed()
                if not st.session_state["game_started"]:
                    st.warning("게임이 종료되었습니다. 홈으로 돌아가 새 게임을 시작하세요.")
                    st.session_state["page"] = "Home"
                else:
                    next_round()
    else:
        st.info("홈에서 카테고리를 고르고 '게임 시작'을 눌러주세요.")

st.markdown("---")
st.caption("Tip: 라벨링은 선택 기능입니다. 필요 시 라벨·이미지를 추가하면 AI가 판정 시 함께 참고합니다.")

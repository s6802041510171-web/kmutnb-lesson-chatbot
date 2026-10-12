import os
import re
from pathlib import Path
from typing import List, Optional, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI
from openpyxl import load_workbook
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_dataset():
    """Load the compact Q&A workbook without a heavyweight ML runtime."""
    workbook = load_workbook(BASE_DIR / "dataset.xlsx", read_only=True, data_only=True)
    sheet = workbook.active
    rows = sheet.iter_rows(values_only=True)
    headers = next(rows, ())
    header_names = [str(value or "").strip().lower() for value in headers]

    try:
        question_index = header_names.index("question")
        answer_index = header_names.index("answer")
    except ValueError as exc:
        workbook.close()
        raise RuntimeError("dataset.xlsx must contain Question and Answer columns") from exc

    records = []
    for row in rows:
        question = str(row[question_index] or "").strip() if question_index < len(row) else ""
        answer = str(row[answer_index] or "").strip() if answer_index < len(row) else ""
        if question and answer:
            records.append({"question": question, "answer": answer})

    workbook.close()
    return records


QA_RECORDS = load_dataset()

KNOWLEDGE_BASE = {
    "แผนการสอนคืออะไร": (
        "แผนการจัดการเรียนรู้ (Lesson Plan) คือ เอกสารเตรียมการสอนอย่างเป็นระบบของครูผู้สอน "
        "ซึ่งกำหนดวัตถุประสงค์เชิงพฤติกรรม (K-P-A) เนื้อหา กิจกรรมการเรียนรู้ (ตามกระบวนการ MIAP 4 ขั้น) สื่อการสอน "
        "และการวัดประเมินผล เพื่อให้การจัดการเรียนการสอนบรรลุผลลัพธ์การเรียนรู้ที่ตั้งไว้อย่างมีประสิทธิภาพ"
    ),
    "แผนการสอนทำยังไง": (
        "ขั้นตอนการจัดทำแผนการจัดการเรียนรู้ตามแบบฟอร์ม คณะครุศาสตร์อุตสาหกรรม มจพ. มีดังนี้ครับ:\n\n"
        "1. กำหนดหัวข้อวิชา และระบุวัตถุประสงค์เชิงพฤติกรรม (พุทธิพิสัย, ทักษะพิสัย, จิตพิสัย)\n"
        "2. วางแผนกิจกรรมการเรียนการสอนตามกระบวนการ MIAP 4 ขั้นตอน:\n"
        "   - M (Motivation): ขั้นสนใจปัญหา\n"
        "   - I (Information): ขั้นบอกกล่าว/ให้ข้อมูล\n"
        "   - A (Application): ขั้นพยายาม/ฝึกปฏิบัติ\n"
        "   - P (Progress): ขั้นสำเร็จผล/ประเมินผล\n"
        "3. จัดเตรียมสื่อและเอกสารแนบท้ายแผน ได้แก่ แบบร่างกระดาน, ใบเนื้อหา, ใบงาน, ใบมอบหมายงาน และใบเฉลย"
    )
}

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^\wก-๙]", "", text)
    return text.strip()


def character_ngrams(text: str):
    """Return compact character features suitable for Thai text without spaces."""
    features = set()
    for size in (2, 3, 4):
        features.update(text[index:index + size] for index in range(len(text) - size + 1))
    return features


QA_FEATURES = [character_ngrams(clean_text(item["question"])) for item in QA_RECORDS]


def search_qa(query: str):
    q_clean = clean_text(query)
    if not q_clean:
        return None, []

    for k, v in KNOWLEDGE_BASE.items():
        if clean_text(k) in q_clean or q_clean in clean_text(k):
            return v, []

    query_features = character_ngrams(q_clean)
    if not query_features:
        return None, []

    ranked = []
    for record, features in zip(QA_RECORDS, QA_FEATURES):
        # Measure how much of the user's wording appears in each dataset question.
        score = len(query_features & features) / len(query_features)
        if score:
            ranked.append((score, record))

    ranked.sort(key=lambda item: item[0], reverse=True)
    matched_items = [record for score, record in ranked[:3] if score >= 0.08]
    if ranked and ranked[0][0] >= 0.35:
        return ranked[0][1]["answer"], matched_items

    return None, matched_items

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Any]] = []

@app.get("/")
def read_root():
    return FileResponse(BASE_DIR / "index.html")

@app.get("/style.css", include_in_schema=False)
def read_stylesheet():
    return FileResponse(BASE_DIR / "style.css", media_type="text/css")

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "total_records_in_dataset": len(QA_RECORDS),
        "source": "Excel dataset"
    }

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    user_msg = req.message.strip()
    clean_msg = user_msg.lower()

    if re.search(r"^(สวัสดี|หวัดดี|ดีครับ|ดีค่ะ|hello|hi)$", clean_msg):
        return {
            "source": "rule_based",
            "reply": "สวัสดีครับ! ผมคือ AI ผู้ช่วย แชทบอท สอบถามข้อมูลรายวิชาได้เลยครับ"
        }

    history_messages = []
    if req.history:
        for h in req.history[-6:]:
            if isinstance(h, dict):
                role = "assistant" if h.get("role") in ["bot", "model", "assistant"] else "user"
                content = h.get("text", "")
                if content:
                    history_messages.append({"role": role, "content": content})

    is_followup = any(w in clean_msg for w in ["แล้วทำยังไง", "ทำยังไง", "ทำอย่างไร", "ขั้นตอน", "ยังไงต่อ", "มีอะไรบ้าง"])
    effective_query = "ขั้นตอนการทำ" if (is_followup and len(clean_msg) < 18) else user_msg

    matched_answer, matched_items = search_qa(effective_query)
    
    if matched_answer and not req.history:
        return {"source": "dataset", "reply": matched_answer}

    context_lines = [f"- คำถาม: {m['question']}\n  คำตอบ: {m['answer']}" for m in matched_items]
    reference_context = "\n".join(context_lines) if context_lines else "ไม่มีข้อมูลที่เกี่ยวข้องโดยตรง"

    system_instruction = f"""คุณคือ AI ผู้ช่วยอัจฉริยะ ตอบคำถามผู้ใช้อย่างถูกต้อง ชัดเจน และตรงไปตรงมาอิงตามข้อมูลอ้างอิง
หากคำถามเกี่ยวข้องกับข้อมูลด้านล่าง ให้นำมาตอบให้เป็นธรรมชาติ
หากไม่มีข้อมูลในอ้างอิง ให้ตอบไปตามบริบทบทสนทนา หรือตอบว่าไม่มีข้อมูล

ข้อมูลอ้างอิง (Context):
{reference_context}
"""

    messages_payload = [{"role": "system", "content": system_instruction}]
    messages_payload.extend(history_messages)
    messages_payload.append({"role": "user", "content": user_msg})

    try:
        api_key = os.getenv("TYPHOON_API_KEY", "").strip()
        if not api_key:
            if matched_answer:
                return {"source": "dataset_fallback", "reply": matched_answer}
            elif matched_items:
                # Add a prefix so the user knows it's an imperfect match
                return {"source": "dataset_fallback", "reply": f"(ไม่พบคำถามที่ตรงกัน 100% และไม่มี API Key สำหรับ AI) \n\nคำตอบจากหัวข้อที่ใกล้เคียงที่สุด:\n{matched_items[0]['answer']}"}
            return {"source": "builtin", "reply": "ระบบไม่มีข้อมูลที่เกี่ยวข้องสำหรับคำถามนี้ และยังไม่ได้ตั้งค่า API Key สำหรับ AI ครับ"}

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.opentyphoon.ai/v1"
        )

        response = client.chat.completions.create(
            model="typhoon-v2.5-30b-a3b-instruct",
            messages=messages_payload,
            temperature=0.4,
            max_tokens=1000
        )

        reply_text = response.choices[0].message.content
        return {"source": "typhoon", "reply": reply_text}

    except Exception as e:
        print(f"Typhoon API error: {e}")
        if matched_answer:
            return {"source": "dataset_fallback", "reply": matched_answer}
        return {
            "source": "fallback",
            "reply": "ขออภัย เกิดข้อผิดพลาดในการเชื่อมต่อกับ AI"
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)

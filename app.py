import os
import re
import pandas as pd
from typing import List, Optional, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALL_QA_RECORDS = []

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^\wก-๙]", "", text)
    return text.strip()

def load_all_datasets():
    global ALL_QA_RECORDS
    ALL_QA_RECORDS = []
    
    file_path = "QA_Dataset_1000_Chatbot.xlsx"
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found!")
        return

    try:
        df = pd.read_excel(file_path)
        q_col = next((c for c in df.columns if str(c).strip().lower() in ["คำถาม", "question", "q"]), None)
        a_col = next((c for c in df.columns if str(c).strip().lower() in ["คำตอบ", "answer", "a"]), None)
        cat_col = next((c for c in df.columns if str(c).strip().lower() in ["หมวดหมู่", "category"]), None)

        if q_col and a_col:
            for _, row in df.iterrows():
                q = str(row.get(q_col, "")).strip()
                a = str(row.get(a_col, "")).strip()
                cat = str(row.get(cat_col, "")).strip() if cat_col else ""
                
                if q and a and q.lower() != "nan" and a.lower() != "nan":
                    ALL_QA_RECORDS.append({
                        "question": q,
                        "clean_q": clean_text(q),
                        "answer": a,
                        "category": cat
                    })
        print(f"Loaded {len(ALL_QA_RECORDS)} records from {file_path}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

load_all_datasets()

KNOWLEDGE_BASE = {
    "แผนการสอนทำยังไง": (
        "ขั้นตอนการจัดทำแผนการจัดการเรียนรู้ตามแบบฟอร์ม คณะครุศาสตร์อุตสาหกรรม มจพ. มีดังนี้ครับ:\n\n"
        "1. กำหนดหัวข้อวิชา และระบุวัตถุประสงค์เชิงพฤติกรรม (พุทธิพิสัย, ทักษะพิสัย, จิตพิสัย)\n"
        "2. วางแผนกิจกรรมการเรียนการสอนตามกระบวนการ MIAP 4 ขั้นตอน:\n"
        "   - M (Motivation): ขั้นสนใจปัญหา\n"
        "   - I (Information): ขั้นบอกกล่าว/ให้ข้อมูล\n"
        "   - A (Application): ขั้นพยายาม/ฝึกปฏิบัติ\n"
        "   - P (Progress): ขั้นสำเร็จผล/ประเมินผล\n"
        "3. จัดเตรียมสื่อและเอกสารแนบท้ายแผน ได้แก่ แบบร่างกระดาน, ใบเนื้อหา, ใบงาน, ใบมอบหมายงาน และใบเฉลย"
    ),
    "ขั้นตอนการทำ": (
        "ขั้นตอนการจัดทำแผนการจัดการเรียนรู้ มจพ. ประกอบด้วย 4 ขั้นตอนหลัก (MIAP):\n"
        "1. ขั้นสนใจปัญหา (Motivation) - กระตุ้นความสนใจและนำเข้าสู่บทเรียน\n"
        "2. ขั้นบอกกล่าว (Information) - ให้ความรู้ ทฤษฎี หรือสาธิตขั้นตอนการทำงาน\n"
        "3. ขั้นพยายาม (Application) - ให้ผู้เรียนฝึกปฏิบัติจริงตามใบงาน\n"
        "4. ขั้นสำเร็จผล (Progress) - สรุปผล ตรวจประเมินผลงาน และให้ข้อเสนอแนะ"
    )
}

def search_qa(query: str):
    q_clean = clean_text(query)
    if not q_clean:
        return None, []

    for k, v in KNOWLEDGE_BASE.items():
        if clean_text(k) in q_clean or q_clean in clean_text(k):
            return v, []

    for item in ALL_QA_RECORDS:
        if q_clean == item["clean_q"]:
            return item["answer"], [item]

    candidates = []
    keywords = ["ใบเนื้อหา", "ใบงาน", "ใบมอบหมายงาน", "ใบแบบฝึกหัด", "ใบเฉลย", "miap", "kpa", "rubric", "วัตถุประสงค์เชิงพฤติกรรม"]
    matched_kws = [kw for kw in keywords if kw in q_clean]

    for item in ALL_QA_RECORDS:
        score = 0
        for kw in matched_kws:
            if kw in item["clean_q"]:
                score += 5
        if score > 0:
            candidates.append((score, item))

    candidates.sort(key=lambda x: x[0], reverse=True)
    top_matches = [item for score, item in candidates[:3]]
    return None, top_matches

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Any]] = []

@app.get("/")
def read_root():
    return {
        "status": "ok", 
        "total_records": len(ALL_QA_RECORDS),
        "engine": "Typhoon LLM + KMUTNB Dataset"
    }

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    user_msg = req.message.strip()
    clean_msg = user_msg.lower()

    if re.search(r"^(สวัสดี|หวัดดี|ดีครับ|ดีค่ะ|hello|hi)$", clean_msg):
        return {
            "source": "rule_based",
            "reply": "สวัสดีครับ! ผมคือ AI ผู้ช่วยจัดทำแผนการจัดการเรียนรู้ มจพ. สอบถามขั้นตอนการสอน MIAP หรือเอกสารประกอบแผนได้เลยครับ"
        }

    # ถอดบริบทจากข้อความก่อนหน้า
    history_messages = []
    if req.history:
        for h in req.history[-6:]:
            if isinstance(h, dict):
                role = "assistant" if h.get("role") in ["bot", "model", "assistant"] else "user"
                history_messages.append({"role": role, "content": h.get("text", "")})

    is_followup = any(w in clean_msg for w in ["แล้วทำยังไง", "ทำยังไง", "ทำอย่างไร", "ขั้นตอน", "ยังไงต่อ", "มีอะไรบ้าง", "มีอะไรอีก"])
    effective_query = "ขั้นตอนการทำแผนการสอน" if (is_followup and len(clean_msg) < 18) else user_msg

    matched_answer, matched_items = search_qa(effective_query)
    if matched_answer and not req.history:
        return {"source": "dataset", "reply": matched_answer}

    context_lines = [f"- {m['question']} -> {m['answer']}" for m in matched_items]
    reference_context = "\n".join(context_lines) if context_lines else "ข้อมูลทั่วไปเกี่ยวกับแผนการจัดการเรียนรู้ มจพ."

    system_instruction = f"""คุณคือ AI ผู้เชี่ยวชาญการจัดทำแผนการจัดการเรียนรู้ คณะครุศาสตร์อุตสาหกรรม มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ (มจพ.)
หน้าที่ของคุณคือช่วยตอบคำถามอย่างถูกต้องตามหลักวิชาการ กระชับ สุภาพ และอิงตามกระบวนการสอน MIAP และแบบฟอร์ม มจพ.

ข้อมูลอ้างอิงจากแบบฟอร์ม/คลังข้อสอบ มจพ.:
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
            return {"source": "builtin", "reply": KNOWLEDGE_BASE["แผนการสอนทำยังไง"]}

        # เรียกใช้งาน Typhoon ผ่าน OpenAI-compatible client
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.opentyphoon.ai/v1"
        )

        response = client.chat.completions.create(
            model="typhoon-v1.5x-70b-instruct",  # หรือ typhoon-v1.5-instruct
            messages=messages_payload,
            temperature=0.4,
            max_tokens=1000
        )

        reply_text = response.choices[0].message.content
        return {"source": "typhoon", "reply": reply_text}

    except Exception as e:
        print(f"Typhoon API error: {e}")
        if is_followup or "ทำยังไง" in clean_msg or "ขั้นตอน" in clean_msg:
            return {"source": "builtin", "reply": KNOWLEDGE_BASE["แผนการสอนทำยังไง"]}
        if matched_answer:
            return {"source": "dataset_fallback", "reply": matched_answer}
        return {
            "source": "fallback",
            "reply": "สามารถสอบถามเพิ่มเติมเกี่ยวกับแบบฟอร์มแผนการสอน มจพ., กระบวนการ MIAP หรือเอกสารแนบท้ายได้เลยครับ"
        }
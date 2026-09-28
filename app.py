import os
import re
import glob
import pandas as pd
from typing import List, Optional, Any
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

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
    
    data_files = glob.glob("*.csv") + glob.glob("*.xlsx")
    
    for file_path in data_files:
        try:
            if file_path.endswith(".csv"):
                try:
                    df = pd.read_csv(file_path, encoding="utf-8-sig")
                except Exception:
                    df = pd.read_csv(file_path, encoding="tis-620")
            else:
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
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

load_all_datasets()

# ฐานความรู้หลักสูตร มจพ. สำหรับตอบทันที
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
    keywords = ["ใบเนื้อหา", "ใบงาน", "ใบมอบหมายงาน", "ใบแบบฝึกหัด", "ใบเฉลย", "miap", "kpa", "rubric", "วัตถุประสงค์"]
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
    return {"status": "ok", "total_records": len(ALL_QA_RECORDS)}

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    user_msg = req.message.strip()
    clean_msg = user_msg.lower()

    # 1. จัดการคำทักทาย
    if re.search(r"^(สวัสดี|หวัดดี|ดีครับ|ดีค่ะ|hello|hi)$", clean_msg):
        return {
            "source": "rule_based",
            "reply": "สวัสดีครับ! ผมคือ AI ผู้ช่วยจัดทำแผนการจัดการเรียนรู้ มจพ. สอบถามขั้นตอนการสอน MIAP หรือเอกสารประกอบแผนได้เลยครับ"
        }

    # 2. ถอดบริบทจากข้อความก่อนหน้า หากผู้ใช้ถามคำต่อเนื่อง (Follow-up)
    last_topic = "แผนการสอน"
    history_text = ""
    if req.history:
        for h in req.history[-4:]:
            if isinstance(h, dict):
                r = "ผู้ใช้" if h.get("role") in ["user", "human"] else "บอท"
                history_text += f"{r}: {h.get('text', '')}\n"

    # ถ้าผู้ใช้ถามสั้นๆ เช่น "แล้วทำยังไง", "ทำยังไง", "ขั้นตอนมีอะไรบ้าง"
    is_followup = any(w in clean_msg for w in ["แล้วทำยังไง", "ทำยังไง", "ทำอย่างไร", "ขั้นตอน", "ยังไงต่อ", "มีอะไรบ้าง"])
    effective_query = user_msg
    if is_followup and len(clean_msg) < 15:
        effective_query = f"ขั้นตอนการทำ{last_topic}"

    # 3. ตรวจสอบในคลังคำตอบ
    matched_answer, matched_items = search_qa(effective_query)
    if matched_answer:
        return {"source": "dataset", "reply": matched_answer}

    # 4. ส่งให้ Gemini สังเคราะห์คำตอบ
    prompt_content = f"""คุณคือ AI ผู้ช่วยจัดทำแผนการจัดการเรียนรู้ คณะครุศาสตร์อุตสาหกรรม มจพ.
ตอบคำถามอย่างกระชับ สุภาพ และถูกต้องตามกระบวนการสอนแบบ MIAP

บริบทการสนทนาก่อนหน้า:
{history_text}

คำถามล่าสุดของผู้ใช้: {user_msg}
(หัวข้อที่กำลังสนทนา: {effective_query})
"""

    try:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            return {"source": "fallback", "reply": KNOWLEDGE_BASE["แผนการสอนทำยังไง"]}

        client = genai.Client(api_key=api_key)
        interaction = client.interactions.create(
            model="gemini-3.8-flash",
            input=prompt_content
        )
        return {"source": "gemini-3.8-flash", "reply": interaction.output_text}
    except Exception:
        # หาก Gemini ขัดข้อง ให้ส่งคำตอบมาตรฐานที่ตรงเรื่องทันที
        if is_followup or "ทำยังไง" in clean_msg or "ขั้นตอน" in clean_msg:
            return {"source": "builtin", "reply": KNOWLEDGE_BASE["แผนการสอนทำยังไง"]}
        return {
            "source": "fallback",
            "reply": "สามารถสอบถามเพิ่มเติมเกี่ยวกับแบบฟอร์มแผนการสอน มจพ., กระบวนการ MIAP หรือเอกสารแนบท้ายได้เลยครับ"
        }
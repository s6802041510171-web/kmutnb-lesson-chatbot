import os
import re
import pandas as pd
from typing import List, Optional, Any
from fastapi import FastAPI
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
    
    file_path = "QA_Dataset_1000_Chatbot.xlsx"
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found!")
        return

    try:
        df = pd.read_excel(file_path)

        # ค้นหาชื่อคอลัมน์ คำถาม คำตอบ และหมวดหมู่
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

# ฐานความรู้หลักสูตร มจพ. สำหรับตอบทันทีเมื่อผู้ใช้ถามนิยามหรือขั้นตอน
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
    ),
    "ใบเนื้อหาคืออะไร": (
        "ใบเนื้อหา (Information Sheet) คือ เอกสารประกอบการสอนที่สรุปสาระสำคัญ องค์ความรู้ ทฤษฎี หรือขั้นตอนการปฏิบัติ "
        "เพื่อให้ผู้เรียนใช้ศึกษาประกอบในขั้นบอกกล่าว (I - Information) หรือใช้ทบทวนด้วยตนเอง "
        "โดยส่วนหัวของแบบฟอร์ม มจพ. จะระบุชื่อเรื่อง, ชื่อวิชา, หมายเลขหน้า และหมายเลขแผ่น"
    ),
    "miapคืออะไร": (
        "MIAP คือ รูปแบบกระบวนการจัดการเรียนการสอน 4 ขั้นตอนตามแนวทางของ มจพ. ได้แก่:\n"
        "1. M - Motivation (ขั้นสนใจปัญหา): กระตุ้นความสนใจและเตรียมความพร้อมผู้เรียน\n"
        "2. I - Information (ขั้นบอกกล่าว): ถ่ายทอดความรู้ ทฤษฎี หรือสาธิตขั้นตอนการทำงาน\n"
        "3. A - Application (ขั้นพยายาม): ให้ผู้เรียนฝึกปฏิบัติหรือทำแบบฝึกหัดด้วยตนเอง\n"
        "4. P - Progress (ขั้นสำเร็จผล): ตรวจสอบความถูกต้อง สรุปผล และประเมินผลการเรียนรู้"
    )
}

def search_qa(query: str):
    q_clean = clean_text(query)
    if not q_clean:
        return None, []

    # 1. เช็กกับนิยาม/ขั้นตอนมาตรฐาน
    for k, v in KNOWLEDGE_BASE.items():
        if clean_text(k) in q_clean or q_clean in clean_text(k):
            return v, []

    # 2. เช็กความตรงเป๊ะ 100% กับคำถามใน Dataset Excel
    for item in ALL_QA_RECORDS:
        if q_clean == item["clean_q"]:
            return item["answer"], [item]

    # 3. จับคู่คำสำคัญ
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
        "source_file": "QA_Dataset_1000_Chatbot.xlsx"
    }

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    user_msg = req.message.strip()
    clean_msg = user_msg.lower()

    # 1. คำทักทาย
    if re.search(r"^(สวัสดี|หวัดดี|ดีครับ|ดีค่ะ|hello|hi)$", clean_msg):
        return {
            "source": "rule_based",
            "reply": "สวัสดีครับ! ผมคือ AI ผู้ช่วยจัดทำแผนการจัดการเรียนรู้ มจพ. สอบถามขั้นตอนการสอน MIAP หรือเอกสารประกอบแผนได้เลยครับ"
        }

    # 2. ถอดบริบทจากบทสนทนาก่อนหน้า
    history_text = ""
    last_user_query = ""
    if req.history:
        for h in req.history[-4:]:
            if isinstance(h, dict):
                r = "ผู้ใช้" if h.get("role") in ["user", "human"] else "บอท"
                history_text += f"{r}: {h.get('text', '')}\n"
                if h.get("role") in ["user", "human"]:
                    last_user_query = h.get("text", "")

    # ตรวจสอบว่าเป็นคำถามต่อเนื่อง (Follow-up) หรือไม่
    is_followup = any(w in clean_msg for w in ["แล้วทำยังไง", "ทำยังไง", "ทำอย่างไร", "ขั้นตอน", "ยังไงต่อ", "มีอะไรบ้าง", "มีอะไรอีก"])
    effective_query = user_msg
    if is_followup and len(clean_msg) < 18:
        effective_query = f"ขั้นตอนการทำแผนการสอน"

    # 3. ค้นหาคำตอบจากไฟล์ Excel
    matched_answer, matched_items = search_qa(effective_query)
    if matched_answer:
        return {"source": "dataset", "reply": matched_answer}

    # 4. ส่งให้ Gemini สังเคราะห์คำตอบ
    context_lines = [f"- {m['question']} -> {m['answer']}" for m in matched_items]
    reference_context = "\n".join(context_lines) if context_lines else "ข้อมูลทั่วไปเกี่ยวกับแผนการจัดการเรียนรู้ มจพ."

    prompt_content = f"""คุณคือ AI ผู้ช่วยจัดทำแผนการจัดการเรียนรู้ คณะครุศาสตร์อุตสาหกรรม มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ (มจพ.)
หน้าที่ของคุณ: ตอบคำถามอย่างสละสลวย ถูกต้องตามหลักวิชาการ และต่อเนื่องจากบทสนทนาก่อนหน้า

บริบทการสนทนาก่อนหน้า:
{history_text}

ข้อมูลอ้างอิงจากคลังข้อสอบ/แบบฟอร์ม:
{reference_context}

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
        if is_followup or "ทำยังไง" in clean_msg or "ขั้นตอน" in clean_msg:
            return {"source": "builtin", "reply": KNOWLEDGE_BASE["แผนการสอนทำยังไง"]}
        return {
            "source": "fallback",
            "reply": "สามารถสอบถามเพิ่มเติมเกี่ยวกับแบบฟอร์มแผนการสอน มจพ., กระบวนการ MIAP หรือเอกสารแนบท้ายได้เลยครับ"
        }
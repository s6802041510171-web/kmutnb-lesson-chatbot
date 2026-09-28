import os
import re
import glob
import pandas as pd
from typing import List, Optional
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

# ค้นหาคำตอบจาก Dataset
def search_qa(query: str):
    q_clean = clean_text(query)
    if not q_clean:
        return None, []

    # 1. เช็กความตรงเป๊ะ 100% กับคำถามใน Dataset
    for item in ALL_QA_RECORDS:
        if q_clean == item["clean_q"]:
            return item["answer"], [item]

    # 2. ค้นหาคำสำคัญเฉพาะกลุ่มวิชาการ มจพ.
    keywords = [
        "ใบเนื้อหา", "ใบงาน", "ใบมอบหมายงาน", "ใบแบบฝึกหัด", "ใบเฉลย",
        "miap", "kpa", "rubric", "วัตถุประสงค์เชิงพฤติกรรม", "ขั้นสนใจปัญหา", 
        "ขั้นบอกกล่าว", "ขั้นพยายาม", "ขั้นสำเร็จผล", "แบบร่างกระดาน"
    ]
    matched_kws = [kw for kw in keywords if kw in q_clean]

    candidates = []
    for item in ALL_QA_RECORDS:
        score = 0
        q_target = item["clean_q"]
        for kw in matched_kws:
            if kw in q_target:
                score += 5
        if score > 0:
            candidates.append((score, item))

    candidates.sort(key=lambda x: x[0], reverse=True)
    top_matches = [item for score, item in candidates[:3]]

    # หากตรงคำสำคัญเฉพาะเจาะจง ให้ตอบตรง
    if candidates and candidates[0][0] >= 5 and len(q_clean) >= 6:
        return candidates[0][1]["answer"], top_matches

    return None, top_matches

# โครงสร้างรับข้อมูล รองรับทั้ง message และ history
class MessageItem(BaseModel):
    role: str  # "user" หรือ "model" / "assistant"
    text: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[MessageItem]] = []

@app.get("/")
def read_root():
    return {
        "status": "ok", 
        "total_records": len(ALL_QA_RECORDS),
        "message": "KMUTNB Lesson Chatbot with Context Memory is ready"
    }

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    user_msg = req.message.strip()
    clean_msg = user_msg.lower()

    if re.search(r"^(สวัสดี|หวัดดี|ดีครับ|ดีค่ะ|hello|hi)$", clean_msg):
        return {
            "source": "rule_based",
            "reply": "สวัสดีครับ! ผมคือ AI ผู้ช่วยตรวจสอบและแนะนำวิธีการจัดทำแผนการจัดการเรียนรู้ มจพ. สอบถามโครงสร้างแผน ขั้นตอน MIAP หรือเอกสารประกอบการสอนได้เลยครับ"
        }

    # ตรวจสอบว่าในคำถามเดี่ยวๆ มีใน Dataset เป๊ะๆ หรือไม่
    matched_answer, matched_items = search_qa(user_msg)
    
    # ถ้าไม่มีประวัติบทสนทนา และเจอใน Dataset เป๊ะ ให้ตอบทันที
    if matched_answer and not req.history:
        return {"source": "dataset", "reply": matched_answer}

    # แปลงประวัติการคุยเป็นข้อความบริบท
    history_context = ""
    if req.history:
        history_lines = []
        for h in req.history[-6:]:  # จดจำย้อนหลัง 6 ข้อความล่าสุด
            speaker = "ผู้ใช้" if h.role in ["user", "human"] else "บอท"
            history_lines.append(f"{speaker}: {h.text}")
        history_context = "ประวัติบทสนทนาก่อนหน้านี้:\n" + "\n".join(history_lines) + "\n\n"

    # รวบรวมข้อมูลอ้างอิงจาก Dataset
    context_lines = [f"- {m['question']} -> {m['answer']}" for m in matched_items]
    reference_context = "ข้อมูลอ้างอิงหลักสูตร/แบบฟอร์ม มจพ.:\n" + "\n".join(context_lines) if context_lines else ""

    prompt_content = f"""คุณคือ AI ผู้ช่วยจัดทำแผนการจัดการเรียนรู้ คณะครุศาสตร์อุตสาหกรรม มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ (มจพ.)
หน้าที่ของคุณ: ตอบคำถามผู้ใช้อย่างต่อเนื่อง ให้สอดคล้องกับเรื่องที่กำลังคุยกันในบริบทก่อนหน้า ถูกต้องตามหลักวิชาการ กระชับ และสุภาพ

{history_context}{reference_context}

คำถามล่าสุดของผู้ใช้: {user_msg}

แนวทางการตอบ:
1. หากผู้ใช้ถามสั้นๆ หรือถามต่อยอด (เช่น "ขั้นตอนการทำมีอะไรบ้าง", "มีอะไรอีก", "แล้วอันนี้ล่ะ") ให้อนุมานจากบริบทก่อนหน้า (เช่น หากเพิ่งคุยเรื่องแผนการสอน ให้ตอบขั้นตอนการจัดทำแผนการสอน 4 ขั้นตอน MIAP และองค์ประกอบเอกสาร)
2. โครงสร้าง MIAP ของ มจพ.:
   - M (Motivation): ขั้นสนใจปัญหา
   - I (Information): ขั้นบอกกล่าว/ให้ข้อมูล
   - A (Application): ขั้นพยายาม/ปฏิบัติ
   - P (Progress): ขั้นสำเร็จผล/ประเมินผล
3. ตอบเป็นข้อๆ ชัดเจน เข้าใจง่าย
"""

    try:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            if matched_answer:
                return {"source": "dataset_fallback", "reply": matched_answer}
            return {"source": "default", "reply": "กรุณาสอบถามเกี่ยวกับโครงสร้างแผนการสอน มจพ. หรือกระบวนการสอน MIAP ได้เลยครับ"}

        client = genai.Client(api_key=api_key)
        interaction = client.interactions.create(
            model="gemini-3.8-flash",
            input=prompt_content
        )
        return {"source": "gemini-3.8-flash", "reply": interaction.output_text}

    except Exception:
        # Fallback กรณีคำถามยอดฮิตแต่ Gemini ขัดข้อง
        if "ขั้นตอน" in clean_msg:
            return {
                "source": "fallback_context",
                "reply": (
                    "ขั้นตอนการจัดทำแผนการจัดการเรียนรู้ตามแบบฟอร์ม มจพ. มีดังนี้ครับ:\n\n"
                    "1. กำหนดหัวข้อวิชาและเขียนวัตถุประสงค์เชิงพฤติกรรม (พุทธิพิสัย, ทักษะพิสัย, จิตพิสัย)\n"
                    "2. ออกแบบกิจกรรมการเรียนรู้ตามกระบวนการ MIAP 4 ขั้น:\n"
                    "   - M (Motivation): ขั้นสนใจปัญหา\n"
                    "   - I (Information): ขั้นบอกกล่าว\n"
                    "   - A (Application): ขั้นพยายาม\n"
                    "   - P (Progress): ขั้นสำเร็จผล\n"
                    "3. จัดทำเอกสารแนบท้าย ได้แก่ แบบร่างกระดาน, ใบเนื้อหา, ใบงาน, ใบมอบหมายงาน และใบเฉลย"
                )
            }
        if matched_answer:
            return {"source": "dataset_fallback", "reply": matched_answer}
        return {
            "source": "default",
            "reply": "สามารถสอบถามเพิ่มเติมเกี่ยวกับแบบฟอร์มแผนการสอน มจพ., กระบวนการ MIAP หรือเอกสารแนบท้ายได้เลยครับ"
        }
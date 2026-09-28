import os
import re
import glob
import pandas as pd
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

def search_qa(query: str):
    q_clean = clean_text(query)
    if not q_clean:
        return None, []

    # 1. เช็กความตรงแบบคำถามตรงกันหรือซ้อนอยู่
    for item in ALL_QA_RECORDS:
        if q_clean == item["clean_q"] or (len(q_clean) >= 4 and (q_clean in item["clean_q"] or item["clean_q"] in q_clean)):
            return item["answer"], [item]

    # 2. ค้นหาคำสำคัญเฉพาะกลุ่มวิชาการ มจพ.
    keywords = [
        "ใบเนื้อหา", "ใบงาน", "ใบมอบหมายงาน", "ใบแบบฝึกหัด", "ใบเฉลย",
        "miap", "kpa", "rubric", "วัตถุประสงค์", "ขั้นสนใจปัญหา", 
        "ขั้นบอกกล่าว", "ขั้นพยายาม", "ขั้นสำเร็จผล", "แบบฟอร์ม", 
        "แผนการสอน", "แผนการจัดการเรียนรู้", "พฤติกรรม", "โครงสร้าง"
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

    if candidates and candidates[0][0] >= 5:
        return candidates[0][1]["answer"], top_matches

    # 3. ตรวจสอบความคล้ายตัวอักษรภาษาไทย
    fallback_candidates = []
    for item in ALL_QA_RECORDS:
        common_len = sum(1 for ch in set(q_clean) if ch in item["clean_q"])
        if common_len >= 5:
            fallback_candidates.append((common_len, item))
            
    fallback_candidates.sort(key=lambda x: x[0], reverse=True)
    if fallback_candidates:
        return fallback_candidates[0][1]["answer"], [fallback_candidates[0][1]]

    return None, []

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {
        "status": "ok", 
        "total_records": len(ALL_QA_RECORDS),
        "message": "KMUTNB Lesson Chatbot is ready"
    }

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    user_msg = req.message.strip()
    clean_msg = user_msg.lower()

    if re.search(r"^(สวัสดี|หวัดดี|ดีครับ|ดีค่ะ|hello|hi)$", clean_msg):
        return {
            "source": "rule_based",
            "reply": "สวัสดีครับ! ผมคือ AI ผู้ช่วยตรวจสอบและแนะนำวิธีการจัดทำแผนการจัดการเรียนรู้ มจพ. สอบถามโครงสร้างแผน ขั้นตอน MIAP หรือเกณฑ์วัดผลได้เลยครับ"
        }

    # ค้นหาจาก Dataset ทั้ง 1,001 ข้อ
    matched_answer, matched_items = search_qa(user_msg)
    if matched_answer:
        return {"source": "dataset", "reply": matched_answer}

    # หากไม่ตรงใน Dataset ให้สังเคราะห์ผ่าน Gemini 2.0 Flash (1,500 requests/day)
    context_text = "\n".join([f"- ถาม: {m['question']}\n  ตอบ: {m['answer']}" for m in matched_items]) if matched_items else ""
    prompt_content = f"""คุณคือ AI ผู้เชี่ยวชาญการจัดทำแผนการจัดการเรียนรู้ คณะครุศาสตร์อุตสาหกรรม มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ (มจพ.)
หน้าที่ของคุณ: ตอบคำถามผู้ใช้ให้ถูกต้องตามหลักวิชาการ กระชับ ตรงประเด็น และสุภาพ

ข้อมูลอ้างอิง:
{context_text}

คำถามของผู้ใช้: {user_msg}

ข้อกำหนดสำคัญ:
1. หากคำถามถามถึง "MIAP":
   - M = Motivation (ขั้นสนใจปัญหา)
   - I = Information (ขั้นบอกกล่าว)
   - A = Application (ขั้นพยายาม)
   - P = Progress (ขั้นสำเร็จผล)
2. หากถามถึงเอกสารประกอบ เช่น ใบเนื้อหา ให้ระบุองค์ประกอบสำคัญ (ชื่อเรื่อง, วิชา, หมายเลขหน้า/แผ่น และเนื้อหาที่สอน)
"""

    try:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            if matched_items:
                return {"source": "dataset_fallback", "reply": matched_items[0]["answer"]}
            return {"source": "error", "reply": "ขออภัยครับ ยังไม่พบข้อมูลที่ตรงกับคำถามนี้ในระบบ"}

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt_content
        )
        return {"source": "gemini-2.0-flash", "reply": response.text}

    except Exception as e:
        if matched_items:
            return {"source": "dataset_fallback", "reply": matched_items[0]["answer"]}
        return {
            "source": "error",
            "reply": f"เกิดข้อผิดพลาดในการประมวลผล: {e}"
        }
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

    # 1. เช็กความตรงแบบเป๊ะๆ 100% (ถ้าถามตรงกับหัวข้อใน Dataset เป๊ะ ให้ตอบทันที)
    for item in ALL_QA_RECORDS:
        if q_clean == item["clean_q"]:
            return item["answer"], [item]

    # 2. ตรวจสอบว่าคำถามเป็นเชิงนิยามหรือคำถามปลายเปิดหรือไม่
    # เช่น "คืออะไร", "หมายถึง", "อธิบาย", "มีอะไรบ้าง", "ทำไม", "ยังไง"
    is_explanatory_query = any(k in q_clean for k in [
        "คืออะไร", "คือ", "หมายถึง", "อธิบาย", "มีอะไรบ้าง", "อย่างไร", "ยังไง", "ทำไม", "บทบาท", "สำคัญอย่างไร"
    ])

    # 3. รวบรวมข้อมูลบริบทที่เกี่ยวข้องจาก Dataset ทั้ง 1,001 รายการ
    keywords = [
        "ใบเนื้อหา", "ใบงาน", "ใบมอบหมายงาน", "ใบแบบฝึกหัด", "ใบเฉลย",
        "miap", "kpa", "rubric", "วัตถุประสงค์", "ขั้นสนใจปัญหา", 
        "ขั้นบอกกล่าว", "ขั้นพยายาม", "ขั้นสำเร็จผล", "แบบฟอร์ม", 
        "แผนการสอน", "แผนการจัดการเรียนรู้", "พฤติกรรม", "โครงสร้าง", "นำเข้าสู่บทเรียน"
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
    top_matches = [item for score, item in candidates[:4]]

    # หากเป็นคำถามเชิงนิยาม ให้ส่งต่อไปสังเคราะห์ด้วย Gemini โดยแนบบริบทที่ค้นเจอไปด้วย
    if is_explanatory_query:
        return None, top_matches

    # ถ้าไม่ใช่คำถามนิยาม และคำถามตรงกับในไฟล์มาก ให้ตอบตรง
    if candidates and candidates[0][0] >= 5:
        return candidates[0][1]["answer"], top_matches

    return None, top_matches

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

    matched_answer, matched_items = search_qa(user_msg)
    if matched_answer:
        return {"source": "dataset", "reply": matched_answer}

    # รวบรวมข้อมูลอ้างอิงจาก Dataset ส่งเป็น Context ให้ Gemini
    context_lines = []
    for m in matched_items:
        context_lines.append(f"- ข้อมูลอ้างอิงในเอกสาร: {m['question']} -> {m['answer']}")
    context_text = "\n".join(context_lines) if context_lines else "ไม่มีข้อมูลเฉพาะเจาะจงในแบบฟอร์ม"

    prompt_content = f"""คุณคือ AI ผู้เชี่ยวชาญการจัดทำแผนการจัดการเรียนรู้ คณะครุศาสตร์อุตสาหกรรม มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ (มจพ.)
หน้าที่ของคุณ: อธิบายและตอบคำถามผู้ใช้ให้ถูกต้องตามหลักวิชาการ เข้าใจง่าย กระชับ ตรงประเด็น และสุภาพ

ข้อมูลอ้างอิงจากแบบฟอร์มและเอกสารของ มจพ.:
{context_text}

คำถามของผู้ใช้: {user_msg}

แนวทางการตอบ:
1. หากผู้ใช้ถามนิยาม (เช่น "ใบเนื้อหาคืออะไร", "ใบงานคืออะไร"): ให้อธิบายความหมาย ประโยชน์ หน้าที่ในกระบวนการสอน (เช่น ใช้ในขั้น Information: I) และอ้างอิงองค์ประกอบตามแบบฟอร์ม มจพ. ให้ครบถ้วน
2. หากถามถึง "MIAP" หรือ "ขั้นการสอน 4 ขั้น":
   - M = Motivation (ขั้นสนใจปัญหา)
   - I = Information (ขั้นบอกกล่าว)
   - A = Application (ขั้นพยายาม)
   - P = Progress (ขั้นสำเร็จผล)
3. สรุปเป็นข้อๆ ให้อ่านง่าย สละสลวย
"""

    try:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            if matched_items:
                return {"source": "dataset_fallback", "reply": matched_items[0]["answer"]}
            return {"source": "error", "reply": "ขออภัยครับ ยังไม่พบคีย์สำหรับประมวลผลคำตอบ"}

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
            "reply": f"เกิดข้อผิดพลาดในการประมวลผลคำตอบ: {e}"
        }
import os
import re
import csv
import glob
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

app = FastAPI()

# เปิดสิทธิ์ CORS ให้ Vercel ยิงเข้ามาได้ครบถ้วน
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# คลังข้อมูลเก็บคำถาม-คำตอบทั้งหมด
ALL_QA_RECORDS = []

def clean_text(text: str) -> str:
    """ทำความสะอาดข้อความเพื่อเปรียบเทียบ"""
    text = text.lower()
    text = re.sub(r"[^\w\sก-๙]", "", text)
    return text.strip()

def load_all_datasets():
    """กวาดอ่านข้อมูล Q&A จากไฟล์ Excel และ CSV ทั้งหมดในโฟลเดอร์"""
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

            # ตรวจหาชื่อคอลัมน์คำถาม คำตอบ และหมวดหมู่
            q_col = next((c for c in df.columns if str(c).strip().lower() in ["คำถาม", "question", "q"]), None)
            a_col = next((c for c in df.columns if str(c).strip().lower() in ["คำตอบ", "answer", "a"]), None)
            cat_col = next((c for c in df.columns if str(c).strip().lower() in ["หมวดหมู่", "category"]), None)

            if q_col and a_col:
                count = 0
                for _, row in df.iterrows():
                    q = str(row.get(q_col, "")).strip()
                    a = str(row.get(a_col, "")).strip()
                    cat = str(row.get(cat_col, "")).strip() if cat_col else ""
                    if q and a and q.lower() != "nan" and a.lower() != "nan":
                        ALL_QA_RECORDS.append({
                            "question": q,
                            "clean_q": clean_text(q),
                            "answer": a,
                            "category": cat,
                            "source_file": file_path
                        })
                        count += 1
                print(f"✅ โหลดชุดข้อมูล {file_path}: {count} ข้อ")
        except Exception as e:
            print(f"⚠️ ไม่สามารถเปิดอ่านไฟล์ {file_path}: {e}")

    print(f"🚀 รวมคลังความรู้พร้อมตอบทั้งหมด: {len(ALL_QA_RECORDS)} รายการ")

# สั่งโหลดข้อมูลทันทีเมื่อเซิร์ฟเวอร์เปิด
load_all_datasets()

def search_qa(query: str):
    """ค้นหาคำตอบที่ตรงหรือใกล้เคียงที่สุดจากชุดข้อมูล"""
    q_clean = clean_text(query)
    
    # 1. เช็กความตรงแบบ 100% หรือมีข้อความซ้อนกันอยู่
    for item in ALL_QA_RECORDS:
        if q_clean == item["clean_q"] or q_clean in item["clean_q"] or item["clean_q"] in q_clean:
            return item["answer"], [item]

    # 2. ค้นหาแบบแยกคำสำคัญ (Substrings / Keywords)
    words = [w for w in re.split(r"\s+", q_clean) if len(w) > 1]
    candidates = []
    
    for item in ALL_QA_RECORDS:
        score = 0
        q_target = item["clean_q"]
        for w in words:
            if w in q_target:
                score += 3
            elif w in item["category"].lower():
                score += 2
            elif w in item["answer"].lower():
                score += 1

        if score > 0:
            candidates.append((score, item))

    candidates.sort(key=lambda x: x[0], reverse=True)
    top_matches = [item for score, item in candidates[:3]]

    # หากมีคะแนนความเกี่ยวข้อง ให้ใช้คำตอบที่ดีที่สุด
    if candidates and candidates[0][0] >= 2:
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

    # --- ด่านที่ 1: ตรวจสอบคำทักทาย ---
    if re.search(r"^(สวัสดี|หวัดดี|ดีครับ|ดีค่ะ|hello|hi)$", clean_msg):
        return {
            "source": "rule_based",
            "reply": "สวัสดีครับ! ผมคือ AI ผู้ช่วยตรวจสอบและแนะนำวิธีการจัดทำแผนการจัดการเรียนรู้ มจพ. สอบถามองค์ประกอบ ขั้นตอน หรือเกณฑ์การวัดผลได้เลยครับ"
        }

    # --- ด่านที่ 2: ค้นหาคำตอบจาก Dataset 3 ไฟล์โดยตรง ---
    matched_answer, matched_items = search_qa(user_msg)
    if matched_answer:
        return {"source": "dataset", "reply": matched_answer}

    # --- ด่านที่ 3: สังเคราะห์คำตอบผ่าน AI (ถ้าหาไม่เจอจริงๆ) ---
    context_text = "\n".join([f"- ถาม: {m['question']}\n  ตอบ: {m['answer']}" for m in matched_items])
    prompt_content = f"""คุณคือ AI ผู้เชี่ยวชาญการจัดทำแผนการจัดการเรียนรู้ คณะครุศาสตร์อุตสาหกรรม มจพ.
หน้าที่ของคุณ: ตอบคำถามผู้ใช้ให้ถูกต้องตามหลักวิชาการ กระชับ ตรงประเด็น และสุภาพ

ข้อมูลอ้างอิง:
{context_text}

คำถามของผู้ใช้: {user_msg}

ข้อกำหนดสำคัญ:
1. หากคำถามถามถึง "MIAP" หรือ "ขั้นการสอน 4 ขั้น":
   - M = Motivation (ขั้นสนใจปัญหา)
   - I = Information (ขั้นบอกกล่าว)
   - A = Application (ขั้นพยายาม)
   - P = Progress (ขั้นสำเร็จผล)
2. อธิบายอย่างเป็นขั้นตอนและชัดเจน
"""

    try:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            if matched_items:
                return {"source": "dataset_fallback", "reply": matched_items[0]["answer"]}
            return {"source": "error", "reply": "ขออภัยครับ ยังไม่พบข้อมูลที่ตรงกับคำถามนี้ในระบบ"}

        client = genai.Client(api_key=api_key)
        interaction = client.interactions.create(
            model="gemini-3.8-flash",
            input=prompt_content
        )
        return {"source": "gemini-3.8-flash", "reply": interaction.output_text}

    except Exception as e:
        # หากติด Rate limit 429 หรือมี Error ให้ส่งคำตอบที่ใกล้เคียงที่สุดจาก Dataset กลับไปแทน
        if matched_items:
            return {"source": "dataset_fallback", "reply": matched_items[0]["answer"]}
        return {
            "source": "fallback",
            "reply": "ขออภัยครับ ขณะนี้ระบบประมวลผลคำตอบอัตโนมัติเต็มชั่วคราว กรุณาลองสอบถามเกี่ยวกับแบบฟอร์มแผนการสอน มจพ., โครงสร้าง MIAP หรือเนื้อหาหนังสือแผนการสอนได้เลยครับ"
        }
import os
import re
import csv
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
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

# 1. เชื่อมต่อฐานข้อมูล ChromaDB (แบบเบา ไม่กิน RAM)
DB_PATH = "./chroma_data"
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(
    name="lesson_qa_collection"
)

# 2. ตั้งค่า Gemini Client
api_key = os.getenv("GEMINI_API_KEY", "")
client = genai.Client(api_key=api_key) if api_key else None

# ไฟล์สำหรับเก็บคำตอบที่บอทเรียนรู้ด้วยตัวเอง
LEARNED_FILE = "learned_qa.csv"
if not os.path.exists(LEARNED_FILE):
    with open(LEARNED_FILE, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Question", "Answer"])

def auto_learn(question: str, answer: str):
    """ฟังก์ชันเรียนรู้และบันทึกคำตอบใหม่ลง Vector DB และไฟล์ CSV ทันที"""
    try:
        new_id = f"auto_{collection.count() + 1}"
        searchable_text = f"คำถาม: {question} คำตอบ: {answer}"
        collection.upsert(
            documents=[searchable_text],
            metadatas=[{"question": question, "answer": answer, "source": "self_learned"}],
            ids=[new_id]
        )
        with open(LEARNED_FILE, mode="a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([question, answer])
        print(f"💡 AI เรียนรู้และบันทึกข้อมูลใหม่แล้ว: {question}")
    except Exception as e:
        print(f"Error auto-learning: {e}")

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"status": "ok", "message": "KMUTNB Lesson Chatbot Backend is running live"}

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    user_msg = req.message.strip()
    clean_msg = user_msg.lower()

    # --- ด่านที่ 1: ตรวจสอบคำทักทาย ---
    if re.search(r"^(สวัสดี|หวัดดี|ดีครับ|ดีค่ะ|hello|hi)$", clean_msg):
        return {
            "source": "rule_based",
            "reply": "สวัสดีครับ! ผมคือ AI ผู้ช่วยตรวจสอบและแนะนำแผนการจัดการเรียนรู้ (มจพ.) สอบถามข้อมูลโครงสร้างแผน MIAP หรือเอกสารประกอบได้เลยครับ"
        }

    # --- ด่านที่ 2: ดึงข้อมูลจาก ChromaDB ---
    matched_texts = []
    best_answer = None
    metas = []

    try:
        results = collection.query(
            query_texts=[user_msg],
            n_results=4
        )
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results and results["metadatas"] else []
            distances = results["distances"][0] if "distances" in results and results["distances"] else [1.0] * len(docs)

            best_dist = distances[0] if distances else 1.0

            for doc, meta in zip(docs, metas):
                q_ref = meta.get("question", doc) if meta else doc
                ans = meta.get("answer", "") if meta else ""
                matched_texts.append(f"- คำถาม: {q_ref}\n  คำตอบ: {ans}")

            if best_dist < 0.25 and metas and metas[0].get("answer"):
                best_answer = metas[0].get("answer")
    except Exception as e:
        print(f"ChromaDB Query Error: {e}")

    if best_answer:
        return {
            "source": "vector_direct",
            "reply": best_answer
        }

    # --- ด่านที่ 3: สังเคราะห์คำตอบผ่าน AI ---
    context = "\n".join(matched_texts) if matched_texts else "ไม่มีข้อมูลที่ตรงกันโดยตรง"

    prompt_content = f"""คุณคือ AI ผู้เชี่ยวชาญการจัดทำแผนการจัดการเรียนรู้ คณะครุศาสตร์อุตสาหกรรม มจพ.
หน้าที่ของคุณ: ตอบคำถามผู้ใช้ให้ตรงประเด็น ถูกต้อง กระชับ และสุภาพ

ข้อมูลอ้างอิงจากฐานข้อมูล:
{context}

คำถามของผู้ใช้: {user_msg}

ข้อกำหนดสำคัญในการตอบ:
1. หากคำถามถามถึง "MIAP" หรือ "ขั้นการสอน 4 ขั้น": MIAP คือ Motivation (ขั้นสนใจปัญหา), Information (ขั้นบอกกล่าว), Application (ขั้นพยายาม), Progress (ขั้นสำเร็จผล)
2. หากในข้อมูลอ้างอิงมีเนื้อหาที่เกี่ยวข้อง ให้ดึงเนื้อหานั้นมาตอบเป็นหลัก
3. ตอบเฉพาะหัวข้อที่ถาม ไม่ต้องนำเนื้อหาอื่นที่ไม่เกี่ยวข้องมาปน
"""

    try:
        current_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not current_api_key:
            return {"source": "error", "reply": "ระบบขัดข้อง: ไม่พบคีย์ GEMINI_API_KEY ใน Render Environment"}

        ai_client = genai.Client(api_key=current_api_key)
        response = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_content
        )
        reply = response.text

        if reply and len(reply) > 10 and "ขออภัย" not in reply:
            auto_learn(user_msg, reply)

        return {"source": "gemini_learned", "reply": reply}

    except Exception as e:
        error_msg = str(e)
        print(f"Gemini API Error: {error_msg}")
        if metas and metas[0].get("answer"):
            return {"source": "vector_fallback", "reply": metas[0].get("answer")}
        return {"source": "error", "reply": f"เกิดข้อผิดพลาดจาก AI: {error_msg}"}
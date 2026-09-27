import os
import re
import csv
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
from google import genai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. เชื่อมต่อฐานข้อมูล ChromaDB
DB_PATH = "./chroma_data"
chroma_client = chromadb.PersistentClient(path=DB_PATH)
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
collection = chroma_client.get_or_create_collection(
    name="lesson_qa_collection",
    embedding_function=emb_fn
)

# 2. ตั้งค่า Gemini Client
api_key = os.getenv("GEMINI_API_KEY", "")
client = genai.Client(api_key=api_key) if api_key else None

# ไฟล์สำหรับเก็บคำตอบที่บอทเรียนรู้ด้วยตัวเองแบบถาวร
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

    # --- ด่านที่ 2: ดึงข้อมูลจาก ChromaDB (ดึง Top 4 เพื่อความแม่นยำ) ---
    results = collection.query(
        query_texts=[user_msg],
        n_results=4
    )

    matched_texts = []
    best_answer = None
    best_dist = 1.0

    if results and "documents" in results and results["documents"]:
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0] if "distances" in results and results["distances"] else [1.0]*len(docs)

        if distances:
            best_dist = distances[0]

        for doc, meta in zip(docs, metas):
            q_ref = meta.get("question", doc)
            ans = meta.get("answer", "")
            matched_texts.append(f"- คำถาม: {q_ref}\n  คำตอบ: {ans}")

        # ถ้าคะแนนความแม่นยำสูงมาก ให้ตอบตรงทันที
        if best_dist < 0.25 and metas:
            best_answer = metas[0].get("answer")

    if best_answer:
        return {
            "source": "vector_direct",
            "reply": best_answer
        }

    # --- ด่านที่ 3: สังเคราะห์คำตอบผ่าน AI + บันทึกจำคำตอบให้อัตโนมัติ ---
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
        interaction = client.interactions.create(
            model="gemini-3.8-flash",
            input=prompt_content
        )
        reply = interaction.output_text

        # 🧠 ระบบ Self-Learning: ถ้าคำตอบสมบูรณ์ ให้ AI บันทึกคำถามและคำตอบนี้เข้าฐานข้อมูลทันที
        if len(reply) > 15 and "ขออภัย" not in reply:
            auto_learn(user_msg, reply)

        return {"source": "gemini-3.8-flash_learned", "reply": reply}
    except Exception as e:
        # Fallback หาก API มีปัญหาหรือโควตาเต็ม
        if metas and metas[0].get("answer"):
            return {"source": "vector_fallback", "reply": metas[0].get("answer")}
        return {"source": "error", "reply": "ขออภัยครับ ยังไม่พบข้อมูลที่ตรงกับคำถามนี้ในระบบ"}
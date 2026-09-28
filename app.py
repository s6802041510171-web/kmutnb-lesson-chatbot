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

# คลังคำตอบมาตรฐานสำหรับคำถามยอดนิยมเชิงวิชาการ มจพ.
BUILTIN_KNOWLEDGE = {
    "แผนการสอนคืออะไร": (
        "แผนการจัดการเรียนรู้ (Lesson Plan) คือ เอกสารเตรียมการสอนอย่างเป็นระบบของครูผู้สอน "
        "ซึ่งกำหนดวัตถุประสงค์เชิงพฤติกรรม (K-P-A) เนื้อหา กิจกรรมการเรียนรู้ (ตามกระบวนการ MIAP 4 ขั้น) สื่อการสอน "
        "และการวัดประเมินผล เพื่อให้การจัดการเรียนการสอนบรรลุผลลัพธ์การเรียนรู้ที่ตั้งไว้อย่างมีประสิทธิภาพ"
    ),
    "แผนการสอนทำยังไง": (
        "การจัดทำแผนการสอนตามแบบฟอร์ม คณะครุศาสตร์อุตสาหกรรม มจพ. มีขั้นตอนหลักดังนี้:\n\n"
        "1. กำหนดหัวข้อวิชา และเขียนวัตถุประสงค์เชิงพฤติกรรม (พุทธิพิสัย, ทักษะพิสัย, จิตพิสัย)\n"
        "2. ออกแบบกิจกรรมการเรียนรู้ตามกระบวนการ MIAP 4 ขั้น:\n"
        "   - M (Motivation): ขั้นสนใจปัญหา\n"
        "   - I (Information): ขั้นบอกกล่าว/ให้ความรู้\n"
        "   - A (Application): ขั้นพยายาม/ฝึกปฏิบัติ\n"
        "   - P (Progress): ขั้นสำเร็จผล/ประเมินผล\n"
        "3. จัดเตรียมสื่อและเอกสารประกอบ ได้แก่ แบบร่างกระดาน, ใบเนื้อหา, ใบงาน, ใบมอบหมายงาน และใบเฉลย"
    ),
    "ใบเนื้อหาคืออะไร": (
        "ใบเนื้อหา (Information Sheet) คือ เอกสารประกอบการสอนที่สรุปสาระสำคัญ องค์ความรู้ ทฤษฎี หรือขั้นตอนการปฏิบัติ "
        "เพื่อให้ผู้เรียนใช้ศึกษาประกอบในขั้นบอกกล่าว (I - Information) หรือใช้ทบทวนด้วยตนเอง "
        "โดยส่วนหัวของแบบฟอร์ม มจพ. จะระบุชื่อเรื่อง, ชื่อวิชา, หมายเลขหน้า และหมายเลขแผ่น"
    ),
    "ใบงานคืออะไร": (
        "ใบงาน (Work Sheet) คือ เอกสารคำสั่งหรือกิจกรรมที่มอบหมายให้ผู้เรียนฝึกปฏิบัติจริงในขั้นพยายาม (A - Application) "
        "เพื่อพัฒนาทักษะการปฏิบัติงานตามขั้นตอนที่กำหนด มีเกณฑ์การให้คะแนนและการประเมินผลชัดเจน"
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

    # 1. เช็กกับ Built-in Knowledge นิยามมาตรฐาน
    for key, ans in BUILTIN_KNOWLEDGE.items():
        if key in q_clean or q_clean in key:
            return ans, []

    # 2. เช็กความตรงเป๊ะ 100% กับคำถามใน Dataset (ตรงทั้งประโยค)
    for item in ALL_QA_RECORDS:
        if q_clean == item["clean_q"]:
            return item["answer"], [item]

    # 3. รวบรวมข้อใกล้เคียงสำหรับส่งให้ Gemini เป็น Context (ไม่ตอบมั่ว)
    candidates = []
    keywords = [
        "ใบเนื้อหา", "ใบงาน", "ใบมอบหมายงาน", "ใบแบบฝึกหัด", "ใบเฉลย",
        "miap", "kpa", "rubric", "วัตถุประสงค์เชิงพฤติกรรม", "ขั้นสนใจปัญหา", 
        "ขั้นบอกกล่าว", "ขั้นพยายาม", "ขั้นสำเร็จผล", "แบบฟอร์ม", 
        "แบบร่างกระดาน", "การประเมินผล"
    ]
    matched_kws = [kw for kw in keywords if kw in q_clean]

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

    # ถ้าผู้ใช้ถามคำถามที่มีข้อความเหมือนใน Dataset มากกว่า 80% ถึงจะตอบตรง
    for item in ALL_QA_RECORDS:
        if len(q_clean) >= 8 and (q_clean in item["clean_q"] or item["clean_q"] in q_clean):
            # ป้องกันไม่ให้หยิบข้อสถาบันมาตอบคำถามนิยาม
            if "หน่วยงาน" in item["clean_q"] or "สถาบัน" in item["clean_q"]:
                continue
            return item["answer"], [item]

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
            "reply": "สวัสดีครับ! ผมคือ AI ผู้ช่วยตรวจสอบและแนะนำวิธีการจัดทำแผนการจัดการเรียนรู้ มจพ. สอบถามโครงสร้างแผน ขั้นตอน MIAP หรือเอกสารประกอบการสอนได้เลยครับ"
        }

    matched_answer, matched_items = search_qa(user_msg)
    if matched_answer:
        return {"source": "dataset", "reply": matched_answer}

    # ส่งบริบทอ้างอิงให้ Gemini
    context_lines = [f"- คำถามในเอกสาร: {m['question']}\n  คำตอบ: {m['answer']}" for m in matched_items]
    context_text = "\n".join(context_lines) if context_lines else "ข้อมูลทั่วไปเกี่ยวกับแผนการจัดการเรียนรู้ มจพ."

    prompt_content = f"""คุณคือ AI ผู้เชี่ยวชาญการจัดทำแผนการจัดการเรียนรู้ คณะครุศาสตร์อุตสาหกรรม มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ (มจพ.)
จงตอบคำถามต่อไปนี้อย่างถูกต้องตามหลักวิชาการ ชัดเจน สุภาพ และเข้าใจง่าย

ข้อมูลอ้างอิง:
{context_text}

คำถาม: {user_msg}
"""

    try:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            if matched_items:
                return {"source": "dataset_fallback", "reply": matched_items[0]["answer"]}
            return {"source": "default", "reply": "กรุณาสอบถามเกี่ยวกับโครงสร้างแผนการสอน มจพ., ขั้นตอน MIAP หรือเอกสารแนบท้ายได้เลยครับ"}

        client = genai.Client(api_key=api_key)
        interaction = client.interactions.create(
            model="gemini-3.8-flash",
            input=prompt_content
        )
        return {"source": "gemini-3.8-flash", "reply": interaction.output_text}

    except Exception:
        if matched_items:
            return {"source": "dataset_fallback", "reply": matched_items[0]["answer"]}
        return {
            "source": "default",
            "reply": "สามารถสอบถามเกี่ยวกับแบบฟอร์มแผนการสอน มจพ., กระบวนการ MIAP (Motivation, Information, Application, Progress) หรือเอกสารประกอบการสอนได้เลยครับ"
        }
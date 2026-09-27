import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# 1. เชื่อมต่อฐานข้อมูล ChromaDB
client = chromadb.PersistentClient(path="./chroma_data")

# 2. ตั้งค่า Embedding Model ภาษาไทย/อังกฤษ
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

# 3. ดึงหรือสร้าง Collection
collection = client.get_or_create_collection(
    name="lesson_qa_collection",
    embedding_function=emb_fn
)

def index_all_data():
    documents = []
    metadatas = []
    ids = []

    # --- ส่วนที่ 1: อ่านไฟล์ Excel เดิม (1,000 ข้อ) ---
    excel_file = "QA_Dataset_1000_Chatbot.xlsx"
    if os.path.exists(excel_file):
        print(f"กำลังอ่าน {excel_file}...")
        df_excel = pd.read_excel(excel_file).dropna(subset=["คำถาม", "คำตอบ"])
        for _, row in df_excel.iterrows():
            q_num = str(row.get("ลำดับข้อ", len(documents) + 1))
            question = str(row["คำถาม"]).strip()
            answer = str(row["คำตอบ"]).strip()

            documents.append(question)
            metadatas.append({"answer": answer, "source": "excel_1000"})
            ids.append(f"excel_{q_num}")
        print(f"-> โหลดข้อมูลจาก Excel ได้ {len(df_excel)} ข้อ")

    # --- ส่วนที่ 2: อ่านไฟล์ CSV ใหม่ (QA_Form_KMUTNB_2.csv) ---
    csv_file = "QA_Form_KMUTNB_2.csv"
    if os.path.exists(csv_file):
        print(f"กำลังอ่าน {csv_file}...")
        df_csv = pd.read_csv(csv_file).dropna(subset=["คำถาม", "คำตอบ"])
        for idx, row in df_csv.iterrows():
            q_num = str(row.get("ลำดับข้อ", idx + 1))
            question = str(row["คำถาม"]).strip()
            answer = str(row["คำตอบ"]).strip()
            category = str(row.get("หมวดหมู่", "ทั่วไป")).strip()

            documents.append(question)
            metadatas.append({
                "answer": answer,
                "category": category,
                "source": "kmutnb_form"
            })
            ids.append(f"form_{q_num}")
        print(f"-> โหลดข้อมูลจาก CSV เพิ่มได้ {len(df_csv)} ข้อ")
    else:
        print(f"⚠️ ไม่พบไฟล์ {csv_file} ในโฟลเดอร์โปรเจกต์")

    total = len(documents)
    print(f"\nรวมข้อมูลทั้งหมด {total} ข้อ กำลังแปลงและบันทึกลง Vector Database...")

    # บันทึกลง ChromaDB รอบละ 100 ข้อ
    batch_size = 100
    for i in range(0, total, batch_size):
        collection.upsert(
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
            ids=ids[i:i + batch_size]
        )
        print(f"บันทึกแล้ว {min(i + batch_size, total)}/{total} ข้อ")

    print("\n✅ อัปเดตข้อมูลเข้าสู่ ChromaDB ครบถ้วนเสร็จสมบูรณ์!")

if __name__ == "__main__":
    index_all_data()
import os
import sys
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma_data")
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

try:
    client.delete_collection(name="lesson_qa_collection")
except Exception:
    pass

collection = client.get_or_create_collection(
    name="lesson_qa_collection",
    embedding_function=emb_fn
)

def index_all_data():
    documents = []
    metadatas = []
    ids = []

    excel_file = "dataset.xlsx"
    if os.path.exists(excel_file):
        print(f"กำลังอ่าน {excel_file}...")
        df = pd.read_excel(excel_file)
        
        q_col = "Question" if "Question" in df.columns else next((c for c in df.columns if str(c).strip().lower() in ["คำถาม", "question", "q"]), None)
        a_col = "Answer" if "Answer" in df.columns else next((c for c in df.columns if str(c).strip().lower() in ["คำตอบ", "answer", "a"]), None)
        cat_col = "Category" if "Category" in df.columns else next((c for c in df.columns if str(c).strip().lower() in ["หมวดหมู่", "category"]), None)
        subcat_col = "Subcategory" if "Subcategory" in df.columns else None

        df = df.dropna(subset=[q_col, a_col])
        
        for idx, row in df.iterrows():
            q_num = str(row.get("ID", idx + 1))
            question = str(row[q_col]).strip()
            answer = str(row[a_col]).strip()
            category = str(row.get(cat_col, "ทั่วไป")).strip() if cat_col else "ทั่วไป"
            subcategory = str(row.get(subcat_col, "")).strip() if subcat_col else ""

            documents.append(question)
            metadatas.append({
                "answer": answer,
                "category": category,
                "subcategory": subcategory,
                "source": "new_dataset"
            })
            ids.append(f"qa_{q_num}")
        print(f"-> โหลดข้อมูลจาก Excel ได้ {len(df)} ข้อ")
    else:
        print(f"⚠️ ไม่พบไฟล์ {excel_file}")
        return

    total = len(documents)
    print(f"\nรวมข้อมูลทั้งหมด {total} ข้อ กำลังแปลงและบันทึกลง Vector Database...")

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
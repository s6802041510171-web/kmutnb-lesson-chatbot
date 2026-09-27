import os
import pandas as pd
import chromadb

# 1. เชื่อมต่อฐานข้อมูล ChromaDB
client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_collection(name="lesson_qa_collection")

# 2. ดึงรายการ ID ทั้งหมดที่มีอยู่ใน ChromaDB ตอนนี้
all_db_data = collection.get()
db_ids = set(all_db_data["ids"])
print(f"จำนวนข้อมูลทั้งหมดในฐานข้อมูลตอนนี้: {len(db_ids)} ข้อ")

# 3. รวบรวม ID ที่ยังคงมีอยู่ในไฟล์ข้อมูลปัจจุบัน
valid_ids = set()

# ตรวจจากไฟล์ Excel (ถ้ามี)
excel_file = "QA_Dataset_1000_Chatbot.xlsx"
if os.path.exists(excel_file):
    df_excel = pd.read_excel(excel_file).dropna(subset=["คำถาม", "คำตอบ"])
    for _, row in df_excel.iterrows():
        q_num = str(row.get("ลำดับข้อ", ""))
        if q_num:
            valid_ids.add(f"excel_{q_num}")

# ตรวจจากไฟล์ CSV (ถ้ามี)
csv_file = "QA_Form_KMUTNB_2.csv"
if os.path.exists(csv_file):
    df_csv = pd.read_csv(csv_file).dropna(subset=["คำถาม", "คำตอบ"])
    for idx, row in df_csv.iterrows():
        q_num = str(row.get("ลำดับข้อ", idx + 1))
        valid_ids.add(f"form_{q_num}")

# 4. หาเฉพาะ ID ที่ไม่อยู่ในไฟล์แล้ว (ถูกลบออกไป)
ids_to_delete = list(db_ids - valid_ids)

if ids_to_delete:
    print(f"พบข้อที่ถูกลบออกจากไฟล์จำนวน: {len(ids_to_delete)} ข้อ")
    print("กำลังนำออกจากฐานข้อมูล...")
    collection.delete(ids=ids_to_delete)
    print("ลบเฉพาะข้อที่ต้องการเรียบร้อยแล้ว!")
else:
    print("ฐานข้อมูลตรงกับไฟล์ปัจจุบันแล้ว ไม่มีข้อตกค้างที่ต้องลบครับ")
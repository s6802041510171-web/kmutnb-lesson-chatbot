import math
import os
import re
from collections import Counter
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openpyxl import load_workbook

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_dataset():
    """Load the compact Q&A workbook without a heavyweight ML runtime."""
    workbook = load_workbook(BASE_DIR / "dataset.xlsx", read_only=True, data_only=True)
    sheet = workbook.active
    rows = sheet.iter_rows(values_only=True)
    headers = next(rows, ())
    header_names = [str(value or "").strip().lower() for value in headers]

    try:
        question_index = header_names.index("question")
        answer_index = header_names.index("answer")
    except ValueError as exc:
        workbook.close()
        raise RuntimeError("dataset.xlsx must contain Question and Answer columns") from exc

    records = []
    for row in rows:
        question = str(row[question_index] or "").strip() if question_index < len(row) else ""
        # Keep the answer cell verbatim; only the question is normalized for indexing.
        answer = str(row[answer_index]) if answer_index < len(row) and row[answer_index] is not None else ""
        if question and answer.strip():
            records.append({"question": question, "answer": answer})

    workbook.close()
    return records


QA_RECORDS = load_dataset()

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^\wก-๙]", "", text)
    return text.strip()


def character_ngrams(text: str):
    """Return compact character features suitable for Thai text without spaces."""
    features = set()
    for size in (2, 3, 4):
        features.update(text[index:index + size] for index in range(len(text) - size + 1))
    return features


ANSWER_RECORDS = []
SEEN_ANSWERS = set()
for record in QA_RECORDS:
    if record["answer"] not in SEEN_ANSWERS:
        ANSWER_RECORDS.append(record)
        SEEN_ANSWERS.add(record["answer"])
RECORD_FEATURES = [
    character_ngrams(clean_text(item["question"] + " " + item["answer"]))
    for item in ANSWER_RECORDS
]
FEATURE_FREQUENCIES = Counter(feature for features in RECORD_FEATURES for feature in features)
FEATURE_IDF = {
    feature: math.log((len(ANSWER_RECORDS) + 1) / (frequency + 1)) + 1
    for feature, frequency in FEATURE_FREQUENCIES.items()
}


def search_qa(query: str):
    q_clean = clean_text(query)
    query_features = character_ngrams(q_clean)
    if not query_features:
        return None

    ranked = []
    query_weight = sum(FEATURE_IDF.get(feature, 1) for feature in query_features)
    for record, features in zip(ANSWER_RECORDS, RECORD_FEATURES):
        # Match against both the indexed question and its source passage. The
        # question often carries page/section context absent from the passage.
        score = sum(FEATURE_IDF.get(feature, 1) for feature in query_features & features) / query_weight
        if score:
            ranked.append((score, record))

    ranked.sort(key=lambda item: item[0], reverse=True)
    # Short Thai questions and section titles have little character overlap
    # with long passages, so use a softer threshold while still requiring
    # meaningful evidence from the document index.
    if ranked and ranked[0][0] >= 0.16:
        return ranked[0][1]["answer"]
    return None

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def read_root():
    return FileResponse(BASE_DIR / "index.html")

@app.get("/style.css", include_in_schema=False)
def read_stylesheet():
    return FileResponse(BASE_DIR / "style.css", media_type="text/css")

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "total_records_in_dataset": len(QA_RECORDS),
        "source": "Excel dataset"
    }

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    user_msg = req.message.strip()
    answer = search_qa(user_msg)
    if answer is not None:
        return {"source": "dataset", "reply": answer}
    return {"source": "not_found", "reply": "ไม่พบคำตอบที่ตรงกับข้อมูลในเอกสาร DatasetภาษาC"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)

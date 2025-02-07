import logging
from transformers import AutoTokenizer, AutoModel
import torch
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

# ตั้งค่าการแสดงผล log
logging.basicConfig(level=logging.INFO)

# 1. เชื่อมต่อ Milvus
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "law"

connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

# 2. ตรวจสอบ Collection
if COLLECTION_NAME in utility.list_collections():
    collection = Collection(name=COLLECTION_NAME)
    logging.info(f"Collection '{COLLECTION_NAME}' already exists.")
else:
    # สร้าง Schema สำหรับ Collection
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="law_name", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="year", dtype=DataType.INT64),
        FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
    ]
    schema = CollectionSchema(fields, description="Legal document embeddings")

    # สร้าง Collection ใหม่
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    logging.info(f"Created new collection '{COLLECTION_NAME}'.")

# 3. โหลดโมเดลและ Tokenizer
MODEL_PATH = r"D:\Chat\Streamlit\v1\volumes\BGE-M3"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True)
except Exception as e:
    logging.error(f"Error loading model or tokenizer: {e}")
    raise


# ฟังก์ชันสร้าง Embedding
def generate_embedding(text):
    if not text or text.strip() == "":
        logging.error("Text is empty or invalid.")
        return None
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0].cpu().numpy()  # ✅ ใช้ [CLS] Token

        if len(embedding[0]) != 1024:
            raise ValueError(f"Embedding size mismatch: expected 1024, got {len(embedding[0])}")

        return embedding[0].tolist()
    except Exception as e:
        logging.error(f"Error generating embedding for text: {text}. Error: {e}")
        return None


# 4. ข้อมูลตัวอย่าง
documents = [
    {
        "law_name": "ประมวลกฎหมายแพ่งและพาณิชย์",
        "year": 2535,
        "section": "มาตรา ๑๐",
        "content": "เมื่อความข้อใดข้อหนึ่งในเอกสารอาจตีความได้สองนัย ...",
    },
    {
        "law_name": "ประมวลกฎหมายแพ่งและพาณิชย์",
        "year": 2535,
        "section": "มาตรา ๙",
        "content": "ความในวรรคสองไม่ใช้บังคับแก่การลงลายพิมพ์นิ้วมือ ...",
    },
]

# 5. เตรียมข้อมูลสำหรับเพิ่มเข้า Milvus
data_to_insert = {
    "law_name": [],
    "year": [],
    "section": [],
    "content": [],
    "embedding": [],
}

valid_documents = 0
for doc in documents:
    embedding = generate_embedding(doc["content"])
    if embedding is not None:
        embedding = [float(x) for x in embedding]  # ✅ บังคับให้เป็น list ของ float
        data_to_insert["law_name"].append(doc["law_name"])
        data_to_insert["year"].append(doc["year"])
        data_to_insert["section"].append(doc["section"])
        data_to_insert["content"].append(doc["content"])
        data_to_insert["embedding"].append(embedding)
        valid_documents += 1

if valid_documents > 0:
    try:
        # ✅ แปลงข้อมูลให้อยู่ในรูปแบบ list of lists
        insert_data = [
            data_to_insert["law_name"],
            data_to_insert["year"],
            data_to_insert["section"],
            data_to_insert["content"],
            data_to_insert["embedding"]
        ]
        collection.insert(insert_data)
        logging.info(f"Inserted {valid_documents} valid documents into collection '{COLLECTION_NAME}'.")
    except Exception as e:
        logging.error(f"Error inserting data into Milvus. Error: {e}")
else:
    logging.warning("No valid documents to insert.")

# 6. ตรวจสอบและสร้างดัชนีใน Milvus
try:
    # ถ้าไม่มี Index ให้สร้างใหม่
    if not collection.indexes:
        collection.create_index(
            field_name="embedding",
            index_params={"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 128}},
        )
        logging.info("Index created successfully.")

    # โหลด Collection หลังจากสร้าง Index
    collection.load()
    logging.info("Collection loaded successfully.")
except Exception as e:
    logging.error(f"Error creating or loading index: {e}")

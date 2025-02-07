import os
import streamlit as st
import time
import logging
import boto3
import torch
import json
from pymilvus import connections, Collection, utility
from transformers import AutoTokenizer, AutoModel
from botocore.config import Config

# ปิดข้อความแจ้งเตือนจาก Torch ที่ไม่จำเป็น
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

# ตั้งค่า Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MILVUS_HOST = "milvus-standalone"
MILVUS_PORT = "19530"
MILVUS_COLLECTION_NAME = "law"
BEDROCK_REGION = "us-east-1"
BEDROCK_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"  # ตรวจสอบให้ตรงกับ Model ID ที่รองรับ
MODEL_PATH = "/app/BGE-M3"  # ตรวจสอบให้แน่ใจว่า path ถูกต้อง
DIMENSION = 1024
MAX_RETRIES = 3
RETRY_DELAY = 1.5

# ตรวจสอบและเลือกอุปกรณ์ (GPU หรือ CPU)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"🚀 ใช้อุปกรณ์: {DEVICE}")

#######################################
# ฟังก์ชันโหลดโมเดล Embedding (cached)
#######################################
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    try:
        logger.info(f"🚀 โหลดโมเดล Embedding บนอุปกรณ์: {DEVICE}")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            cache_dir="/tmp",
            trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            cache_dir="/tmp",
            trust_remote_code=True
        ).to(DEVICE)
        logger.info("✅ โหลด Tokenizer และ Model สำเร็จ")

        if DEVICE.type == "cuda":
            try:
                # model = torch.compile(model)  # Uncomment ถ้าระบบรองรับและมี C compiler
                logger.info("✅ torch.compile ผ่าน (ถ้าใช้ได้)")
            except Exception as compile_error:
                logger.warning(f"torch.compile ล้มเหลว: {compile_error}. ดำเนินการต่อโดยไม่ compile โมเดล")
            model = model.half()  # ลด precision เป็น 16-bit สำหรับ GPU

            # ทดสอบด้วย dummy input
            dummy_input = tokenizer("ข้อความทดสอบ", return_tensors="pt", padding=True, truncation=True, max_length=512)
            dummy_input = {k: v.to(DEVICE) for k, v in dummy_input.items()}
            with torch.no_grad():
                _ = model(**dummy_input)

        model.eval()
        logger.info(f"✅ โหลด BGE-M3 สำเร็จบน {DEVICE}")
        return tokenizer, model
    except Exception as e:
        logger.error(f"❌ ข้อผิดพลาดในการโหลดโมเดล: {e}")
        return None, None

#######################################
# ฟังก์ชันเชื่อมต่อ Milvus (retry และ cached)
#######################################
def milvus_connection_with_retry():
    for attempt in range(MAX_RETRIES):
        try:
            connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
            if not utility.has_collection(MILVUS_COLLECTION_NAME):
                logger.error(f"Collection not found: {MILVUS_COLLECTION_NAME}")
                return None
            collection = Collection(MILVUS_COLLECTION_NAME)

            # 🔹 โหลด Collection เข้าหน่วยความจำ
            collection.load()  # <== เพิ่มบรรทัดนี้

            logger.info(f"✅ Connected to Milvus and loaded collection on attempt {attempt + 1}")
            return collection
        except Exception as e:
            logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
            time.sleep(RETRY_DELAY ** (attempt + 1))
    return None


@st.cache_resource
def connect_to_milvus():
    collection = milvus_connection_with_retry()
    if collection is None:
        logger.error("❌ ไม่สามารถเชื่อมต่อกับ Milvus ได้")
    return collection

def test_milvus_search(collection, tokenizer, model):
    test_text = "ข้อความทดสอบ"
    logger.info(f"🚀 เริ่มทดสอบค้นหาด้วยข้อความ: {test_text}")
    embedding = generate_embedding(test_text, tokenizer, model)
    if not embedding:
        logger.error("❌ สร้าง embedding สำหรับทดสอบไม่สำเร็จ")
        return
    results = search_milvus(collection, embedding, top_k=3)
    if results is None:
        logger.error("❌ ไม่สามารถค้นหาใน Milvus ได้")
    else:
        processed = process_results(results)
        logger.info(f"✅ ผลลัพธ์การทดสอบ: {processed}")

#######################################
# ฟังก์ชันโหลด Bedrock LLM Client (cached)
#######################################
@st.cache_resource(show_spinner=False)
def get_llm_client():
    try:
        client = boto3.client(
            service_name="bedrock-runtime",
            region_name=BEDROCK_REGION,
            config=Config(connect_timeout=5, read_timeout=60)
        )
        logger.info("✅ Initialized Bedrock LLM client")
        return client
    except Exception as e:
        logger.error(f"❌ Error initializing LLM client: {e}")
        return None


#######################################
# ฟังก์ชันสร้าง Embedding
#######################################
def generate_embedding(text, tokenizer, model):
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        logger.info(f"Input tokens: {inputs}")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            # ใช้ mean pooling ของ hidden states เป็น embedding
            embedding = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()[0].tolist()
        logger.info(f"Embedding dimension: {len(embedding)} (คาดหวัง: {DIMENSION})")
        return embedding
    except Exception as e:
        logger.error(f"❌ ข้อผิดพลาดในการสร้าง embedding: {e}")
        return None


#######################################
# ฟังก์ชันค้นหาใน Milvus
#######################################
def search_milvus(collection, query_embedding, top_k=5):
    try:
        # logger.info("🚀 เริ่ม flush collection ก่อนค้นหา")
        # collection.flush()  # ให้แน่ใจว่าข้อมูลล่าสุดถูก flush
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
        logger.info(f"🔍 เริ่มค้นหาใน Milvus ด้วยพารามิเตอร์: {search_params}")
        start_time = time.time()
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["law_name", "year", "section", "content"],
            consistency_level="Bounded"
        )
        elapsed_time = time.time() - start_time
        logger.info(f"✅ ค้นหา Milvus เสร็จสิ้นใน {elapsed_time:.2f} วินาที")
        if results:
            total_hits = sum(len(hits) for hits in results)
            logger.info(f"พบผลลัพธ์: {total_hits} รายการ")
        else:
            logger.info("ไม่พบผลลัพธ์จาก Milvus")
        return results
    except Exception as e:
        logger.error(f"❌ ข้อผิดพลาดในการค้นหา Milvus: {e}")
        return None

#######################################
# ฟังก์ชันจัดรูปแบบผลลัพธ์การค้นหา
#######################################
def process_results(search_results):
    try:
        return [
            {
                "law_name": getattr(hit.entity, "law_name", "N/A"),
                "year": getattr(hit.entity, "year", "N/A"),
                "section": getattr(hit.entity, "section", "N/A"),
                "content": getattr(hit.entity, "content", "N/A"),
                "score": hit.score
            }
            for hits in search_results for hit in hits
        ] if search_results else []
    except Exception as e:
        logger.error(f"❌ ข้อผิดพลาดในการประมวลผลผลลัพธ์: {e}")
        return []

#######################################
# ฟังก์ชันเรียกใช้ LLM (Bedrock) เพื่อสร้างคำตอบ
#######################################
def generate_response(llm_client, query, context, category):
    if not context:
        return "ไม่พบข้อมูลที่เกี่ยวข้องในฐานข้อมูล"
    prompt_template = f"""
<บทบาท> คุณเป็นผู้ช่วยทางกฎหมายไทยอัจฉริยะ </บทบาท>
<ข้อมูลอ้างอิง> {context} </ข้อมูลอ้างอิง>
<คำถาม> {query} </คำถาม>
<ข้อกำหนด>
1. ตอบด้วยภาษาไทยที่เข้าใจง่าย
2. ระบุชื่อกฎหมายและมาตราอย่างชัดเจน
3. หากข้อมูลไม่พอให้แจ้งผู้ใช้
4. เรียงลำดับตามความสำคัญ
5. หลีกเลี่ยงศัพท์ทางเทคนิคที่ไม่จำเป็น
6. ถ้าคำตอบยาวเกินให้ตอบพอสังเขป
</ข้อกำหนด>
"""
    payload = {
        "messages": [
            {"role": "user", "content": prompt_template}
        ],
        "max_tokens": 500,  # จำนวน token สูงสุดที่ต้องการให้ตอบ
        "anthropic_version": "bedrock-2023-05-31"  # เวอร์ชันที่ระบุ (ตามที่ Bedrock ต้องการ)
    }
    try:
        logger.info("📨 ส่งข้อความไปยัง LLM ผ่าน Bedrock...")
        response = llm_client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload)
        )
        # อ่าน response และแปลงเป็น dict
        response_body = response["body"].read().decode("utf-8")
        result = json.loads(response_body)

        # Logging โครงสร้างของ response เพื่อวิเคราะห์
        logger.info(f"LLM raw response: {json.dumps(result, ensure_ascii=False, indent=2)}")

        # ปรับการดึงข้อมูลคำตอบตามโครงสร้างที่ได้พบ (ใช้ key "content" แทน "completions")
        content = result.get("content", [])
        if content and isinstance(content, list):
            # ดึงข้อความจาก dictionary แรกใน list
            answer = content[0].get("text", "")
            if answer:
                return answer
        return "ไม่สามารถดึงคำตอบจาก LLM ได้"
    except Exception as e:
        logger.error(f"❌ Error generating response: {e}")
        return "เกิดข้อผิดพลาดในการสร้างคำตอบ"


#######################################
# ฟังก์ชันโหลดทรัพยากรทั้งหมด
#######################################
def load_all():
    tokenizer, model = load_embedding_model()
    collection = connect_to_milvus()
    llm_client = get_llm_client()
    return tokenizer, model, collection, llm_client

# ฟังก์ชันจัดการเอกสาร S3
def setup_s3_client():
    if "s3_client" not in st.session_state:
        st.session_state.s3_client = boto3.client('s3')

def list_files_in_s3(bucket_name):
    try:
        objects = st.session_state.s3_client.list_objects_v2(Bucket=bucket_name)
        return [obj['Key'] for obj in objects.get('Contents', [])] if 'Contents' in objects else []
    except Exception as e:
        st.error(f"Error listing files: {e}")
        return []

def upload_to_s3(file, bucket_name, file_name):
    try:
        sanitized_file_name = file_name.replace(" ", "_")  # ทำความสะอาดชื่อไฟล์
        st.session_state.s3_client.upload_fileobj(file, bucket_name, sanitized_file_name)
        return True
    except Exception as e:
        st.error(f"Error uploading file: {e}")
        return False

#######################################
# ส่วนติดต่อผู้ใช้ (Streamlit UI)
#######################################
def main():
    st.set_page_config(page_title="แชทบอททนายและการจัดการเอกสาร", page_icon="⚖️", layout="wide")
    st.title("แชทบอททนาย 📜")

    st.title("Main Page")

########################################
#ส่วนเพิ่มไซต์บาร์ (Sidebar)
########################################
    with st.sidebar:
        st.header("🗂️ จัดการแชท")

        # ตรวจสอบ state และกำหนดค่าเริ่มต้น
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = {}
        if "current_chat" not in st.session_state:
            st.session_state.current_chat = None
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # ปุ่มสร้างแชทใหม่
        if st.button("➕ สร้างแชทใหม่"):
            new_chat_id = f"แชท {len(st.session_state.chat_history) + 1}"
            st.session_state.chat_history[new_chat_id] = []
            st.session_state.current_chat = new_chat_id
            st.session_state.messages = []
            st.success("สร้างแชทใหม่สำเร็จ!")

        # แสดงรายการแชทที่มี และเลือกแชท
        if st.session_state.chat_history:
            chat_names = list(st.session_state.chat_history.keys())
            selected_chat = st.selectbox("🔍 เลือกแชท", chat_names,
                                         index=chat_names.index(st.session_state.current_chat)
                                         if st.session_state.current_chat in chat_names else 0)

            st.session_state.current_chat = selected_chat
            st.session_state.messages = st.session_state.chat_history[selected_chat]

            # ปุ่มลบแชท (มีเงื่อนไขกัน Error)
            if st.button("🗑️ ลบแชทนี้"):
                if selected_chat in st.session_state.chat_history:
                    del st.session_state.chat_history[selected_chat]

                    # เลือกแชทใหม่หลังจากลบ
                    if st.session_state.chat_history:
                        st.session_state.current_chat = list(st.session_state.chat_history.keys())[0]
                    else:
                        st.session_state.current_chat = None
                        st.session_state.messages = []

                    st.success(f"ลบแชทสำเร็จ!")
                else:
                    st.warning("ไม่สามารถลบแชทนี้ได้!")

        # แสดงประวัติการแชทของแชทที่เลือก
        if st.session_state.chat_history and st.session_state.current_chat:
            st.subheader("💬 ประวัติการแชท")
            current_chat = st.session_state.current_chat
            if current_chat in st.session_state.chat_history:
                for role, content in st.session_state.chat_history[current_chat]:  # ✅ แก้ไขให้ใช้ tuple unpacking
                    st.write(f"{role}: {content}")
            else:
                st.write("ไม่มีประวัติการแชท")
        else:
            st.write("ไม่มีประวัติการแชท")

        # ปุ่มรีเซ็ตการแชท
        if st.button("🔄 รีเซ็ตการแชท"):
            st.session_state.chat_history = {}
            st.session_state.messages = []
            st.session_state.current_chat = None
            st.success("รีเซ็ตการแชทเรียบร้อยแล้ว")

    st.subheader("📂 จัดการเอกสาร")
    uploaded_files = st.file_uploader("อัปโหลดไฟล์ PDF", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            st.write(f"กำลังอัปโหลด {file_name}...")
            if upload_to_s3(uploaded_file, "lmskm", file_name):
                st.success(f"อัปโหลด {file_name} สำเร็จ!")

    st.subheader("ถามคำถามเกี่ยวกับกฎหมาย")
    category = st.selectbox("เลือกหมวดหมู่",
                            ["ทั่วไป", "กฎหมายเกี่ยวกับการค้ามนุษย์", "กฎหมาย พ.ร.บ คอมพิวเตอร์", "กฎหมายรัฐธรรมนูญ"])

    # ตรวจสอบและเตรียม session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = "default_chat"
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for role, content in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(content)

    # โหลดทรัพยากรทั้งหมดพร้อม spinner
    with st.spinner("กำลังโหลดทรัพยากร..."):
        tokenizer, model, collection, llm_client = load_all()

    # เรียกฟังก์ชันทดสอบ Milvus (เพื่อวินิจฉัยปัญหา)
    if collection and tokenizer and model:
        test_milvus_search(collection, tokenizer, model)

    if not (tokenizer and model and collection and llm_client):
        st.error("ระบบไม่สามารถโหลดทรัพยากรครบถ้วน กรุณาตรวจสอบ log ข้อผิดพลาด")
        return

    # รับข้อความค้นหาจากผู้ใช้
    query = st.chat_input("พิมพ์คำถามที่นี่:")
    if query:
        st.session_state.messages.append(("user", query))
        with st.chat_message("user"):
            st.markdown(query)
            st.session_state['start_time'] = time.time()

        # สร้าง embedding จาก query (แก้ไข: ไม่ส่ง category เข้าไปใน generate_embedding)
        with st.spinner("🔍 กำลังค้นหาข้อมูล..."):
            embedding = generate_embedding(query, tokenizer, model)
        if not embedding:
            st.error("❌ ไม่สามารถสร้าง embedding ได้จากคำถามที่ป้อนได้")
            return

        # ค้นหาใน Milvus โดยกำหนด top_k=5 (หรือปรับเปลี่ยนตามต้องการ)
        search_results = search_milvus(collection, embedding, top_k=5)
        processed_results = process_results(search_results)
        if not processed_results:
            st.warning("❌ ไม่พบข้อมูลที่เกี่ยวข้อง")
            return

        # รวมผลลัพธ์ค้นหาเป็น context สำหรับ LLM
        context = "\n\n".join([
            f"""📜 **กฎหมาย:** {res['law_name']} (📅 ปี: {res['year']})  
            🔖 **มาตรา:** {res['section']}  
            📝 **เนื้อหา:** {res['content']}  
            🎯 **คะแนน:** {res['score']:.4f}  
            ---"""
                for res in processed_results
        ])

        with st.spinner("🔍 กำลังค้นหาข้อมูล..."):
            # เรียกใช้ LLM เพื่อสร้างคำตอบ โดยใช้ context ที่ได้จาก Milvus
            response = generate_response(llm_client, query, context, category)

        # แสดงผลคำตอบจากระบบ (LLM)
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append(("assistant", response))
            st.session_state.chat_history[st.session_state.current_chat] = list(st.session_state.messages)

        # แสดงผลการค้นหา Top 5 จาก Milvus
        st.subheader("🔍 ผลการค้นหา (Top 5)")
        for res in processed_results:
            st.markdown(
                f"""📜 **กฎหมาย:** {res['law_name']} (📅 ปี: {res['year']})  
        🔖 **มาตรา:** {res['section']}  
        📝 **เนื้อหา:** {res['content']}  
        🎯 **คะแนน:** {res['score']:.4f}  
        ---
        """
            )

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    main()

# Logging สุดท้ายสำหรับตรวจสอบการใช้งานของ GPU
logger.info(f"CUDA Available in code: {torch.cuda.is_available()}")
logger.info(f"Using device: {DEVICE}")

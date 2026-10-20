__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
import numpy as np
from groq import Groq

# --- 1. إعداد الصفحة ---
st.set_page_config(page_title="NeuralDocs | AI", page_icon="🧠", layout="wide")

# --- 2. تصميم (Futuristic Dark Mode + Glassmorphism) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0d0f17 0%, #161a29 100%); color: #e2e8f0; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    [data-testid="stSidebar"] { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(15px); border-right: 1px solid rgba(255, 255, 255, 0.05); }
    .stChatMessage { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 15px; padding: 15px; margin-bottom: 15px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2); color: #e2e8f0; }
    .stButton>button { background: linear-gradient(90deg, #4f46e5, #06b6d4); color: white; border-radius: 10px; font-weight: 700; font-size: 16px; width: 100%; border: none; padding: 12px; box-shadow: 0 0 15px rgba(6, 182, 212, 0.4); transition: all 0.3s ease; }
    .stButton>button:hover { box-shadow: 0 0 25px rgba(6, 182, 212, 0.7); transform: translateY(-2px); }
    .stChatInput { border-radius: 15px; background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); color: white; }
    .neon-title { text-align: center; font-weight: 700; font-size: 2.8rem; background: linear-gradient(90deg, #06b6d4, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 0 20px rgba(139, 92, 246, 0.3); margin-bottom: 5px; }
    .sub-title { text-align: center; color: #94a3b8; font-size: 1rem; margin-bottom: 2rem; }
    .source-badge { display: inline-block; background: rgba(6, 182, 212, 0.1); color: #06b6d4; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; margin-right: 8px; margin-top: 8px; border: 1px solid rgba(6, 182, 212, 0.3); }
    </style>
""", unsafe_allow_html=True)

# --- 3. الإعدادات الأساسية ---
client_groq = Groq(api_key=st.secrets["GROQ_API_KEY"])

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_embedding_model()
client = chromadb.Client()
collection = client.get_or_create_collection(name="docs")

def split_text_enhanced(text, chunk_size=1200, overlap=150):
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            if len(para) > chunk_size:
                for i in range(0, len(para), chunk_size - overlap):
                    chunks.append(para[i:i + chunk_size])
                current_chunk = ""
            else:
                current_chunk = para + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# --- 4. الشريط الجانبي ---
with st.sidebar:
    st.markdown("<h3 style='color: #e2e8f0; font-size: 1.2rem; margin-bottom: 20px;'>⚙️ لوحة التحكم</h3>", unsafe_allow_html=True)
    
    with st.expander("📖 دليل الاستخدام"):
        st.markdown("""
        1. اضغط على **'حمل ملفك PDF'** واختر الملف الذي تريد تحليله.
        2. اضغط على زر **'🚀 تحليل ملفك'** وانتظر حتى تظهر رسالة النجاح.
        3. اطرح أي سؤال في صندوق المحادثة بالأسفل.
        *(ملاحظة: التطبيق يدعم ملفات PDF فقط حالياً)*
        """)

    uploaded_files = st.file_uploader("حمل ملفك PDF", type="pdf", accept_multiple_files=True)
    
    button_placeholder = st.empty()
    
    if button_placeholder.button("🚀 تحليل ملفك"):
        if uploaded_files:
            client.delete_collection(name="docs")
            collection = client.get_or_create_collection(name="docs")

            all_chunks, all_metadata, all_ids = [], [], []
            chunk_id = 0

            for pdf_file in uploaded_files:
                reader = PdfReader(pdf_file)
                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text()
                    if not text: continue
                    for chunk in split_text_enhanced(text):
                        all_chunks.append(chunk)
                        all_metadata.append({"source": pdf_file.name, "page": page_num, "chunk_id": chunk_id})
                        all_ids.append(f"id_{chunk_id}")
                        chunk_id += 1

            if all_chunks:
                with st.spinner("جاري تحليل الوثائق... (قد يستغرق دقيقة)"):
                    progress_bar = st.progress(0)
                    embeddings = embed_model.encode(all_chunks, show_progress_bar=False, batch_size=32).tolist()
                    progress_bar.progress(100)
                    
                    collection.add(embeddings=embeddings, documents=all_chunks, metadatas=all_metadata, ids=all_ids)
                    st.session_state.chunks = all_chunks
                    st.session_state.metadatas = all_metadata
                    st.session_state.bm25 = BM25Okapi([doc.lower().split() for doc in all_chunks])
                
                button_placeholder.empty()
                st.success(f"✅ النظام جاهز! تم تحليل {len(all_chunks)} مقطع.")
            else:
                st.error("⚠️ الملف فارغ أو غير قابل للقراءة.")
        else:
            st.error("⚠️ يرجى رفع ملف أولاً.")

    st.markdown("---")
    st.markdown("<p style='color: #475569; font-size: 0.8rem; text-align: center;'>NeuralDocs v1.0 | Powered by Groq</p>", unsafe_allow_html=True)

# --- 5. الواجهة الرئيسية ---
st.markdown("<h1 class='neon-title'>🧠 NeuralDocs</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>مساعدك الذكي لاستخراج المعرفة من وثائقك.</p>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🧠"):
        st.markdown("مرحباً بك. النظام جاهز. يمكنك سؤالي عن أي شيء يخص المستندات المرفوعة.")

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🧠"):
        st.markdown(message["content"])

# --- 6. منطق المحادثة ---
if prompt := st.chat_input("اكتب سؤالك هنا..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🧠"):
        if "bm25" not in st.session_state or not st.session_state.get("chunks"):
            st.warning("⚠️ الرجاء رفع وتحليل ملف PDF أولاً.")
        else:
            with st.spinner("جاري معالجة البيانات..."):
                query_embedding = embed_model.encode([prompt]).tolist()
                vector_results = collection.query(query_embeddings=query_embedding, n_results=20)

                bm25_scores = st.session_state.bm25.get_scores(prompt.lower().split())
                top_bm25_idx = np.argsort(bm25_scores)[-20:][::-1]

                fused = {}
                alpha = 0.5
                for rank, meta in enumerate(vector_results['metadatas'][0]):
                    cid = meta['chunk_id']
                    fused[cid] = fused.get(cid, 0) + alpha * (1 / (rank + 60))
                for rank, idx in enumerate(top_bm25_idx):
                    cid = st.session_state.metadatas[idx]['chunk_id']
                    fused[cid] = fused.get(cid, 0) + (1 - alpha) * (1 / (rank + 60))

                sorted_cids = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:5]
                id_to_chunk = {m['chunk_id']: (c, m) for c, m in zip(st.session_state.chunks, st.session_state.metadatas)}
                top_data = [id_to_chunk[cid] for cid, _ in sorted_cids if cid in id_to_chunk]

                if not top_data:
                    st.warning("لم أجد نتائج مناسبة.")
                else:
                    context = "\n\n---\n\n".join([c for c, _ in top_data])
                    sources = [m for _, m in top_data]

                    system_prompt = f"""أنت مساعد دقيق. أجب على السؤال بناءً على السياق التالي فقط. إذا لم تكن الإجابة موجودة في السياق، قل بوضوح: "لا أجد هذه المعلومة في الوثائق المرفقة." لا تخترع معلومات من عندك.
السياق:
{context}
"""
                    try:
                        response = client_groq.chat.completions.create(
                            model="openai/gpt-oss-120b",
                            messages=[
                                {'role': 'system', 'content': system_prompt},
                                {'role': 'user', 'content': prompt}
                            ]
                        )
                        answer = response.choices[0].message.content
                        st.markdown(answer)

                        source_html = "<div style='margin-top: 15px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 10px;'>"
                        source_html += "<span style='font-size: 0.85rem; color: #94a3b8; font-weight: 600;'>المصادر المستخدمة:</span><br>"
                        for src in sources:
                            source_html += f"<span class='source-badge'>📄 {src['source']} (صفحة {src['page']})</span>"
                        source_html += "</div>"
                        st.markdown(source_html, unsafe_allow_html=True)

                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"حدث خطأ في الاتصال: {e}")
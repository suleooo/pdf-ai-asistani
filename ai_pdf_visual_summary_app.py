import fitz  # PyMuPDF
import streamlit as st
from openai import OpenAI

# Sayfa ayarları
st.set_page_config(page_title="📄 AI Destekli PDF Asistanı", page_icon="🤖", layout="centered")

st.title("📄 Yapay Zeka PDF Asistanı")
st.markdown("Yükle, sor, cevabını al ve etkileyici bir görselle sun!")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# PDF Yükleme
st.markdown("### 📥 PDF Yükle")
uploaded_file = st.file_uploader("Bir PDF dosyası seç", type="pdf")

if uploaded_file:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    full_text = "\n".join([page.get_text() for page in doc])

    st.markdown("### ❓ Soru Sor")
    question = st.text_input("PDF içeriğiyle ilgili bir soru sor...")

    if question:
        with st.spinner("🤖 GPT yanıtlıyor..."):
            # PDF Soru-Cevap
            full_prompt = f"PDF içeriği:\n{full_text[:3000]}\n\nSoru: {question}"
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.7
            )
            gpt_answer = response.choices[0].message.content

        st.markdown("### 💬 GPT-4 Cevabı")
        st.write(gpt_answer)

        # Özet/Etki Cümlesi
        with st.spinner("🪄 Özetleniyor..."):
            summary_response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "Verilen metni yaratıcı, etkileyici ve kısa bir cümleye dönüştür."},
                    {"role": "user", "content": gpt_answer}
                ]
            )
            summary = summary_response.choices[0].message.content.strip()

        st.markdown("### ✍️ Etki Cümlesi")
        st.success(summary)

        # Görsel Üretimi (DALL·E 3)
        with st.spinner("🎨 Görsel oluşturuluyor..."):
            image_response = client.images.generate(
                model="dall-e-3",
                prompt=summary,
                size="1024x1024",
                quality="standard",
                n=1
            )
            image_url = image_response.data[0].url

        st.markdown("### 🖼️ Yapay Zekâ Tarafından Oluşturulan Görsel")
        st.image(image_url, caption=summary)

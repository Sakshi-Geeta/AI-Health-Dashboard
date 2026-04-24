import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import plotly.express as px
from datetime import datetime
import io
import re
from openai import OpenAI
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Health Dashboard", layout="wide")

# ---------------- GROQ SETUP ----------------
client = OpenAI(
    api_key=st.secrets["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

# ---------------- STYLE ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0b0f19;
    color: white;
}
.log-box {
    background-color: #111827;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 10px;
    font-family: monospace;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    tumor_model = tf.keras.models.load_model('models/model.h5')
    skin_model = tf.keras.models.load_model('skin_cancer_cnn.h5')
    return tumor_model, skin_model

tumor_model, skin_model = load_models()
brain_classes = ['pituitary', 'glioma', 'notumor', 'meningioma']

# ---------------- SESSION ----------------
for key, default in {
    "history": [],
    "chat_history": [],
    "signed_up": False,
    "user_name": "",
    "user_email": ""
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------- PDF ----------------
def generate_pdf(title, result, confidence, img):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    img_buffer = io.BytesIO()
    img.save(img_buffer, format="PNG")
    img_buffer.seek(0)

    content = [
        Paragraph(title, styles['Title']),
        Spacer(1, 10),
        Paragraph(f"Result: {result}", styles['Normal']),
        Paragraph(f"Confidence: {confidence:.2f}%", styles['Normal']),
        Paragraph(f"Generated on: {datetime.now()}", styles['Normal']),
        Spacer(1, 20),
        RLImage(img_buffer, width=300, height=300)
    ]

    doc.build(content)
    buffer.seek(0)
    return buffer

# ---------------- VALIDATION ----------------
def preprocess_tumor(img):
    img = img.convert("RGB").resize((128, 128))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def is_mri_like(img):
    img = img.convert("RGB")
    arr = np.array(img)

    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    color_diff = (np.mean(abs(r-g)) + np.mean(abs(r-b)) + np.mean(abs(g-b))) / 3

    gray = np.mean(arr, axis=2)
    variance = np.var(gray)

    # Relaxed (so real MRI doesn't fail)
    return color_diff < 20 and variance > 50

def predict_tumor(img):
    arr = preprocess_tumor(img)
    preds = tumor_model.predict(arr)[0]

    max_conf = np.max(preds)
    result = brain_classes[np.argmax(preds)]

    entropy = -np.sum(preds * np.log(preds + 1e-10))

    if max_conf < 0.70 or entropy > 1.2:
        return "Invalid / Not MRI", max_conf, preds

    return result, max_conf, preds

def is_skin_like(img):
    arr = np.array(img.convert("RGB"))

    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

    skin_pixels = ((r > 95) & (g > 40) & (b > 20) &
                   (r > g) & (r > b) &
                   (np.abs(r - g) > 15))

    skin_ratio = np.sum(skin_pixels) / (arr.shape[0] * arr.shape[1])

    return skin_ratio > 0.15


# ---------------- MODELS ----------------
def preprocess_brain(img):
    img = img.convert("RGB").resize((128, 128))
    return np.expand_dims(np.array(img) / 255.0, axis=0)

def predict_brain(img):
    preds = tumor_model.predict(preprocess_brain(img))[0]
    conf = np.max(preds)
    label = brain_classes[np.argmax(preds)]
    return label, conf, preds

def preprocess_skin(img):
    img = img.convert("RGB").resize((224, 224))
    return np.expand_dims(np.array(img) / 255.0, axis=0)

def predict_skin(img):
    pred = skin_model.predict(preprocess_skin(img))[0][0]
    if pred > 0.5:
        return "Malignant", pred, [1-pred, pred]
    return "Benign", 1-pred, [1-pred, pred]

# ---------------- GROQ ----------------
def groq_call(prompt):
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are Emoly, an empathetic mental wellness AI." "You never diagnose medical conditions. You respond in a warm, supportive, non-judgmental tone. " "You focus on emotional reflection, reassurance, and practical coping strategies."},
            {"role": "user", "content": prompt}
        ]
    )
    return res.choices[0].message.content

def generate_report(quiz_type, responses, journal, time, interest):
    prompt = f"""
User took {quiz_type} quiz.

Responses:
{responses}

Journal:
{journal}

Time: {time}
Interest: {interest}

Give emotional reflection + suggestions.
"""
    return groq_call(prompt)

def chat_response(msg):
    prompt = f"""
User: {msg}

Reply as Emoly:
empathetic, warm, supportive.
"""
    return groq_call(prompt)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧠 AI Dashboard")

page = st.sidebar.radio("Navigate", [
    "🏠 Home",
    "🧠 Brain Tumor Detection",
    "🧬 Skin Cancer Detection",
    "📜 History",
    "🌿 Emoly Quiz",
    "💬 Emoly Chat"
])

# ---------------- HOME ----------------
if page == "🏠 Home":

    st.title("🧠 AI Health Dashboard")
    st.markdown("### Smart Medical & Mental Wellness Platform")

    st.write(
        "An integrated AI system for early detection of health conditions and mental wellness support. "
        "Upload medical images, analyze results, and take care of your emotional well-being — all in one place."
    )

    st.markdown("---")

    # -------- FEATURE CARDS --------
    st.markdown("## 🚀 Core Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background:#1f2937; padding:20px; border-radius:12px; text-align:center;">
        <h4>🧠 Brain Tumor Detection</h4>
        <p>Detect and classify brain tumors from MRI scans using CNN.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background:#1f2937; padding:20px; border-radius:12px; text-align:center;">
        <h4>🧬 Skin Cancer Detection</h4>
        <p>Classify skin lesions as benign or malignant using AI.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background:#1f2937; padding:20px; border-radius:12px; text-align:center;">
        <h4>🌿 Emoly AI</h4>
        <p>AI-powered mental wellness quiz and emotional support chatbot.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # -------- ABOUT --------
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=200)

    with col2:
        st.markdown("## 📌 About This Project")
        st.write(
            "This platform combines healthcare AI and mental wellness support. "
            "It enables users to analyze medical images and receive intelligent insights, "
            "while also providing emotional support through an AI companion."
        )

        st.markdown("""
        **Key Highlights:**
        - AI-powered predictions
        - Interactive visualizations (charts & heatmaps)
        - Downloadable reports
        - Mental wellness tracking
        """)

    st.markdown("---")

    # -------- BRAIN FLASHCARDS --------
    st.markdown("## 🧠 Brain Tumor Awareness")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background:#1f2937; padding:15px; border-radius:12px;">
        <h4>🧾 What is it?</h4>
        <p>A brain tumor is an abnormal growth of cells in the brain that can affect normal brain function.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background:#1f2937; padding:15px; border-radius:12px;">
        <h4>🛡️ Prevention</h4>
        <p>While not fully preventable, reducing radiation exposure and maintaining a healthy lifestyle can help lower risks.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background:#1f2937; padding:15px; border-radius:12px;">
        <h4>🤖 How this helps</h4>
        <p>This system analyzes MRI scans using AI to assist in early detection and classification of tumors.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # -------- SKIN FLASHCARDS --------
    st.markdown("## 🧬 Skin Cancer Awareness")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background:#1f2937; padding:15px; border-radius:12px;">
        <h4>🧾 What is it?</h4>
        <p>Skin cancer is the abnormal growth of skin cells, commonly caused by exposure to ultraviolet (UV) radiation.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background:#1f2937; padding:15px; border-radius:12px;">
        <h4>🛡️ Prevention</h4>
        <p>Use sunscreen, avoid prolonged sun exposure, and regularly check your skin for unusual changes.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background:#1f2937; padding:15px; border-radius:12px;">
        <h4>🤖 How this helps</h4>
        <p>The AI model classifies skin images as benign or malignant, supporting early-stage detection.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")


# ---------------- BRAIN ----------------
elif page == "🧠 Brain Tumor Detection":

    st.title("🧠 Brain Tumor Detection")
    st.warning("⚠️ This AI tool is for educational purposes only and not a medical diagnosis.")

    file = st.file_uploader("Upload MRI", type=["jpg", "png", "jpeg"])

    if file:
        img = Image.open(file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(img, use_container_width=True)

        if not is_mri_like(img):
            st.error("⚠️ Not a valid MRI scan")
            st.stop()

        result, conf, preds = predict_tumor(img)

        with col2:
            if "Invalid" in result:
                st.error("Low confidence prediction")
                st.stop()

            st.success(f"Result: {result}")
            st.metric("Confidence", f"{conf*100:.2f}%")

        # Save history
        st.session_state.history.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "type": "Brain",
            "result": result,
            "confidence": f"{conf*100:.2f}%"
        })

        # Bar chart
        fig = px.bar(x=brain_classes, y=preds)
        st.plotly_chart(fig, use_container_width=True)

        # Heatmap
        gray = img.convert("L").resize((128,128))
        arr = np.array(gray)
        fig2 = px.imshow(arr, color_continuous_scale='hot')
        st.plotly_chart(fig2, use_container_width=True)

        # PDF
        def create_pdf():
            buffer = io.BytesIO()
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
            from reportlab.lib.styles import getSampleStyleSheet

            doc = SimpleDocTemplate(buffer)
            styles = getSampleStyleSheet()

            img_buf = io.BytesIO()
            img.save(img_buf, format="PNG")
            img_buf.seek(0)

            content = [
                Paragraph("Brain Tumor Report", styles['Title']),
                Spacer(1,10),
                Paragraph(f"Result: {result}", styles['Normal']),
                Paragraph(f"Confidence: {conf*100:.2f}%", styles['Normal']),
                Spacer(1,20),
                RLImage(img_buf, width=300, height=300)
            ]

            doc.build(content)
            buffer.seek(0)
            return buffer

        st.download_button("Download Report", create_pdf(), "brain_report.pdf")

# ---------------- SKIN ----------------
elif page == "🧬 Skin Cancer Detection":
    st.title("🧬 Skin Cancer Detection")
    st.warning("⚠️ This AI tool is for educational purposes only and not a medical diagnosis.")

    file = st.file_uploader("Upload Skin Image", type=["jpg","png","jpeg"])

    if file:
        img = Image.open(file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(img, use_container_width=True)

        if not is_skin_like(img):
            st.error("⚠️ This does not appear to be a valid skin lesion image")
            st.stop()

        with st.spinner("Analyzing skin image..."):
            label, conf, probs = predict_skin(img)

        with col2:
            st.success(label)
            st.metric("Confidence", f"{conf*100:.2f}%")
            # Confidence interpretation
        if conf > 0.85:
            st.success("High confidence prediction")
        elif conf > 0.70:
            st.info("Moderate confidence prediction")
        else:
            st.warning("Low confidence — result may not be reliable")
        
        st.markdown("### 🩺 Suggested Next Steps")
        st.write("- Consult a dermatologist")
        st.write("- Monitor changes in skin")
        st.write("- Avoid self-diagnosis")
        
        
        st.plotly_chart(px.bar(x=["Benign","Malignant"], y=probs), use_container_width=True, key="skin_bar")
        st.subheader("🔥 Heatmap")
        gray = img.convert("L").resize((224,224))
        st.plotly_chart(
            px.imshow(np.array(gray), color_continuous_scale='hot'),
            use_container_width=True,
            key="skin_heatmap"
        )

        st.subheader("📄 Download Report")
        pdf = generate_pdf("Skin Report", label, conf*100, img)
        st.download_button("Download Report", pdf, "skin_report.pdf")

        st.session_state.history.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "type": "Skin",
            "result": label,
            "confidence": f"{conf*100:.2f}%"
        })

# ---------------- EMOLY QUIZ ----------------
elif page == "🌿 Emoly Quiz":

    st.title("📝 Mental Wellness Quiz")

    quiz_type = st.selectbox("Choose a wellness area:", [
        "General Mental Health", "Depression", "Anxiety", "Stress", "Sleep Wellness", "Self-Esteem"
    ])

    quiz_questions = {
        "General Mental Health": [
            "How often do you feel emotionally balanced?",
            "Do you find it hard to concentrate?",
            "Are you feeling connected with your goals?"
        ],
        "Depression": [
            "How often do you feel hopeless?",
            "Do you enjoy old hobbies?",
            "How is your energy level?"
        ],
        "Anxiety": [
            "Do you feel nervous often?",
            "Do you experience racing thoughts?",
            "Can you calm yourself when overwhelmed?"
        ],
        "Stress": [
            "How frequently do you feel overwhelmed?",
            "How well do you manage pressure?",
            "Do you get headaches or fatigue often?"
        ],
        "Sleep Wellness": [
            "How many hours of sleep do you usually get?",
            "Do you wake up feeling rested?",
            "Do you use screens before bed?"
        ],
        "Self-Esteem": [
            "Do you often compare yourself to others?",
            "How do you react to compliments?",
            "Do you feel confident about your strengths?"
        ]
    }

    responses = []

    for i, question in enumerate(quiz_questions[quiz_type]):
        ans = st.radio(
            question,
            ["Never", "Rarely", "Sometimes", "Often"],
            key=f"quiz_{quiz_type}_{i}"
        )
        responses.append(f"{question} — {ans}")

    journal = st.text_area("Optional: How do you feel right now?")

    st.subheader("🌟 Lifestyle Context")
    time = st.slider("Time for self-care today?", 5, 60, 15)
    interest = st.radio("Preferred activity type?", ["physical", "creative", "mindful"])

    if st.button("✅ Get My Wellness Report"):

        full_input = "\n".join(responses)

        prompt = f"""
You're Emoly – an empathetic AI mental wellness companion.

User selected: {quiz_type}

Responses:
{full_input}

Journal:
{journal if journal else "No journal entry"}

Time available: {time} minutes
Preferred activity: {interest}

Generate:
1. Emotional reflection
2. Encouragement
3. 2-3 personalized suggestions
4. Friendly closing
"""

        with st.spinner("Emoly is writing your response..."):

            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are Emoly, an empathetic, supportive, "
                            "and non-judgmental AI mental wellness companion. "
                            "You help users reflect emotionally and give practical, gentle guidance. "
                            "You do not diagnose medical conditions."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            reply = res.choices[0].message.content

        st.markdown("### 💌 Emoly's Response")
        st.info(reply)

        st.download_button(
            "Download Wellness Report",
            reply,
            file_name="emoly_report.txt"
        )

# ---------------- EMOLY CHAT ----------------
elif page == "💬 Emoly Chat":
    st.title("💬 Chat with Emoly")

    if not st.session_state.signed_up:
        name = st.text_input("Name")
        email = st.text_input("Email")

        if st.button("Sign Up"):
            if name and re.match(r"[^@]+@[^@]+\.[^@]+", email):
                st.session_state.signed_up = True
                st.session_state.user_name = name
                st.rerun()

    else:
        # Show chat history
        for user_msg, bot_msg in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(user_msg)

            with st.chat_message("assistant"):
                st.write(bot_msg)

        user_msg = st.text_input("Message")

        if st.button("Send") and user_msg:

            with st.spinner("Emoly is thinking..."):

                res = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are Emoly, a warm, empathetic mental wellness AI chatbot. "
                                "You respond in a friendly, supportive, non-judgmental tone. "
                                "You help users reflect emotionally and feel better."
                            )
                        },
                        {
                            "role": "user",
                            "content": user_msg
                        }
                    ]
                )

                reply = res.choices[0].message.content

            st.session_state.chat_history.append((user_msg, reply))
            st.rerun()

# ---------------- HISTORY ----------------
elif page == "📜 History":
    st.title("📜 History")

    if st.session_state.history:
        for item in reversed(st.session_state.history):
            st.markdown(f"""
            <div class="log-box" style="border-left:4px solid #22c55e;">
            ▶ {item['time']} | {item['type']} <br>
            Result → {item['result']} <br>
            Confidence → {item['confidence']}
</div>
""", unsafe_allow_html=True)
    else:
        st.info("No history yet")


# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<div style="text-align:center; font-size:13px; color:#9ca3af; line-height:1.6;">
© 2026 Sakshi Geeta (2201). All rights reserved.<br>
Developed by Sakshi Geeta, Government Engineering College, Gandhinagar (GTU)
</div>
""", unsafe_allow_html=True)
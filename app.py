# flower_classification_app.py
import streamlit as st
from roboflow import Roboflow
from PIL import Image
import io
import google.generativeai as genai

# ------------------------------- 
# 1. Configuration
# ------------------------------- 
ROBOFLOW_API_KEY = "A7cI7giJIENZSR3kGPuL"
GEMINI_API_KEY = "AIzaSyAzr-oM8Hbze7RWky0PHmdY9p6_CWeZo1g"

# ------------------------------- 
# 2. Initialize Roboflow
# ------------------------------- 
@st.cache_resource
def load_roboflow_model():
    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace("kamal-kflqt").project("flower-classification-5qfoa")
        model = project.version(2).model
        return model
    except Exception as e:
        st.error(f"❌ Failed to load Roboflow model: {str(e)}")
        return None

model = load_roboflow_model()

if model is None:
    st.error("❌ Roboflow model failed to load. Please check your API key and internet connection.")
    st.stop()
else:
    st.success("✅ Roboflow model loaded!")

# ------------------------------- 
# 3. Initialize Gemini
# ------------------------------- 
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    st.success("✅ Gemini API configured!")
except Exception as e:
    st.error(f"❌ Failed to configure Gemini: {str(e)}")
    st.stop()

# ------------------------------- 
# 4. Helper Functions
# ------------------------------- 
def explain_flower(flower_name: str) -> str:
    """Get flower explanation and care tips from Gemini."""
    try:
        prompt = f"Explain the flower '{flower_name}' in simple terms and give tips on how to take care of it. Keep it concise (150 words max)."
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error getting flower info: {str(e)}"

def ask_question(flower_name: str, question: str) -> str:
    """Answer user questions about the flower using Gemini."""
    try:
        prompt = f"You are a botanist. Answer this question about the flower '{flower_name}': {question}. Keep your answer concise and helpful."
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error getting answer: {str(e)}"

# ------------------------------- 
# 5. Flower Icons
# ------------------------------- 
flower_icons = {
    "rose": "🌹",
    "daisy": "🌼",
    "sunflower": "🌻",
    "tulip": "🌷",
    "dandelion": "💛",
    "lily": "🌺",
}

# ------------------------------- 
# 6. Streamlit UI
# ------------------------------- 
st.title("🌸 Flower Classification & Q&A with Gemini")
st.markdown("Upload flower images to identify them and learn more!")

uploaded_files = st.file_uploader(
    "Upload flower images",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files):
        st.markdown("---")
        
        # Display image
        image = Image.open(io.BytesIO(uploaded_file.read()))
        st.image(image, caption=f"📷 {uploaded_file.name}", use_container_width="stretch")
        
        # Save temp image
        temp_path = f"temp_{idx}_{uploaded_file.name}"
        image.save(temp_path)
        
        # ------------------------------- 
        # 7. Predict with Roboflow
        # ------------------------------- 
        with st.spinner("🔍 Analyzing image..."):
            try:
                result = model.predict(temp_path).json()
                predictions = result.get('predictions', [])
                
                if predictions:
                    top_pred = predictions[0]
                    
                    # Handle nested predictions structure
                    if 'predictions' in top_pred and top_pred['predictions']:
                        inner_pred = top_pred['predictions'][0]
                        predicted_class = inner_pred.get('class', 'Unknown')
                        confidence = inner_pred.get('confidence', 0)
                    else:
                        predicted_class = top_pred.get('top', top_pred.get('class', 'Unknown'))
                        confidence = top_pred.get('confidence', 0)
                    
                    # Display prediction
                    icon = flower_icons.get(predicted_class.lower(), "🌸")
                    st.success(f"**Predicted Class:** {predicted_class} {icon}  |  **Confidence:** {confidence*100:.2f}%")
                    
                    # ------------------------------- 
                    # 8. Flower Explanation
                    # ------------------------------- 
                    with st.expander("🌿 Learn about this flower", expanded=True):
                        with st.spinner("Getting flower information..."):
                            flower_info = explain_flower(predicted_class)
                            st.write(flower_info)
                    
                    # ------------------------------- 
                    # 9. Interactive Q&A
                    # ------------------------------- 
                    st.markdown(f"### 💬 Ask about {predicted_class}")
                    
                    # Unique key for each image's question input
                    question_key = f"question_{idx}_{uploaded_file.name}"
                    user_question = st.text_input(
                        f"Your question about {predicted_class}:",
                        key=question_key,
                        placeholder="e.g., How often should I water it?"
                    )
                    
                    if user_question:
                        with st.spinner("🤔 Thinking..."):
                            answer = ask_question(predicted_class, user_question)
                            st.markdown("**🤖 Gemini's Answer:**")
                            st.info(answer)
                
                else:
                    st.warning("⚠️ No predictions found for this image. Try uploading a clearer flower image.")
                    
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
else:
    st.info("👆 Upload one or more flower images to get started!")

# Footer
st.markdown("---")
st.markdown("*Powered by Roboflow & Google Gemini*")
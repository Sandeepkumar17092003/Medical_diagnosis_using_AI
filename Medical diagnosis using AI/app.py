import streamlit as st
import pickle
from streamlit_option_menu import option_menu

# Page Configuration
st.set_page_config(
    page_title="HealthScope AI - Disease Prediction",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS Styling
st.markdown(f"""
    <style>
    .stSelectbox {{
        margin-bottom: 1rem;
    }}
    
    .stNumberInput, .stTextInput {{
        margin-bottom: 1.5rem;
    }}
    
    .stButton button {{
        background: linear-gradient(45deg, #4CAF50 0%, #45a049 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 25px;
        font-size: 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }}
    
    .stButton button:hover {{
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }}
    
    .header-title {{
        font-size: 2.5rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }}
    
    .section-header {{
        font-size: 1.8rem;
        color: #4CAF50;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #4CAF50;
        padding-bottom: 0.5rem;
    }}
    
    .input-label {{
        font-size: 1rem;
        color: #ffffff !important;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }}
    
    .prediction-card {{
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }}
    </style>
""", unsafe_allow_html=True)

# Background Image Configuration
background_image_url = "https://www.strategyand.pwc.com/m1/en/strategic-foresight/sector-strategies/healthcare/ai-powered-healthcare-solutions/img01-section1.jpg"
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background: linear-gradient(rgba(0, 0, 0, 0.85), rgba(0, 0, 0, 0.85)),
                url({background_image_url});
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_models():
    return {
        'diabetes': pickle.load(open('Models/diabetes_model.sav', 'rb')),
        'heart_disease': pickle.load(open('Models/heart_disease_model.sav', 'rb')),
        'parkinsons': pickle.load(open('Models/parkinsons_model.sav', 'rb')),
        'lung_cancer': pickle.load(open('Models/lungs_disease_model.sav', 'rb')),
        'thyroid': pickle.load(open('Models/Thyroid_model.sav', 'rb'))
    }
models = load_models()

# Sidebar Configuration
with st.sidebar:
    st.header("🧭 Navigation")
    selected = option_menu(
        menu_title=None,
        options=['Diabetes', 'Heart Disease', 'Parkinsons', 
                'Lung Cancer', 'Hypo-Thyroid'],
        icons=['activity', 'heart-pulse', 'person-walking',
              'lungs', 'capsule'],
        default_index=0,
        styles={
            "container": {"padding": "0!important"},
            "nav-link": {"font-size": "16px", "margin": "10px 0"}
        }
    )
    
    st.header("ℹ️ About")
    st.write("""
    HealthScope AI is an advanced predictive healthcare solution 
    using machine learning to assess potential health risks.
    Always consult a medical professional for diagnosis.
    """)

# Input Field Generator
def create_input(label, key, min_val=None, max_val=None, step=1):
    st.markdown(f'<div class="input-label">{label}</div>', unsafe_allow_html=True)
    return st.number_input(
        label, key=key, 
        min_value=min_val, max_value=max_val, step=step,
        label_visibility="collapsed"
    )

# Main Content
st.markdown(f'<h1 class="header-title">HealthScope AI 🧠</h1>', unsafe_allow_html=True)

# Diabetes Prediction
if selected == 'Diabetes':
    st.markdown('<div class="section-header">🩸 Diabetes Risk Assessment</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        pregnancies = create_input('Pregnancies', 'preg', 0, 20)
        glucose = create_input('Glucose (mg/dL)', 'gluc', 0, 300)
    with col2:
        bp = create_input('Blood Pressure (mmHg)', 'bp', 0, 200)
        skin_thickness = create_input('Skin Thickness (mm)', 'skin', 0, 100)
    with col3:
        insulin = create_input('Insulin Level (μU/ml)', 'insulin', 0, 1000)
        bmi = create_input('Body Mass Index (BMI)', 'bmi', 0.0, 100.0, 0.1)
    
    with st.expander("Advanced Parameters"):
        col4, col5 = st.columns(2)
        with col4:
            dpf = create_input('Diabetes Pedigree Function', 'dpf', 0.0, 3.0, 0.01)
        with col5:
            age = create_input('Age (Years)', 'age', 0, 120)

    if st.button('🔍 Analyze Diabetes Risk'):
        with st.spinner('Analyzing health parameters...'):
            prediction = models['diabetes'].predict([[pregnancies, glucose, bp, 
                                                     skin_thickness, insulin, bmi, dpf, age]])
            result = '⚠️ High Diabetes Risk Detected' if prediction[0] == 1 else '✅ Low Diabetes Risk'
            color = "#ff4b4b" if prediction[0] == 1 else "#4CAF50"
            st.markdown(f"""
                <div class="prediction-card">
                    <h3 style="color: {color}; text-align: center;">{result}</h3>
                    <p style="text-align: center; margin-top: 1rem;">
                        Always consult with a healthcare professional for detailed analysis.
                    </p>
                </div>
            """, unsafe_allow_html=True)

# Heart Disease Prediction
elif selected == 'Heart Disease':
    st.markdown('<div class="section-header">❤️ Heart Disease Risk Assessment</div>', unsafe_allow_html=True)
    
    cols = st.columns(4)
    inputs = []
    with cols[0]:
        inputs.append(create_input('Age (Years)', 'h_age', 0, 120))
        inputs.append(create_input('Sex (1=M, 0=F)', 'h_sex', 0, 1))
        inputs.append(create_input('Chest Pain Type', 'h_cp', 0, 3))
    with cols[1]:
        inputs.append(create_input('Resting BP (mmHg)', 'h_bp', 0, 300))
        inputs.append(create_input('Cholesterol (mg/dl)', 'h_chol', 0, 600))
        inputs.append(create_input('Fasting Sugar >120', 'h_fbs', 0, 1))
    with cols[2]:
        inputs.append(create_input('Resting ECG', 'h_restecg', 0, 2))
        inputs.append(create_input('Max Heart Rate', 'h_thalach', 0, 250))
        inputs.append(create_input('Exercise Angina', 'h_exang', 0, 1))
    with cols[3]:
        inputs.append(create_input('ST Depression', 'h_oldpeak', 0.0, 10.0, 0.1))
        inputs.append(create_input('Slope', 'h_slope', 0, 2))
        inputs.append(create_input('Fluoroscopy Vessels', 'h_ca', 0, 3))
        inputs.append(create_input('Thalassemia', 'h_thal', 0, 3))

    if st.button('🔍 Analyze Heart Disease Risk'):
        with st.spinner('Evaluating cardiovascular health...'):
            prediction = models['heart_disease'].predict([inputs])
            result = '⚠️ High Heart Disease Risk Detected' if prediction[0] == 1 else '✅ Low Heart Disease Risk'
            color = "#ff4b4b" if prediction[0] == 1 else "#4CAF50"
            st.markdown(f"""
                <div class="prediction-card">
                    <h3 style="color: {color}; text-align: center;">{result}</h3>
                    <p style="text-align: center; margin-top: 1rem;">
                        Regular cardiac checkups are recommended for maintaining heart health.
                    </p>
                </div>
            """, unsafe_allow_html=True)

# Parkinson's Prediction
elif selected == 'Parkinsons':
    st.markdown('<div class="section-header">🧠 Parkinson\'s Disease Assessment</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fo = create_input('MDVP:Fo(Hz)', 'p_fo', 0.0, 3000.0, 0.1)
        fhi = create_input('MDVP:Fhi(Hz)', 'p_fhi', 0.0, 3000.0, 0.1)
        flo = create_input('MDVP:Flo(Hz)', 'p_flo', 0.0, 3000.0, 0.1)
        jitter_percent = create_input('MDVP:Jitter(%)', 'p_jit_per', 0.0, 1.0, 0.001)
    with col2:
        jitter_abs = create_input('MDVP:Jitter(Abs)', 'p_jit_abs', 0.0, 0.1, 0.0001)
        rap = create_input('MDVP:RAP', 'p_rap', 0.0, 0.2, 0.001)
        ppq = create_input('MDVP:PPQ', 'p_ppq', 0.0, 0.2, 0.001)
        ddp = create_input('Jitter:DDP', 'p_ddp', 0.0, 0.5, 0.001)

    if st.button('🔍 Analyze Parkinson\'s Risk'):
        with st.spinner('Analyzing voice parameters...'):
            prediction = models['parkinsons'].predict([[fo, fhi, flo, jitter_percent, 
                                                        jitter_abs, rap, ppq, ddp]])
            result = '⚠️ Parkinson\'s Disease Likely' if prediction[0] == 1 else '✅ Low Parkinson\'s Risk'
            color = "#ff4b4b" if prediction[0] == 1 else "#4CAF50"
            st.markdown(f"""
                <div class="prediction-card">
                    <h3 style="color: {color}; text-align: center;">{result}</h3>
                    <p style="text-align: center; margin-top: 1rem;">
                        Neurological consultations recommended for comprehensive assessment.
                    </p>
                </div>
            """, unsafe_allow_html=True)

# Lung Cancer Prediction
elif selected == 'Lung Cancer':
    st.markdown('<div class="section-header">🫁 Lung Cancer Risk Assessment</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = create_input('Gender (1=M, 0=F)', 'lc_gender', 0, 1)
        age = create_input('Age (Years)', 'lc_age', 0, 120)
    with col2:
        smoking = create_input('Smoking (1=Yes)', 'lc_smoking', 0, 1)
        yellow_fingers = create_input('Yellow Fingers', 'lc_yellow', 0, 1)
    with col3:
        anxiety = create_input('Anxiety (1=Yes)', 'lc_anxiety', 0, 1)
        peer_pressure = create_input('Peer Pressure', 'lc_peer', 0, 1)

    if st.button('🔍 Analyze Lung Cancer Risk'):
        with st.spinner('Evaluating respiratory health markers...'):
            prediction = models['lung_cancer'].predict([[gender, age, smoking, yellow_fingers, 
                                                        anxiety, peer_pressure]])
            result = '⚠️ High Lung Cancer Risk Detected' if prediction[0] == 1 else '✅ Low Lung Cancer Risk'
            color = "#ff4b4b" if prediction[0] == 1 else "#4CAF50"
            st.markdown(f"""
                <div class="prediction-card">
                    <h3 style="color: {color}; text-align: center;">{result}</h3>
                    <p style="text-align: center; margin-top: 1rem;">
                        Recommended to consult a pulmonologist for detailed screening.
                    </p>
                </div>
            """, unsafe_allow_html=True)

# Thyroid Prediction
elif selected == 'Hypo-Thyroid':
    st.markdown('<div class="section-header">🦋 Thyroid Function Assessment</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age = create_input('Age (Years)', 'thy_age', 0, 120)
        sex = create_input('Sex (1=M, 0=F)', 'thy_sex', 0, 1)
    with col2:
        on_thyroxine = create_input('On Thyroxine', 'thy_thyrox', 0, 1)
        tsh = create_input('TSH Level', 'thy_tsh', 0.0, 100.0, 0.1)
    with col3:
        t3 = create_input('T3 Level', 'thy_t3', 0.0, 10.0, 0.01)
        tt4 = create_input('TT4 Level', 'thy_tt4', 0.0, 500.0, 0.1)

    if st.button('🔍 Analyze Thyroid Function'):
        with st.spinner('Evaluating thyroid parameters...'):
            prediction = models['thyroid'].predict([[age, sex, on_thyroxine, tsh, t3, tt4]])
            result = '⚠️ Hypo-Thyroid Condition Detected' if prediction[0] == 1 else '✅ Normal Thyroid Function'
            color = "#ff4b4b" if prediction[0] == 1 else "#4CAF50"
            st.markdown(f"""
                <div class="prediction-card">
                    <h3 style="color: {color}; text-align: center;">{result}</h3>
                    <p style="text-align: center; margin-top: 1rem;">
                        Endocrine system evaluation recommended for comprehensive analysis.
                    </p>
                </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="text-align: center; color: #ffffff; margin-top: 3rem;">
        <hr style="border-color: #4CAF50;">
        <p>HealthScope AI v1.0 | Predictive Healthcare Solution</p>
        <p style="font-size: 0.8rem;">Note: This tool provides probabilistic assessments and should not be used as a substitute for professional medical advice.</p>
    </div>
""", unsafe_allow_html=True)
import streamlit as st
import requests
import json

# Page Configuration with a clean theme
st.set_page_config(
    page_title="AI & Data Engineering Control Center",
    page_icon="⚡",
    layout="wide"
)

# Custom Styling to match a clean Slate and Teal tech aesthetic
st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #64748B; margin-bottom: 2rem; }
    .card { background-color: #F8FAFC; padding: 1.5rem; border-radius: 0.5rem; border: 1px solid #E2E8F0; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⚡ Core AI Stack Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Interact with SQL, NoSQL, Vector Databases, ML, and RAG in one execution slice.</div>', unsafe_allow_html=True)

# Define the FastAPI URL endpoint
BACKEND_URL = "http://127.0.0.1:8000/process-pipeline"

# Setup Layout Columns split between Input Panel and Live Database Outputs
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("📥 1. Simulation Input Engine")
    
    with st.form("pipeline_form"):
        user_id = st.text_input("User Identification ID", value="user_dev_01", help="Maps relational profile entries inside SQL.")
        
        c1, c2 = st.columns(2)
        with c1:
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=250.0, step=50.0)
        with c2:
            login_failures = st.number_input("Recent Login Failures", min_value=0, value=0, step=1, 
                                             help="High numbers combined with high amounts trigger the ML Fraud Classifier.")
        
        st.markdown("**NoSQL Dynamic Payload Metadata**")
        device = st.selectbox("Client Device Signature", ["MacBook Pro", "iPhone 15", "Linux Workstation", "Windows Desktop"])
        ip_location = st.text_input("IP Geolocation Tracer", value="Dubai, UAE")
        latency = st.slider("Network Gateway Latency (ms)", min_value=1.0, max_value=200.0, value=12.5)
        
        st.markdown("**Vector Knowledge Base & RAG Query**")
        product_notes = st.text_area("Product Description Context", value="Premium AI Architecture Strategy Blueprint Edition with native vector indexing support.")
        user_question = st.text_input("Ask a Contextual Question", value="What security configuration status exists for my product notes?")
        
        submit_btn = st.form_submit_button("Execute End-to-End Pipeline", type="primary")

with col2:
    st.subheader("🖥️ 2. Live Multi-Database Execution Metrics")
    
    if submit_btn:
        # Reconstruct dynamic input into structured JSON body payload
        payload = {
            "user_id": user_id,
            "amount": amount,
            "login_failures": int(login_failures),
            "meta_payload": {
                "device": device,
                "ip_location": ip_location,
                "latency_ms": latency
            },
            "product_description": product_notes,
            "user_question": user_question
        }
        
        with st.spinner("Processing transaction across pipeline layers..."):
            try:
                response = requests.post(BACKEND_URL, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    st.success("✅ Architecture Layer Loop Executed Successfully!")
                    
                    # Highlight ML inference results
                    fraud_res = data.get("fraud_classification_result", "approved")
                    if fraud_res == "flagged_fraud":
                        st.error(f"🚨 **ML Classifier Alert:** Transaction status evaluates to **{fraud_res.upper()}**")
                    else:
                        st. those_good = st.info(f"🟢 **ML Classifier Alert:** Transaction status evaluates to **{fraud_res.upper()}**")
                    
                    # Metric Displays mapping directly back to core learning layers
                    m1, m2, m3 = st.columns(3)
                    m1.metric(label="SQL Generated Order ID", value=f"#{data.get('order_id')}")
                    m2.metric(label="NoSQL Documents Inserted", value=f"{data.get('nosql_records_captured')} Record")
                    m3.metric(label="Pipeline State", value=data.get("pipeline_execution_status").upper())
                    
                    st.write("---")
                    
                    # RAG Prompt Output Visualization Area
                    st.markdown("### 🤖 Synthesized Contextual RAG Prompt Context")
                    st.caption("This contains the raw, augmented data profile ready for LLM consumption:")
                    st.code(data.get("engineered_rag_prompt"), language="markdown")
                    
                else:
                    st.error(f"Backend Server returned status code: {response.status_code}")
                    st.json(response.json())
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Critical Connection Failure: Could not reach the FastAPI backend server layer. Please verify that `main.py` is currently running locally on port 8000.")
    else:
        st.info("Fill out the transaction metrics configuration details on the left dashboard and click 'Execute End-to-End Pipeline' to visualize data flow transformations across your infrastructure stack.")

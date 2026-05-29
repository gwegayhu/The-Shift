import streamlit as st
import os

# 1. Force ensure local data directory structure exists in the cloud container
os.makedirs("data", exist_ok=True)

# 2. Direct absolute imports from your backend framework architecture
from src.db_sql import SQLManager
from src.db_nosql import NoSQLManager
from src.db_vector import VectorManager
from src.service_ml import FraudModel
from src.service_rag import RAGEngine

# 3. Cache the initializations so they don't reload on every single button click
@st.cache_resource
def initialize_core_ai_stack():
    sql = SQLManager()
    nosql = NoSQLManager()
    vector = VectorManager()
    # Seed initial knowledge base record
    vector.index_product_notes("PROD_01", "Premium AI Architecture Strategy Blueprint Edition.")
    ml = FraudModel()
    rag = RAGEngine(sql, vector)
    return sql, nosql, vector, ml, rag

sql_db, nosql_db, vector_db, ml_model, rag_engine = initialize_core_ai_stack()

# ... Keep your standard page config layout and input forms exactly the same ...

if submit_btn:
    with st.spinner("Processing transaction across pipeline layers..."):
        # 4. SWAP OUT THE REQUESTS.POST CALL FOR DIRECT PYTHON LOGIC:
        
        # Run Machine Learning inference
        is_fraud = ml_model.predict_fraud_risk(amount, login_failures)
        status = "flagged_fraud" if is_fraud else "approved"

        # Record structured data to SQL
        order_id = sql_db.add_order(user_id, amount, status)

        # Log telemetry to NoSQL document store
        nosql_db.log_clickstream(order_id, {
            "device": device,
            "ip_location": ip_location,
            "latency_ms": latency
        })

        # Synthesize contextual RAG prompt
        rag_prompt = rag_engine.generate_contextual_prompt(user_id, user_question)
        
        # 5. Display metrics directly using the locally generated data profile
        st.success("✅ Architecture Layer Loop Executed Successfully inside Cloud Container!")
        if status == "flagged_fraud":
            st.error(f"🚨 **ML Classifier Alert:** Transaction status evaluates to **{status.upper()}**")
        else:
            st.info(f"🟢 **ML Classifier Alert:** Transaction status evaluates to **{status.upper()}**")
            
        m1, m2, m3 = st.columns(3)
        m1.metric(label="SQL Generated Order ID", value=f"#{order_id}")
        m2.metric(label="NoSQL Documents Inserted", value="1 Record")
        m3.metric(label="Pipeline State", value="SUCCESS")
        
        st.write("---")
        st.markdown("### 🤖 Synthesized Contextual RAG Prompt Context")
        st.code(rag_prompt, language="markdown")

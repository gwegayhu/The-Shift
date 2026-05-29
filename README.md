# The Shift

To master the modern engineering stack, you need to see exactly how these data layers connect in clean, minimal Python code.

Here is the repository setup. It builds a mini application that tracks an **E-Commerce Order System with AI Insights**. 
It uses standard, lightweight local libraries (`sqlite3` for SQL, `tinydb` for NoSQL, `chromadb` for Vector, and `scikit-learn` for ML).
So that you can run the entire stack locally with zero external dependencies or API keys.

---

## 1. Project Directory Layout

Create this exact folder structure on your local machine:

```text
modern-engineer-stack/
├── data/                  # Auto-generated local database stores (git-ignored)
├── src/
│   ├── __init__.py
│   ├── db_sql.py          # 1. Structured relational data storage
│   ├── db_nosql.py        # 2. Dynamic, schema-less logs & metadata
│   ├── db_vector.py       # 3. Semantic similarity embeddings
│   ├── service_ml.py      # 4. Classic Machine Learning (Fraud Classifier)
│   └── service_rag.py     # 5. RAG Engine combining SQL + Vector context
├── app.py                 # Application file
├── main.py                # Asynchronous API Engine (FastAPI application gateway)
├── Dockerfile             # Multi-stage container wrapper
├── requirements.txt       # Unified Python dependencies
└── README.md              # Instructions & Learning Blueprint

```

---

## 2. Core Code Implementation Files

### `requirements.txt`

```text
fastapi==0.111.0
uvicorn==0.30.1
tinydb==4.8.0
chromadb==0.5.0
scikit-learn==1.5.0
streamlit==1.35.0
requests==2.32.3

```

### `src/db_sql.py` (The SQL Layer)

```python
import sqlite3
from typing import Dict, Any, List

class SQLManager:
    """Manages transactional relational data using local SQLite."""
    def __init__(self, db_path: str = "data/app.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    amount REAL NOT NULL,
                    status TEXT NOT NULL
                )
            """)
            conn.commit()

    def add_order(self, user_id: str, amount: float, status: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO orders (user_id, amount, status) VALUES (?, ?, ?)",
                (user_id, amount, status)
            )
            conn.commit()
            return cursor.lastrowid

    def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Executes relational aggregations (Must-Know SQL concept)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*), SUM(amount) FROM orders WHERE user_id = ?", (user_id,)
            )
            count, total = cursor.fetchone()
            return {"total_orders": count or 0, "total_spent": total or 0.0}

```

### `src/db_nosql.py` (The NoSQL Layer)

```python
from tinydb import TinyDB, Query
from typing import Dict, Any, List

class NoSQLManager:
    """Manages dynamic, unstructured metadata payloads using a JSON document store."""
    def __init__(self, db_path: str = "data/metadata.json"):
        self.db = TinyDB(db_path)

    def log_clickstream(self, order_id: int, dynamic_payload: Dict[str, Any]):
        """Stores schema-less payloads (device signatures, click maps, nested JSON)."""
        dynamic_payload["order_id"] = order_id
        self.db.insert(dynamic_payload)

    def get_logs_by_order(self, order_id: int) -> List[Dict[str, Any]]:
        Log = Query()
        return self.db.search(Log.order_id == order_id)

```

### `src/db_vector.py` (The Vector Layer)

```python
import chromadb
from typing import List, Dict, Any

class VectorManager:
    """Handles semantic text processing and nearest-neighbor vector space lookups."""
    def __init__(self, path: str = "data/vector_store"):
        # Local embedded vector database initialization
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name="product_knowledge")

    def index_product_notes(self, product_id: str, text_description: str):
        """Generates vectors and indexes text content automatically."""
        self.collection.add(
            documents=[text_description],
            ids=[product_id]
        )

    def search_similar_products(self, query: str, top_n: int = 1) -> List[str]:
        """Performs a mathematical cosine similarity calculation across embeddings."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_n
        )
        return results["documents"][0] if results["documents"] else []

```

### `src/service_ml.py` (The AI/ML Layer)

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

class FraudModel:
    """A classic machine learning classification pipeline predicting transaction safety."""
    def __init__(self):
        self.model = LogisticRegression()
        self._train_mock_model()

    def _train_mock_model(self):
        # Features matrix: [Amount, FailureCount]
        X = np.array([[10.0, 0], [15.0, 0], [5000.0, 4], [3500.0, 3], [20.0, 0], [4500.0, 5]])
        # Labels vector: 0 = Clear, 1 = Fraud flag
        y = np.array([0, 0, 1, 1, 0, 1])
        self.model.fit(X, y)

    def predict_fraud_risk(self, amount: float, login_failures: int) -> bool:
        """Evaluates live input metrics against the trained classification boundary."""
        prediction = self.model.predict([[amount, login_failures]])
        return bool(prediction[0])

```

### `src/service_rag.py` (The RAG Engine Layer)

```python
from typing import Dict, Any

class RAGEngine:
    """Retrieves multi-source data context to build a deterministic LLM context injection prompt."""
    def __init__(self, sql: Any, vector: Any):
        self.sql = sql
        self.vector = vector

    def generate_contextual_prompt(self, user_id: str, user_query: str) -> str:
        # Step 1: Retrieve relational database metrics
        stats = self.sql.get_user_analytics(user_id)
        
        # Step 2: Retrieve semantic vector space documentation matching query intent
        matched_docs = self.vector.search_similar_products(user_query, top_n=1)
        doc_context = matched_docs[0] if matched_docs else "No matching historical items."

        # Step 3: Augment data profiles cleanly into a structured inference frame
        prompt_template = f"""
        [CONTEXT ENVIRONMENT SYSTEM]
        User History Profile: Total Spent = ${stats['total_spent']:.2f} | Total Invoices = {stats['total_orders']}
        Relevant Historical Product Details: {doc_context}

        [USER INQUIRY]
        {user_query}

        [INSTRUCTION]
        Synthesize an insightful profile answer combining the data metrics and documentation above.
        """
        return prompt_template.strip()

```
Create app.py (The Interactive UI Layer)
Python
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
3. Launching Your Ecosystem
To see the dashboard communicate with your database backend, run them side-by-side:

Step 1: Fire up the FastAPI Backend
Open a terminal window inside your project folder, activate your virtual environment, and boot up your API engine:

Bash
uvicorn main:app --reload --port 8000
Step 2: Fire up the Streamlit UI
Open a second terminal window inside the same folder, activate your virtual environment, and launch the user interface app:

Bash
streamlit run app.py
Streamlit will automatically open a fresh browser tab at http://localhost:8501.

 
### `main.py` (The Deployment Layer)

```python
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.db_sql import SQLManager
from src.db_nosql import NoSQLManager
from src.db_vector import VectorManager
from src.service_ml import FraudModel
from src.service_rag import RAGEngine

# Initialize standard directory storage parameters safely
os.makedirs("data", exist_ok=True)

app = FastAPI(title="Unified AI Engineering Learning Stack", version="2026.1")

# Single global injection configuration
sql_db = SQLManager()
nosql_db = NoSQLManager()
vector_db = VectorManager()
ml_model = FraudModel()
rag_engine = RAGEngine(sql_db, vector_db)

class TransactionRequest(BaseModel):
    user_id: str
    amount: float
    login_failures: int
    meta_payload: dict
    product_description: str
    user_question: str

@app.on_event("startup")
def seed_initial_knowledge():
    vector_db.index_product_notes("PROD_01", "Premium AI Architecture Strategy Blueprint Edition.")

@app.post("/process-pipeline")
async def process_pipeline(payload: TransactionRequest):
    # 1. Run Machine Learning inference check first
    is_fraud = ml_model.predict_fraud_risk(payload.amount, payload.login_failures)
    status = "flagged_fraud" if is_fraud else "approved"

    # 2. Persist transaction data into SQL relational rows
    order_id = sql_db.add_order(payload.user_id, payload.amount, status)

    # 3. Stream deep telemetry logs safely into a flexible NoSQL Document Store
    nosql_db.log_clickstream(order_id, payload.meta_payload)

    # 4. Synthesize structural contextual prompt via RAG pipeline logic
    rag_prompt = rag_engine.generate_contextual_prompt(payload.user_id, payload.user_question)

    return {
        "pipeline_execution_status": "success",
        "order_id": order_id,
        "fraud_classification_result": status,
        "nosql_records_captured": len(nosql_db.get_logs_by_order(order_id)),
        "engineered_rag_prompt": rag_prompt
    }

```

### `Dockerfile`

```dockerfile
# Multi-stage production builds minimize deployment layer footprints
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim AS runner
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
EXPOSE 8000
CMD ["uvicorn", "main.py:app", "--host", "0.0.0.0", "--port", "8000"]

```

---

## 3. Local Verification Run Guide

Execute these clean CLI steps inside your tracking repository directory to run your end-to-end sandbox:

```bash
# Set up a clean, isolated virtual workspace environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start your FastAPI gateway local application instance
uvicorn main:app --reload --port 8000

```

Open a second terminal window and post a live data stream request payload using standard curl to view your entire architecture in action:

```bash
curl -X POST "http://127.0.0.1:8000/process-pipeline" \
     -H "Content-Type: application/json" \
     -d '{
       "user_id": "user_dev_01",
       "amount": 4200.00,
       "login_failures": 3,
       "meta_payload": {"device": "MacBookPro", "ip_location": "Dubai", "latency_ms": 14.2},
       "product_description": "AI Blueprint Enterprise Suite",
       "user_question": "What is the security status on my product notes?"
     }'

```

---

## 4. Key Architectural Lessons Built In

By studying this mini-codebase, you learn how structural layers connect:

* **The ML Engine (`service_ml.py`)** flags this transaction as high-risk (`flagged_fraud`) because your transaction amount ($4200) and login failures (3) cross the learned classification threshold matrix.
* **The SQL Manager (`db_sql.py`)** records the data to compute transaction analytics over time using structured grouping queries.
* **The NoSQL Layer (`db_nosql.py`)** captures the dynamic browser telemetry without needing a rigid schema alteration.
* **The RAG Engine (`service_rag.py`)** extracts these metrics alongside your vector embeddings, packing them into an optimized prompt context window ready for an LLM generation pipeline.

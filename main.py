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

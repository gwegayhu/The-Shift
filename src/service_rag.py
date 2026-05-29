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

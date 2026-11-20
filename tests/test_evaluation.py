"""
Tests for RAG Evaluation Module.
Verifies calculation of Precision@K, Recall@K, Reciprocal Rank (MRR),
Faithfulness, Answer Relevance, and CSV score persistence.
"""

import os
import pytest
import pandas as pd
from backend.evaluation import RAGEvaluator
from backend.rag_pipeline import INSUFFICIENT_CONTEXT_MESSAGE


def test_retrieval_metrics_exact_hit():
    """Verify Precision@K, Recall@K, and Reciprocal Rank with exact top-1 match."""
    retrieved = [41, 10, 20, 30]
    relevant = [41]

    prec, rec, rr = RAGEvaluator.compute_retrieval_metrics(retrieved, relevant, k=4)
    assert prec == 0.25  # 1 out of 4 retrieved chunks is relevant
    assert rec == 1.0    # 100% of relevant pages found
    assert rr == 1.0     # Relevant page is rank 1


def test_retrieval_metrics_rank_two_hit():
    """Verify Reciprocal Rank is 0.5 when relevant page is at rank 2."""
    retrieved = [10, 41, 20, 30]
    relevant = [41]

    prec, rec, rr = RAGEvaluator.compute_retrieval_metrics(retrieved, relevant, k=4)
    assert prec == 0.25
    assert rec == 1.0
    assert rr == 0.5     # 1 / 2 = 0.5


def test_retrieval_metrics_out_of_domain():
    """Verify metrics return 1.0 when relevant is empty and nothing is retrieved."""
    prec, rec, rr = RAGEvaluator.compute_retrieval_metrics([], [], k=4)
    assert prec == 1.0
    assert rec == 1.0
    assert rr == 1.0


def test_faithfulness_grounded_answer():
    """Verify faithfulness computation for grounded text."""
    contexts = [
        "SECTION 9.1: When a Severity-1 (P0) alert triggers in PagerDuty, L1 has 5 minutes."
    ]
    answer = "Under Section 9.1, when a Severity-1 (P0) alert triggers, L1 has 5 minutes."

    score = RAGEvaluator.compute_faithfulness(answer, contexts, is_sufficient=True)
    assert score >= 0.8


def test_faithfulness_insufficient_context():
    """Verify faithfulness is 1.0 when insufficient context is correctly stated."""
    score = RAGEvaluator.compute_faithfulness(INSUFFICIENT_CONTEXT_MESSAGE, [], is_sufficient=False)
    assert score == 1.0


def test_answer_relevance_matching_entities():
    """Verify answer relevance measures overlap with expected entities."""
    answer = "The L1 on-call engineer has 5 minutes to acknowledge in PagerDuty, escalating to L2 Staff SRE Lead."
    expected = "5 minutes for L1, escalating to L2 Staff SRE Lead."
    entities = ["5 minutes", "L1", "L2 Staff SRE Lead", "PagerDuty"]

    score = RAGEvaluator.compute_answer_relevance(answer, expected, entities, is_sufficient=True)
    assert score == 1.0


def test_evaluation_csv_generation():
    """Verify that rag_eval_scores.csv exists and has required schema."""
    csv_path = "evaluation/rag_eval_scores.csv"
    assert os.path.exists(csv_path), "Evaluation CSV was not generated"

    df = pd.read_csv(csv_path)
    assert len(df) == 10
    required_columns = [
        "question_id",
        "category",
        "question",
        "expected_answer",
        "generated_answer",
        "retrieved_pages",
        "precision_at_k",
        "recall_at_k",
        "reciprocal_rank",
        "faithfulness",
        "answer_relevance",
    ]
    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"

    # Verify scores are bounded between 0 and 1
    assert (df["precision_at_k"] >= 0.0).all() and (df["precision_at_k"] <= 1.0).all()
    assert (df["recall_at_k"] >= 0.0).all() and (df["recall_at_k"] <= 1.0).all()
    assert (df["reciprocal_rank"] >= 0.0).all() and (df["reciprocal_rank"] <= 1.0).all()
    assert (df["faithfulness"] >= 0.0).all() and (df["faithfulness"] <= 1.0).all()


"""
TitanCloud SRE Incident Assistant - Evaluation Module
Evaluates retrieval quality (Precision@K, Recall@K, MRR) and generation quality
(Faithfulness, Answer Relevance) against the verified 50-page enterprise runbook dataset.
"""

import csv
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from backend.models import RAGResponse
from backend.rag_pipeline import INSUFFICIENT_CONTEXT_MESSAGE, RAGPipeline
from backend.retrieval import DocumentRetriever

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluates retrieval accuracy and groundedness of the RAG pipeline."""

    def __init__(self, pipeline: Optional[RAGPipeline] = None):
        self.pipeline = pipeline or RAGPipeline()

    @staticmethod
    def compute_retrieval_metrics(
        retrieved_pages: List[int],
        relevant_pages: List[int],
        k: int = 4,
    ) -> Tuple[float, float, float]:
        """
        Computes Precision@K, Recall@K, and Reciprocal Rank (RR).

        Args:
            retrieved_pages: List of 1-indexed page numbers retrieved by FAISS.
            relevant_pages: Ground-truth list of relevant 1-indexed page numbers.
            k: Top-K cutoff.

        Returns:
            (precision_at_k, recall_at_k, reciprocal_rank)
        """
        top_k_retrieved = retrieved_pages[:k]

        # Case 1: Out-of-domain / unanswerable questions (relevant_pages is empty)
        if not relevant_pages:
            if not top_k_retrieved:
                # Correctly filtered out by relevance threshold!
                return 1.0, 1.0, 1.0
            else:
                # Retrieved chunks when nothing was relevant
                return 0.0, 0.0, 0.0

        if not top_k_retrieved:
            return 0.0, 0.0, 0.0

        # Precision@K: relevant chunks in top K / K
        relevant_in_top_k = sum(1 for p in top_k_retrieved if p in relevant_pages)
        precision_at_k = relevant_in_top_k / len(top_k_retrieved)

        # Recall@K: unique relevant pages retrieved / total unique relevant pages
        unique_relevant_retrieved = len(set(top_k_retrieved) & set(relevant_pages))
        recall_at_k = unique_relevant_retrieved / len(set(relevant_pages))

        # Reciprocal Rank (RR): 1 / rank of first relevant page
        reciprocal_rank = 0.0
        for rank, p in enumerate(top_k_retrieved, 1):
            if p in relevant_pages:
                reciprocal_rank = 1.0 / rank
                break

        return round(precision_at_k, 4), round(recall_at_k, 4), round(reciprocal_rank, 4)

    @staticmethod
    def compute_faithfulness(
        generated_answer: str,
        retrieved_contexts: List[str],
        is_sufficient: bool,
    ) -> float:
        """
        Measures groundedness: checks that facts and numbers asserted in generated_answer
        are backed by the retrieved context.
        """
        if not is_sufficient:
            # If system correctly stated insufficient context, it is 100% faithful (no hallucination)
            if INSUFFICIENT_CONTEXT_MESSAGE.lower() in generated_answer.lower():
                return 1.0
            return 0.0

        if not generated_answer or not retrieved_contexts:
            return 0.0

        combined_context = " ".join(retrieved_contexts).lower()
        answer_clean = generated_answer.lower()

        # Extract numerical and uppercase tokens from answer
        tokens_to_verify = re.findall(r"\b\d+(?:\.\d+)?%?\b|\b[A-Z0-9_]{3,}\b", generated_answer)
        if not tokens_to_verify:
            return 1.0

        supported = 0
        for token in tokens_to_verify:
            if token.lower() in combined_context:
                supported += 1

        score = supported / len(tokens_to_verify)
        return round(score, 4)

    @staticmethod
    def compute_answer_relevance(
        generated_answer: str,
        expected_answer: str,
        expected_entities: List[str],
        is_sufficient: bool,
    ) -> float:
        """
        Measures semantic relevance against expected ground truth and key entities.
        """
        if not is_sufficient:
            if INSUFFICIENT_CONTEXT_MESSAGE.lower() in generated_answer.lower():
                return 1.0
            return 0.0

        if not expected_entities:
            return 1.0 if generated_answer.strip() else 0.0

        gen_lower = generated_answer.lower()
        matched_entities = sum(1 for e in expected_entities if e.lower() in gen_lower)
        score = matched_entities / len(expected_entities)
        return round(score, 4)

    def run_evaluation(
        self,
        test_questions_path: str = "evaluation/test_questions.json",
        output_csv_path: str = "evaluation/rag_eval_scores.csv",
        top_k: int = 4,
    ) -> pd.DataFrame:
        """
        Runs the full evaluation benchmark against test_questions.json
        and persists results to rag_eval_scores.csv.
        """
        if not os.path.exists(test_questions_path):
            raise FileNotFoundError(f"Test questions dataset not found at '{test_questions_path}'.")

        with open(test_questions_path, "r", encoding="utf-8") as f:
            test_cases = json.load(f)

        logger.info(f"Running RAG evaluation across {len(test_cases)} benchmark questions (top_k={top_k})...")

        eval_records: List[Dict[str, Any]] = []

        for item in test_cases:
            qid = item["id"]
            question = item["question"]
            expected_answer = item["expected_answer"]
            relevant_pages = item.get("relevant_pages", [])
            expected_entities = item.get("expected_entities", [])

            # Run through actual RAG pipeline
            rag_result: RAGResponse = self.pipeline.answer(question=question, top_k=top_k)

            retrieved_pages = [s.page_number for s in rag_result.sources]
            retrieved_texts = [s.text for s in rag_result.sources]

            prec_at_k, rec_at_k, rr = self.compute_retrieval_metrics(
                retrieved_pages=retrieved_pages,
                relevant_pages=relevant_pages,
                k=top_k,
            )

            faithfulness = self.compute_faithfulness(
                generated_answer=rag_result.answer,
                retrieved_contexts=retrieved_texts,
                is_sufficient=rag_result.is_sufficient,
            )

            answer_rel = self.compute_answer_relevance(
                generated_answer=rag_result.answer,
                expected_answer=expected_answer,
                expected_entities=expected_entities,
                is_sufficient=rag_result.is_sufficient,
            )

            eval_records.append(
                {
                    "question_id": qid,
                    "category": item.get("category", "General"),
                    "question": question,
                    "expected_answer": expected_answer,
                    "generated_answer": rag_result.answer,
                    "retrieved_pages": str(retrieved_pages),
                    "expected_pages": str(relevant_pages),
                    "precision_at_k": prec_at_k,
                    "recall_at_k": rec_at_k,
                    "reciprocal_rank": rr,
                    "faithfulness": faithfulness,
                    "answer_relevance": answer_rel,
                }
            )

        df = pd.DataFrame(eval_records)
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
        logger.info(f"Evaluation completed. Scores saved to '{output_csv_path}'.")

        return df


def print_evaluation_summary(df: pd.DataFrame) -> None:
    """Prints a clean developer-facing summary table of the evaluation scores."""
    border = "=" * 78
    print("\n" + border)
    print("      TITANCLOUD SRE RUNBOOK — RAG PIPELINE EVALUATION SUMMARY")
    print(border)
    print(f" Total Benchmark Questions Evaluated: {len(df)}")
    print(f" Mean Precision@K:                   {df['precision_at_k'].mean():.4f}")
    print(f" Mean Recall@K:                      {df['recall_at_k'].mean():.4f}")
    print(f" Mean Reciprocal Rank (MRR):         {df['reciprocal_rank'].mean():.4f}")
    print(f" Mean Faithfulness / Groundedness:   {df['faithfulness'].mean():.4f}")
    print(f" Mean Answer Relevance:              {df['answer_relevance'].mean():.4f}")
    print(border)
    print(" Breakdown by Category:")
    cat_grp = df.groupby("category")[["precision_at_k", "recall_at_k", "reciprocal_rank", "faithfulness"]].mean()
    print(cat_grp.to_string())
    print(border + "\n")


if __name__ == "__main__":
    evaluator = RAGEvaluator()
    results_df = evaluator.run_evaluation()
    print_evaluation_summary(results_df)

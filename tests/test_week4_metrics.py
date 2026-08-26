import unittest

from src.week4_metrics import (
    abstention_detected,
    answer_relevance_proxy,
    concept_coverage,
    percentile,
    retrieval_metrics,
    unsupported_claim_proxy,
)


class Week4MetricTests(unittest.TestCase):
    def test_concept_coverage_accepts_aliases(self):
        facts = [["chest pain", "chest pressure"], ["call 911", "call emergency services"]]
        self.assertEqual(concept_coverage("Chest pressure means call emergency services.", facts), 1.0)

    def test_retrieval_metrics_binary_relevance(self):
        result = retrieval_metrics([3, 2, 8], [2, 9], 3)
        self.assertAlmostEqual(result["precision_at_k"], 1 / 3)
        self.assertAlmostEqual(result["recall_at_k"], 1 / 2)
        self.assertAlmostEqual(result["reciprocal_rank"], 1 / 2)
        self.assertEqual(result["hit_at_k"], 1.0)

    def test_retrieval_metrics_are_undefined_for_out_of_scope(self):
        result = retrieval_metrics([1, 2, 3], [], 3)
        self.assertTrue(all(value is None for value in result.values()))

    def test_retrieval_metrics_support_file_paths(self):
        result = retrieval_metrics(["a.py", "b.py"], ["b.py"], 2)
        self.assertEqual(result["precision_at_k"], 0.5)
        self.assertEqual(result["reciprocal_rank"], 0.5)

    def test_abstention_detection(self):
        self.assertTrue(abstention_detected("The verified context does not contain this information."))
        self.assertFalse(abstention_detected("The dose is 20 mg."))

    def test_unsupported_claim_proxy(self):
        context = ["Metformin can cause nausea and diarrhoea. Monitor renal function."]
        supported = unsupported_claim_proxy("Metformin can cause nausea and diarrhoea.", context)
        unsupported = unsupported_claim_proxy("Metformin permanently cures diabetes in all patients.", context)
        self.assertLess(supported["unsupported_sentence_rate"], unsupported["unsupported_sentence_rate"])

    def test_relevance_proxy_empty_answer(self):
        self.assertEqual(answer_relevance_proxy("", "What treats asthma?", "Albuterol treats asthma."), 0.0)

    def test_percentile(self):
        self.assertEqual(percentile([10, 20, 30], 50), 20)
        self.assertIsNone(percentile([], 95))


if __name__ == "__main__":
    unittest.main()

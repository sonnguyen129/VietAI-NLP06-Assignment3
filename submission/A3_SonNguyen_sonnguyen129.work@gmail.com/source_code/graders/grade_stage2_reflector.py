import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock

# Ensure students_code directory is in the import path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

try:
    from graders.messages import print_stage_message
    from src.executor import EvalResult, QuestionResult
    from src.strategy import Strategy, CoTFormat, RetrievalConfig, StrategyMetadata, Reflection
    from src.self_reflector import _build_reflect_message, reflect_self
    from src.model import QwenInference
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please make sure you are running the grader from the students_code directory.")
    sys.exit(1)


class TestStage2Reflector(unittest.TestCase):
    def setUp(self):
        # Create a mock EvalResult with 6 failures
        self.failures = [
            QuestionResult(
                question_id=str(i),
                passage=f"Passage {i}",
                question=f"Question {i}",
                gold_answer=f"gold_{i}",
                predicted_answer=f"pred_{i}",
                is_correct=False,
                raw_output=f"raw_{i}",
                question_type="addition",
                input_tokens=10,
                output_tokens=5
            )
            for i in range(1, 7)
        ]
        self.eval_result = EvalResult(
            strategy_id="strat-1",
            split="dev",
            num_examples=10,
            num_correct=4,
            accuracy=0.4,
            accuracy_by_type={"addition": 0.4},
            count_by_type={"addition": 10},
            per_question=self.failures
        )

    def test_progressive_decay(self):
        print("\n--> Testing progressive context decay in reflection message...")
        
        # Test iteration 0 (progressive=True) -> Should include 5 failures
        strategy_it0 = Strategy(
            id="strat-1",
            prompt_template="Test",
            cot_format=CoTFormat.NONE,
            few_shot_examples=[],
            retrieval_config=RetrievalConfig(enabled=False),
            metadata=StrategyMetadata(iteration=0)
        )
        msg_it0 = _build_reflect_message(strategy_it0, self.eval_result, progressive=True)
        self.assertIn("Lỗi 5:", msg_it0)
        self.assertNotIn("Lỗi 6:", msg_it0)

        # Test iteration 2 (progressive=True) -> Should include 3 failures
        strategy_it2 = Strategy(
            id="strat-1",
            prompt_template="Test",
            cot_format=CoTFormat.NONE,
            few_shot_examples=[],
            retrieval_config=RetrievalConfig(enabled=False),
            metadata=StrategyMetadata(iteration=2)
        )
        msg_it2 = _build_reflect_message(strategy_it2, self.eval_result, progressive=True)
        self.assertIn("Lỗi 3:", msg_it2)
        self.assertNotIn("Lỗi 4:", msg_it2)

        # Test iteration 3 (progressive=True) -> Should include 1 failure
        strategy_it3 = Strategy(
            id="strat-1",
            prompt_template="Test",
            cot_format=CoTFormat.NONE,
            few_shot_examples=[],
            retrieval_config=RetrievalConfig(enabled=False),
            metadata=StrategyMetadata(iteration=3)
        )
        msg_it3 = _build_reflect_message(strategy_it3, self.eval_result, progressive=True)
        self.assertIn("Lỗi 1:", msg_it3)
        self.assertNotIn("Lỗi 2:", msg_it3)

        # Test iteration 3 (progressive=False) -> Should default back to 5 failures
        msg_it3_no_prog = _build_reflect_message(strategy_it3, self.eval_result, progressive=False)
        self.assertIn("Lỗi 5:", msg_it3_no_prog)
        self.assertNotIn("Lỗi 6:", msg_it3_no_prog)
        print("    [PASS] Progressive context decay verified.")

    def test_reflect_self_success(self):
        print("\n--> Testing reflect_self success path...")
        strategy = Strategy(
            id="strat-1",
            prompt_template="Test",
            cot_format=CoTFormat.NONE,
            few_shot_examples=[],
            retrieval_config=RetrievalConfig(enabled=False),
            metadata=StrategyMetadata(iteration=0)
        )

        model = MagicMock(spec=QwenInference)
        model.format_prompt.side_effect = lambda system_message, user_message, enable_thinking: user_message
        model.count_tokens.return_value = 50

        # Mock generate_text calls:
        # Pass 1: Raw text reflection
        # Pass 2: JSON Coercion
        model.generate_text.side_effect = [
            "<think>Thinking...</think> Raw analysis text proposal.",
            '{"accuracy_by_type": {"addition": 0.4}, "failure_patterns": ["wrong_formula"], "hypothesis": "Better instruction needed", "summary": "Low score summary"}'
        ]

        reflection, tokens = reflect_self(strategy, self.eval_result, model, max_retries=3)

        self.assertIsInstance(reflection, Reflection)
        self.assertEqual(reflection.strategy_id, strategy.id)
        self.assertEqual(reflection.hypothesis, "Better instruction needed")
        self.assertEqual(reflection.summary, "Low score summary")
        self.assertEqual(reflection.accuracy_by_type.get("addition"), 0.4)
        self.assertGreater(tokens, 0)
        print("    [PASS] reflect_self success path verified.")

    def test_reflect_self_fallback(self):
        print("\n--> Testing reflect_self fallback path on parse failure...")
        strategy = Strategy(
            id="strat-1",
            prompt_template="Test",
            cot_format=CoTFormat.NONE,
            few_shot_examples=[],
            retrieval_config=RetrievalConfig(enabled=False),
            metadata=StrategyMetadata(iteration=0)
        )

        model = MagicMock(spec=QwenInference)
        model.format_prompt.side_effect = lambda system_message, user_message, enable_thinking: user_message
        model.count_tokens.return_value = 50

        # Always return garbage JSON, causing parser validation to fail
        model.generate_text.return_value = "Garbage output, not JSON"

        # Should fall back to direct reflection object without crashing
        reflection, tokens = reflect_self(strategy, self.eval_result, model, max_retries=2)

        self.assertIsInstance(reflection, Reflection)
        self.assertEqual(reflection.strategy_id, strategy.id)
        self.assertTrue(reflection.hypothesis.startswith("Chiến lược hiện tại yếu nhất"))
        self.assertIn("Fallback", reflection.summary)
        self.assertEqual(tokens, 0)
        print("    [PASS] reflect_self fallback path verified.")


if __name__ == "__main__":
    print("="*60)
    print("RUNNING GRADER: STAGE 2 (SELF REFLECTOR)")
    print("="*60)
    
    suite = unittest.TestSuite()
    suite.addTest(TestStage2Reflector("test_progressive_decay"))
    suite.addTest(TestStage2Reflector("test_reflect_self_success"))
    suite.addTest(TestStage2Reflector("test_reflect_self_fallback"))
    
    runner = unittest.TextTestRunner(verbosity=1)
    res = runner.run(suite)
    
    if res.wasSuccessful():
        print("\n" + "="*60)
        print("SUCCESS! STAGE 2 COMPLETELY VERIFIED.")
        print("="*60 + "\n")
        print_stage_message("stage2")
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("STAGE 2 GRADING FAILED. Please review the errors above.")
        print("="*60 + "\n")
        sys.exit(1)

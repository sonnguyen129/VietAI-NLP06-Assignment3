import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock

# Ensure students_code directory is in the import path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

try:
    from graders.messages import print_stage_message
    from src.executor import TokenBudget, EvalResult, QuestionResult, classify_question_type, evaluate
    from src.strategy import Strategy, CoTFormat, RetrievalConfig, StrategyMetadata
    from src.model import QwenInference, GenerationResult
    from datasets import Dataset
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please make sure you are running the grader from the students_code directory.")
    sys.exit(1)


class TestStage1Executor(unittest.TestCase):
    def test_token_budget(self):
        print("\n--> Testing TokenBudget...")
        budget = TokenBudget()
        self.assertEqual(budget.qwen_input, 0)
        self.assertEqual(budget.qwen_output, 0)
        self.assertEqual(budget.meta_total, 0)
        self.assertEqual(budget.qwen_total, 0)

        # Create a mock EvalResult
        result = EvalResult(
            strategy_id="test-strategy",
            split="train",
            num_examples=2,
            num_correct=1,
            accuracy=0.5,
            total_input_tokens=150,
            total_output_tokens=50
        )
        budget.add_eval(result)
        self.assertEqual(budget.qwen_input, 150)
        self.assertEqual(budget.qwen_output, 50)
        self.assertEqual(budget.qwen_total, 200)

        budget.add_meta(300)
        self.assertEqual(budget.meta_total, 300)
        self.assertIn("200", budget.summary())
        self.assertIn("300", budget.summary())
        print("    [PASS] TokenBudget verification complete.")

    def test_classify_question_type(self):
        print("\n--> Testing classify_question_type...")
        self.assertEqual(classify_question_type("add(10, 5)"), "addition")
        self.assertEqual(classify_question_type("subtract(10, 5)"), "subtraction")
        self.assertEqual(classify_question_type("multiply(10, 5)"), "multiplication")
        self.assertEqual(classify_question_type("divide(10, 5)"), "division")
        self.assertEqual(classify_question_type("table_average(Revenue, none)"), "table_op")
        self.assertEqual(classify_question_type("other_op(10, 5)"), "other")
        self.assertEqual(classify_question_type(""), "other")
        print("    [PASS] classify_question_type verification complete.")

    def test_evaluate_loop(self):
        print("\n--> Testing evaluate loop...")
        # Create a mock strategy
        strategy = Strategy(
            id="test-strat",
            prompt_template="Solve: {passage} | {question}",
            cot_format=CoTFormat.NONE,
            few_shot_examples=[],
            retrieval_config=RetrievalConfig(enabled=False),
            metadata=StrategyMetadata(iteration=0)
        )

        # Mock dataset with 2 questions
        mock_data = [
            {
                "context": "doanh thu năm 2020 là 100 tỷ.",
                "question": "Tính doanh thu?",
                "answer": "add(100, 0)",
                "exe_ans": "100.0",
                "table": []
            },
            {
                "context": "chi phí năm 2020 là 50 tỷ.",
                "question": "Tính chi phí?",
                "answer": "subtract(100, 50)",
                "exe_ans": "50.0",
                "table": []
            }
        ]
        dataset = Dataset.from_list(mock_data)

        # Mock model
        model = MagicMock(spec=QwenInference)
        model.format_prompt.side_effect = lambda system_message, user_message, enable_thinking: user_message
        
        # When model evaluates, return specific answers
        mock_gen_results = [
            GenerationResult(raw_output="PROGRAM: add(100, 0)", predicted_answer="add(100, 0)", input_tokens=10, output_tokens=5),
            # Let the second prediction be wrong or fail program check, but fallback to string matching
            GenerationResult(raw_output="PROGRAM: subtract(100, 50)", predicted_answer="subtract(100, 50)", input_tokens=12, output_tokens=6)
        ]
        model.generate_batch.return_value = mock_gen_results

        # Run evaluate
        result = evaluate(strategy, "train", dataset, model)

        self.assertEqual(result.strategy_id, strategy.id)
        self.assertEqual(result.num_examples, 2)
        self.assertEqual(result.num_correct, 2)
        self.assertEqual(result.accuracy, 1.0)
        self.assertEqual(result.total_input_tokens, 22)
        self.assertEqual(result.total_output_tokens, 11)
        self.assertEqual(result.count_by_type.get("addition"), 1)
        self.assertEqual(result.count_by_type.get("subtraction"), 1)
        self.assertEqual(result.accuracy_by_type.get("addition"), 1.0)
        self.assertEqual(result.accuracy_by_type.get("subtraction"), 1.0)
        print("    [PASS] evaluate loop verification complete.")


if __name__ == "__main__":
    print("="*60)
    print("RUNNING GRADER: STAGE 1 (EXECUTOR & TOKEN BUDGET)")
    print("="*60)
    
    suite = unittest.TestSuite()
    suite.addTest(TestStage1Executor("test_token_budget"))
    suite.addTest(TestStage1Executor("test_classify_question_type"))
    suite.addTest(TestStage1Executor("test_evaluate_loop"))
    
    runner = unittest.TextTestRunner(verbosity=1)
    res = runner.run(suite)
    
    if res.wasSuccessful():
        print("\n" + "="*60)
        print("SUCCESS! STAGE 1 COMPLETELY VERIFIED.")
        print("="*60 + "\n")
        print_stage_message("stage1")
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("STAGE 1 GRADING FAILED. Please review the errors above.")
        print("="*60 + "\n")
        sys.exit(1)

import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock

# Ensure students_code directory is in the import path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

try:
    from graders.messages import print_stage_message
    from src.executor import EvalResult
    from src.strategy import Strategy, CoTFormat, RetrievalConfig, StrategyMetadata, Reflection, StrategyHistory, FewShotExample
    from src.self_proposer import _is_valid_dsl_program, propose_self
    from src.model import QwenInference
    from datasets import Dataset
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please make sure you are running the grader from the students_code directory.")
    sys.exit(1)


class TestStage3Proposer(unittest.TestCase):
    def test_dsl_validator(self):
        print("\n--> Testing _is_valid_dsl_program...")
        self.assertTrue(_is_valid_dsl_program("subtract(108.50, 100), divide(#0, 100)"))
        self.assertTrue(_is_valid_dsl_program("add(-167.4, -53.3)"))
        self.assertTrue(_is_valid_dsl_program("table_max(Revenue, none)"))
        
        # Invalid math symbols/syntax
        self.assertFalse(_is_valid_dsl_program("100 + 50"))
        self.assertFalse(_is_valid_dsl_program("x = 5"))
        self.assertFalse(_is_valid_dsl_program("unknown_operator(10, 5)"))
        self.assertFalse(_is_valid_dsl_program(""))
        print("    [PASS] _is_valid_dsl_program verified.")

    def test_dynamic_few_shot_selection(self):
        print("\n--> Testing dynamic few-shot selection...")
        
        # Mock model that returns:
        # Pass 1: raw proposal
        # Pass 2: JSON ProposerSchema
        # And when generating few-shot CoT reasoning, returns matching program.
        model = MagicMock(spec=QwenInference)
        model.format_prompt.side_effect = lambda system_message, user_message, enable_thinking: user_message
        model.count_tokens.return_value = 50

        # Define model outputs
        def mock_generate_text(prompt, max_new_tokens=256, temperature=0.7, guided_json=None):
            if "ProposerSchema" in str(guided_json) or "ProposerSchema" in prompt:
                return '{"instruction_phrasing": "mock instruction", "hypothesis": "mock", "cot_format": "none", "few_shot_examples": [], "reasoning": "mock"}'
            if "Chương trình đúng:" in prompt:
                # Extracts the program from prompt and returns verified CoT
                import re
                prog_match = re.search(r"Chương trình đúng:\s*(.+)", prompt)
                program = prog_match.group(1).strip() if prog_match else "subtract(10, 5)"
                return (
                    f"<think>Lập luận mẫu.</think>\n"
                    f'{{"Reasoning": "Lập luận ví dụ.", '
                    f'"Program syntax": "{program}", '
                    f'"Numerical result": 5.0}}'
                )
            return "<think>Thinking...</think> Strategy proposal text."
        
        model.generate_text.side_effect = mock_generate_text

        # Strategy history setup
        history = StrategyHistory("mock_history.jsonl")
        
        # Iteration 0 seed
        seed_strategy = Strategy(
            id="strat-0",
            prompt_template="Solve: {passage} {question}",
            cot_format=CoTFormat.NONE,
            few_shot_examples=[
                FewShotExample(passage="P_seed", question="Q_seed", answer="add(1, 1)")
            ],
            retrieval_config=RetrievalConfig(enabled=False),
            metadata=StrategyMetadata(iteration=0, dev_accuracy=0.5)
        )
        history.strategies.append(seed_strategy)
        
        # Reflection indicating division is the weakest category
        reflection = Reflection(
            strategy_id="strat-0",
            accuracy_by_type={"addition": 0.8, "division": 0.2},
            top_failures=[],
            hypothesis="Weak on division.",
            summary="Reflection summary."
        )
        history.reflections.append(reflection)

        # Train dataset containing a division question and other types
        train_data = [
            {
                "id": "1",
                "context": "Context division 1",
                "question": "Question division 1",
                "answer": "divide(100, 50)",
                "exe_ans": "2.0"
            },
            {
                "id": "2",
                "context": "Context division 2",
                "question": "Question division 2",
                "answer": "divide(50, 10)",
                "exe_ans": "5.0"
            },
            {
                "id": "3",
                "context": "Context addition 1",
                "question": "Question addition 1",
                "answer": "add(10, 20)",
                "exe_ans": "30.0"
            }
        ]
        train_dataset = Dataset.from_list(train_data)

        # Run proposer with train_dataset
        new_strategy, tokens = propose_self(
            history=history,
            model=model,
            max_retries=1,
            train_dataset=train_dataset
        )

        # It should merge the 2 division examples with the parent's addition example, yielding 3 few-shots
        self.assertEqual(len(new_strategy.few_shot_examples), 3)
        
        # Parent example preserved
        parent_ex = new_strategy.few_shot_examples[0]
        self.assertEqual(parent_ex.passage, "P_seed")
        self.assertEqual(parent_ex.question, "Q_seed")
        self.assertEqual(parent_ex.answer, "add(1, 1)")

        ex1 = new_strategy.few_shot_examples[1]
        self.assertEqual(ex1.passage, "Context division 1")
        self.assertEqual(ex1.question, "Question division 1")
        self.assertEqual(ex1.answer, "divide(100, 50)")
        self.assertIn("</think>", ex1.reasoning)
        self.assertIn("divide(100, 50)", ex1.reasoning)

        ex2 = new_strategy.few_shot_examples[2]
        self.assertEqual(ex2.passage, "Context division 2")
        self.assertEqual(ex2.question, "Question division 2")
        self.assertEqual(ex2.answer, "divide(50, 10)")
        self.assertIn("</think>", ex2.reasoning)
        self.assertIn("divide(50, 10)", ex2.reasoning)
        print("    [PASS] Dynamic few-shot selection verified.")


if __name__ == "__main__":
    print("="*60)
    print("RUNNING GRADER: STAGE 3 (SELF PROPOSER)")
    print("="*60)
    
    suite = unittest.TestSuite()
    suite.addTest(TestStage3Proposer("test_dsl_validator"))
    suite.addTest(TestStage3Proposer("test_dynamic_few_shot_selection"))
    
    runner = unittest.TextTestRunner(verbosity=1)
    res = runner.run(suite)
    
    if res.wasSuccessful():
        print("\n" + "="*60)
        print("SUCCESS! STAGE 3 COMPLETELY VERIFIED.")
        print("="*60 + "\n")
        print_stage_message("stage3")
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("STAGE 3 GRADING FAILED. Please review the errors above.")
        print("="*60 + "\n")
        sys.exit(1)

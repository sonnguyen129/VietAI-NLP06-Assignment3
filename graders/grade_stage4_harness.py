import sys
from pathlib import Path
import unittest
import tempfile
import shutil
from unittest.mock import MagicMock, patch

# Ensure students_code directory is in the import path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

try:
    from graders.messages import print_stage_message
    from src.executor import EvalResult, QuestionResult
    from src.strategy import Strategy, CoTFormat, RetrievalConfig, StrategyMetadata, Reflection, StrategyHistory
    from src.harness import select_parent_strategy, run_evoagent
    from src.model import QwenInference
    from datasets import Dataset
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please make sure you are running the grader from the students_code directory.")
    sys.exit(1)


def _read_evolution_accuracy():
    proof_path = Path("evolution_proof.json")
    if not proof_path.exists():
        return None, None
    try:
        import json
        with open(proof_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("best_dev_accuracy"), data.get("baseline_accuracy")
    except Exception:
        return None, None


class TestStage4Harness(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.output_dir = Path(self.tmpdir)

        # Setup mock strategy history
        self.history = StrategyHistory(self.output_dir / "history.jsonl")

        self.s0 = Strategy(
            id="strat-0",
            prompt_template="Solve 0",
            cot_format=CoTFormat.NONE,
            few_shot_examples=[],
            retrieval_config=RetrievalConfig(enabled=False),
            metadata=StrategyMetadata(iteration=0, dev_accuracy=0.5)
        )
        self.s1 = Strategy(
            id="strat-1",
            prompt_template="Solve 1",
            cot_format=CoTFormat.NONE,
            few_shot_examples=[],
            retrieval_config=RetrievalConfig(enabled=False),
            metadata=StrategyMetadata(iteration=1, dev_accuracy=0.8)
        )
        self.s2 = Strategy(
            id="strat-2",
            prompt_template="Solve 2",
            cot_format=CoTFormat.NONE,
            few_shot_examples=[],
            retrieval_config=RetrievalConfig(enabled=False),
            metadata=StrategyMetadata(iteration=2, dev_accuracy=0.6)
        )

        self.history.strategies = [self.s0, self.s1, self.s2]
        self.history.reflections = [None, None, None]

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_select_parent_strategy(self):
        print("\n--> Testing select_parent_strategy (AFO) policies...")
        
        # 1. Mode 'none' -> Returns latest (s2)
        parent_none = select_parent_strategy(self.history, afo_mode="none")
        self.assertEqual(parent_none.id, self.s2.id)

        # 2. Mode 'best' -> Returns best (s1)
        parent_best = select_parent_strategy(self.history, afo_mode="best")
        self.assertEqual(parent_best.id, self.s1.id)

        # 3. Mode 'original' -> Returns original (s0)
        parent_original = select_parent_strategy(self.history, afo_mode="original")
        self.assertEqual(parent_original.id, self.s0.id)

        # 4. Mode 'probabilistic' -> check that it works under weight bounds
        # If best prob is 1.0, it should return best (s1)
        parent_prob_best = select_parent_strategy(
            self.history,
            afo_mode="probabilistic",
            afo_prob_best=1.0,
            afo_prob_original=0.0,
            afo_prob_latest=0.0
        )
        self.assertEqual(parent_prob_best.id, self.s1.id)
        
        # If original prob is 1.0, it should return original (s0)
        parent_prob_orig = select_parent_strategy(
            self.history,
            afo_mode="probabilistic",
            afo_prob_best=0.0,
            afo_prob_original=1.0,
            afo_prob_latest=0.0
        )
        self.assertEqual(parent_prob_orig.id, self.s0.id)

        # If latest prob is 1.0, it should return latest (s2)
        parent_prob_late = select_parent_strategy(
            self.history,
            afo_mode="probabilistic",
            afo_prob_best=0.0,
            afo_prob_original=0.0,
            afo_prob_latest=1.0
        )
        self.assertEqual(parent_prob_late.id, self.s2.id)
        print("    [PASS] select_parent_strategy policies verified.")

    @patch("src.harness.propose_self")
    @patch("src.harness.reflect_self")
    @patch("src.harness.evaluate")
    def test_run_evoagent_loop(self, mock_evaluate, mock_reflect_self, mock_propose_self):
        print("\n--> Testing run_evoagent loop execution...")

        # Setup mock dataset
        train_dataset = Dataset.from_list([{"context": "ctx", "question": "q", "answer": "ans"}])
        dev_dataset = Dataset.from_list([{"context": "ctx", "question": "q", "answer": "ans"}])

        # Mock model
        model = MagicMock(spec=QwenInference)
        model.max_new_tokens = 256

        # Proposer returns a mutated strategy
        def dummy_propose(history, model, max_retries=5, parent_strategy_id=None, **kwargs):
            return Strategy(
                id="mutated-strat",
                prompt_template="Solve: {passage} {question}",
                cot_format=CoTFormat.NONE,
                few_shot_examples=[],
                retrieval_config=RetrievalConfig(enabled=False),
                metadata=StrategyMetadata(iteration=1, parent_id=parent_strategy_id)
            ), 50
        mock_propose_self.side_effect = dummy_propose

        # Evaluator returns success result
        mock_evaluate.return_value = EvalResult(
            strategy_id="strat",
            split="train",
            num_examples=1,
            num_correct=1,
            accuracy=1.0,
            total_input_tokens=10,
            total_output_tokens=5
        )

        # Reflector returns mock reflection
        mock_reflect_self.return_value = (
            Reflection(
                strategy_id="strat",
                accuracy_by_type={"addition": 1.0},
                top_failures=[],
                hypothesis="Mock hyp",
                summary="Mock sum"
            ),
            20
        )

        # Run loop for T=2
        history = run_evoagent(
            T=2,
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            model=model,
            output_dir=self.output_dir,
            train_size=1,
            afo_mode="best",
            early_stop_accuracy=2.0
        )

        # Verify loop executed and files created
        self.assertEqual(len(history.strategies), 2)
        self.assertTrue((self.output_dir / "history.jsonl").exists())
        self.assertTrue((self.output_dir / "iter_000_strategy.json").exists())
        self.assertTrue((self.output_dir / "iter_000_eval_dev.json").exists())
        self.assertTrue((self.output_dir / "iter_001_strategy.json").exists())
        self.assertTrue((self.output_dir / "iter_000_reflection.json").exists())
    @patch("src.harness.evaluate")
    def test_run_smoke_test(self, mock_evaluate):
        print("\n--> Testing run_smoke_test logic...")
        from src.harness import run_smoke_test
        
        train_dataset = Dataset.from_list([{"context": "ctx", "question": "q", "answer": "ans"}])
        model = MagicMock(spec=QwenInference)
        model.max_new_tokens = 256
        
        # Test Case 1: Success (non-null predictions and tokens under limit)
        strategy_none = Strategy(
            id="strat",
            prompt_template="Solve: {passage} {question}",
            cot_format=CoTFormat.NONE,
            few_shot_examples=[],
            retrieval_config=RetrievalConfig(enabled=False),
            metadata=StrategyMetadata(iteration=0)
        )
        mock_evaluate.return_value = EvalResult(
            strategy_id="strat",
            split="smoke_test",
            num_examples=1,
            num_correct=1,
            accuracy=1.0,
            total_output_tokens=10,
            per_question=[QuestionResult(
                question_id="1", passage="ctx", question="q", gold_answer="ans",
                predicted_answer="add(1, 1)", is_correct=True, raw_output="add(1,1)",
                output_tokens=10
            )]
        )
        self.assertTrue(run_smoke_test(strategy_none, train_dataset, model))
        
        # Test Case 2: Failure (zero predictions extracted)
        mock_evaluate.return_value = EvalResult(
            strategy_id="strat",
            split="smoke_test",
            num_examples=1,
            num_correct=0,
            accuracy=0.0,
            total_output_tokens=0,
            per_question=[QuestionResult(
                question_id="1", passage="ctx", question="q", gold_answer="ans",
                predicted_answer=None, is_correct=False, raw_output="garbage",
                output_tokens=0
            )]
        )
        self.assertFalse(run_smoke_test(strategy_none, train_dataset, model))
        
        # Test Case 3: Failure (output token limit threshold triggered)
        # For CoTFormat.NONE, limit set to 256. 90% is 230. 240 average is over 90%.
        mock_evaluate.return_value = EvalResult(
            strategy_id="strat",
            split="smoke_test",
            num_examples=1,
            num_correct=1,
            accuracy=1.0,
            total_output_tokens=240,
            per_question=[QuestionResult(
                question_id="1", passage="ctx", question="q", gold_answer="ans",
                predicted_answer="add(1, 1)", is_correct=True, raw_output="add(1,1)",
                output_tokens=240
            )]
        )
        self.assertFalse(run_smoke_test(strategy_none, train_dataset, model))
        print("    [PASS] run_smoke_test logic verified.")

    def test_evolution_proof(self):
        print("\n--> Verifying evolution_proof.json...")
        proof_path = Path("evolution_proof.json")
        self.assertTrue(
            proof_path.exists(),
            "evolution_proof.json not found! You must run the evolution loop on Modal and run get_proof to download it."
        )
        
        import json
        try:
            with open(proof_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self.fail(f"Failed to parse evolution_proof.json: {e}")
            
        self.assertEqual(data.get("status"), "success", "Evolution run was not successful.")
        self.assertIn("best_dev_accuracy", data, "Evolution proof is missing 'best_dev_accuracy' field.")
        self.assertIn("history", data, "Evolution proof is missing 'history' field.")
        
        best_acc = data["best_dev_accuracy"]
        baseline = data.get("baseline_accuracy", 0.42)
        
        self.assertGreaterEqual(
            best_acc,
            baseline,
            f"Your best dev accuracy {best_acc:.3f} is lower than the baseline {baseline:.3f}!"
        )
        print(f"    [PASS] evolution_proof.json verified successfully. Best Dev Accuracy: {best_acc:.3f} (Baseline: {baseline:.3f})")


if __name__ == "__main__":
    print("="*60)
    print("RUNNING GRADER: STAGE 4 (EVOLUTION LOOP & AFO ORCHESTRATION)")
    print("="*60)
    
    suite = unittest.TestSuite()
    suite.addTest(TestStage4Harness("test_select_parent_strategy"))
    suite.addTest(TestStage4Harness("test_run_smoke_test"))
    suite.addTest(TestStage4Harness("test_run_evoagent_loop"))
    suite.addTest(TestStage4Harness("test_evolution_proof"))
    
    runner = unittest.TextTestRunner(verbosity=1)
    res = runner.run(suite)
    
    if res.wasSuccessful():
        print("\n" + "="*60)
        print("SUCCESS! STAGE 4 COMPLETELY VERIFIED.")
        print("="*60 + "\n")
        accuracy, baseline = _read_evolution_accuracy()
        print_stage_message("stage4", accuracy=accuracy, baseline=baseline)
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("STAGE 4 GRADING FAILED. Please review the errors above.")
        print("="*60 + "\n")
        sys.exit(1)

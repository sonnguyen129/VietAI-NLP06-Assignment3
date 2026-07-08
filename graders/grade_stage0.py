import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

# Ensure students_code directory is in the import path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

try:
    from graders.messages import print_stage_message
    from src.sandbox import run_sandbox_prediction
    from src.model import QwenInference
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please make sure you are running the grader from the students_code directory.")
    sys.exit(1)


# Known valid DSL operations
_VALID_OPS = {"add", "subtract", "multiply", "divide", "table_max", "table_min",
              "table_sum", "table_average", "exp", "greater", "abs"}


def _looks_like_dsl_program(program: str) -> bool:
    """Return True if the string resembles a valid DSL program."""
    if not program or not isinstance(program, str):
        return False
    p = program.strip().lower()
    return any(p.startswith(op + "(") for op in _VALID_OPS)


def _read_baseline_accuracy():
    proof_path = Path("sandbox_proof.json")
    if not proof_path.exists():
        return None
    try:
        import json
        with open(proof_path, "r", encoding="utf-8") as f:
            return json.load(f).get("baseline_accuracy")
    except Exception:
        return None


class TestStage0Sandbox(unittest.TestCase):
    @patch("src.sandbox.load_data_splits")
    @patch("src.sandbox.QwenInference")
    def test_sandbox_prediction(self, mock_qwen_cls, mock_load_splits):
        print("\n--> Testing run_sandbox_prediction...")

        # 1. Setup mock training dataset split
        mock_train_example = {
            "context": "Context example 1",
            "question": "Question example 1",
            "answer": "add(10, 5)",
            "exe_ans": "15.0"
        }
        mock_load_splits.return_value = ([mock_train_example], [])

        # 2. Setup mock model instance
        mock_model = MagicMock(spec=QwenInference)
        mock_qwen_cls.return_value = mock_model

        mock_model.format_prompt.side_effect = lambda system_message, user_message, enable_thinking: user_message
        mock_model.generate_text.return_value = "<think>Sample thought</think> PROGRAM: add(10, 5)"

        # 3. Call sandbox function
        pred, gold = run_sandbox_prediction(model_name="mock-model")

        # 4. Assertions
        mock_load_splits.assert_called_once()
        mock_qwen_cls.assert_called_once()
        self.assertEqual(mock_qwen_cls.call_args.kwargs.get("model_name_or_path"), "mock-model")
        mock_model.load.assert_called_once()
        mock_model.format_prompt.assert_called_once()
        mock_model.generate_text.assert_called_once()

        self.assertEqual(pred, "<think>Sample thought</think> PROGRAM: add(10, 5)")
        self.assertEqual(gold, "add(10, 5)")
        print("    [PASS] run_sandbox_prediction verification complete.")

    def test_sandbox_proof(self):
        print("\n--> Checking sandbox_proof.json...")
        proof_path = Path("sandbox_proof.json")
        self.assertTrue(
            proof_path.exists(),
            "sandbox_proof.json not found! You must run 'modal run run_modal.py::sandbox' successfully first."
        )

        import json
        try:
            with open(proof_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self.fail(f"Failed to parse sandbox_proof.json: {e}")

        # ── Basic status check ──────────────────────────────────────────────
        self.assertEqual(data.get("status"), "success", "Sandbox run on Modal was not successful.")

        # ── samples must exist and be non-empty ─────────────────────────────
        samples = data.get("samples", [])
        self.assertIsInstance(samples, list, "'samples' field must be a list.")
        self.assertGreater(len(samples), 0, "'samples' list is empty — no predictions recorded.")

        # ── Resolve gold_program and predicted_answer from samples[0] ────────
        first = samples[0]
        gold = first.get("gold_program")
        pred = first.get("predicted_answer")

        self.assertIsNotNone(
            gold,
            "samples[0] is missing 'gold_program'. Check that run_sandbox_accuracy_check() saves this field."
        )
        self.assertIsNotNone(
            pred,
            "samples[0] is missing 'predicted_answer'. Check that run_sandbox_accuracy_check() saves this field."
        )

        # ── Verify it's a real run, not the mock ────────────────────────────
        MOCK_PRED = "<think>Sample thought</think> PROGRAM: add(10, 5)"
        self.assertNotEqual(pred, MOCK_PRED, "Cannot use mock values in sandbox_proof.json!")

        # ── Structural validity: gold must be a proper DSL program ───────────
        self.assertTrue(
            _looks_like_dsl_program(gold),
            f"gold_program '{gold}' does not look like a valid DSL program "
            f"(should start with a known operation such as add(...), subtract(...), etc.)."
        )

        # ── pred must be non-trivially long ─────────────────────────────────
        self.assertGreater(
            len(str(pred).strip()),
            4,
            "predicted_answer in samples[0] looks empty or trivially short."
        )

        # ── Sanity: baseline_accuracy must be a float in [0, 1] ─────────────
        acc = data.get("baseline_accuracy")
        if acc is not None:
            self.assertIsInstance(acc, (int, float), "baseline_accuracy must be a number.")
            self.assertGreaterEqual(float(acc), 0.0, "baseline_accuracy must be >= 0.")
            self.assertLessEqual(float(acc), 1.0, "baseline_accuracy must be <= 1.")

        print(f"    [PASS] sandbox_proof.json verification complete.")
        print(f"           gold_program       = {str(gold)[:80]}")
        print(f"           predicted_answer[:80] = {str(pred)[:80]}")
        if acc is not None:
            print(f"           baseline_accuracy  = {acc:.4f}")



if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING GRADER: STAGE 0 (THE SANDBOX)")
    print("=" * 60)

    suite = unittest.TestSuite()
    suite.addTest(TestStage0Sandbox("test_sandbox_prediction"))
    suite.addTest(TestStage0Sandbox("test_sandbox_proof"))

    runner = unittest.TextTestRunner(verbosity=1)
    res = runner.run(suite)

    if res.wasSuccessful():
        print("\n" + "=" * 60)
        print("SUCCESS! STAGE 0 COMPLETELY VERIFIED.")
        print("=" * 60 + "\n")
        print_stage_message("stage0", accuracy=_read_baseline_accuracy())
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("STAGE 0 GRADING FAILED. Please review the errors above.")
        print("=" * 60 + "\n")
        sys.exit(1)

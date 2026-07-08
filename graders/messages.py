"""Student-facing success messages for the staged graders."""

import os


GREEN = "\033[92m"
RESET = "\033[0m"


def _success(text):
    if os.environ.get("NO_COLOR"):
        return text
    return f"{GREEN}{text}{RESET}"


def _format_accuracy(value):
    if value is None:
        return None
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(value)


def print_stage_message(stage, accuracy=None, baseline=None):
    """Print a short encouragement and next-step guide after a grader passes."""
    accuracy_text = _format_accuracy(accuracy)
    baseline_text = _format_accuracy(baseline)

    messages = {
        "stage0": (
            "🎉 [STAGE 0 CLEARED] Yayyy, you did it! The sandbox is alive! \n\n"
            "This is the first step, you're getting your feet wet. Let's see how smart your AI is..."
            + (f" oh... {accuracy_text}. Oh... " if accuracy_text is not None else " oh... wait... ")
            + "Yeah... thatttt is EXACTLY why we need evolution :) \n\n"
            "Right now, your AI is essentially a toddler mashing a keyboard. In Stage 1, we are going to give it a speedometer. \n\n"
            "Go build the Execution Engine so we can track tokens, measure progress, and actually fix this mess!"
        ),
        "stage1": (
            "🔥 [STAGE 1 CLEARED] The Execution Engine is online! \n\n"
            "Your agent can now count its tokens and score its own tests. We have officially moved from 'vibes' to hard data. \n\n"
            "But data is useless without insight, right? 😉.\n\nYour next mission is Stage 2. \n\n"
            "Go build the Critic so the AI can read its own failures and categorize its mistakes."
        ),
        "stage2": (
            "🧠 [STAGE 2 CLEARED] The Critic is awake! \n\n"
            "Mistakes are no longer just sad rows in a JSON file—they are actionable evidence. \n\n"
            "Your AI now understands its own weaknesses... but it can't fix them yet. \n\n"
            "Onward to Stage 3: build the Architect so the system can rewrite its own playbook using dynamic few-shot examples. Let's see if it can learn to learn 🥸"
        ),
        "stage3": (
            "⚡ [STAGE 3 CLEARED] The Architect is writing valid DSL! \n\n"
            "Your AI can now turn a failure diagnosis into a brand new prompt strategy. You have all the pieces of a self-improving system. \n\n"
            "Now, we put the ghost in the machine. Proceed to Stage 4 to build the Evolution Harness, connect the loop, and watch your creation come alive."
        ),
        "smoke": (
            "🚀 [MODAL SMOKE TEST: SUCCESS] \n\n"
            "Your strategy just executed in the cloud without collapsing. The engines look good, and you just saved yourself from discovering an expensive bug halfway through a full run. \n\n"
            "You are cleared for the full Stage 4 integration. Send it!"
        ),
        "stage4": (
            "🏆 [STAGE 4 CLEARED: MILESTONE 1 SECURED] \n\n"
            "Incredible work. The loop is closed, the Best-Parent selection is firing, and your AI just evolved"
            + (f" to hit {accuracy_text}" if accuracy_text is not None else "")
            + (f" (crushing the {baseline_text} baseline)." if baseline_text is not None else ".")
            + " You have officially secured your 6 points. Take a breath and celebrate—you just built a self-improving AI from scratch. \n\n"
            "But remember: we don't just build engines here. We race them. \n\n"
            "The training wheels are off. The sandbox is closed. It is time to enter The Crucible. \n\n"
            "Go read PHASE3_KAGGLE.md, tune this machine to its absolute limit, and I will see you on the global Leaderboard."
        ),
    }

    message = messages.get(stage)
    if message:
        print("\n" + _success(message) + "\n")

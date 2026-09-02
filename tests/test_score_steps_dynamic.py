import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts" / "evaluation"))
from score_steps_dynamic import build_gemini_payload, main_async  # noqa: E402


TREE = {
    "parents": {
        "axis_or_layout_misread": {
            "label": "Axis / layout misread",
            "children": [
                {"child_id": "axis_0", "top_terms": ["axis"], "exemplars": ["The axis is wrong."]}
            ],
        },
        "wrong_series_or_color": {
            "label": "Wrong series / color",
            "children": [
                {"child_id": "series_0", "top_terms": ["color"], "exemplars": ["Wrong color."]}
            ],
        },
    }
}


class TestScoreStepsDynamic(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.cleaned_path = os.path.join(self.test_dir, "cleaned.jsonl")
        self.pilot_ids_path = os.path.join(self.test_dir, "pilot_ids.json")
        self.tree_path = os.path.join(self.test_dir, "tree.json")
        self.image_dir = os.path.join(self.test_dir, "images")
        self.output_file = os.path.join(self.test_dir, "output.jsonl")
        os.makedirs(self.image_dir, exist_ok=True)

        Image.new("RGB", (50, 50), color="blue").save(
            os.path.join(self.image_dir, "42.jpg"), format="JPEG"
        )

        with open(self.cleaned_path, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "question_id": "42",
                        "rollout_index": 0,
                        "question": "Which line is highest?",
                        "ground_truth": "Blue",
                        "parsed_steps": ["Step 0: read the axis.", "Step 1: compare colors."],
                    }
                )
                + "\n"
            )

        with open(self.pilot_ids_path, "w", encoding="utf-8") as handle:
            json.dump({"question_ids": ["42"]}, handle)

        with open(self.tree_path, "w", encoding="utf-8") as handle:
            json.dump(TREE, handle)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("score_steps_dynamic.call_gemini_api")
    async def test_scores_rollout_and_writes_resumable_output(self, mock_call):
        mock_call.return_value = json.dumps(
            [
                {"step_index": 0, "scores": [{"criterion_id": "axis_0", "score": 3}]},
                {"step_index": 1, "scores": [{"criterion_id": "series_0", "score": 1}]},
            ]
        )

        await main_async(
            self.cleaned_path, self.pilot_ids_path, self.tree_path, self.image_dir, self.output_file
        )

        self.assertTrue(os.path.exists(self.output_file))
        with open(self.output_file, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["question_id"], "42")
        self.assertEqual(row["n_steps"], 2)
        self.assertEqual(row["scores"][0]["scores"][0]["score"], 3)
        self.assertEqual(mock_call.call_count, 1)  # rollout-batched: one call for both steps

        # Prompt sent to the (mocked) judge shows the full tree once and never the ground truth
        prompt_call = mock_call.call_args
        payload = prompt_call.args[1] if len(prompt_call.args) > 1 else prompt_call.kwargs["payload"]
        prompt_text = payload["contents"][0]["parts"][1]["text"]
        self.assertNotIn("Blue", prompt_text)
        self.assertIn("[axis_0]", prompt_text)
        self.assertIn("[series_0]", prompt_text)
        self.assertIn("Step 0: read the axis.", prompt_text)
        self.assertIn("Step 1: compare colors.", prompt_text)

    @patch("score_steps_dynamic.call_gemini_api")
    async def test_resume_skips_already_scored_rollout(self, mock_call):
        with open(self.output_file, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"question_id": "42", "rollout_index": 0}) + "\n")

        await main_async(
            self.cleaned_path, self.pilot_ids_path, self.tree_path, self.image_dir, self.output_file
        )

        mock_call.assert_not_called()

    @patch("score_steps_dynamic.call_gemini_api")
    async def test_question_outside_pilot_set_is_skipped(self, mock_call):
        with open(self.pilot_ids_path, "w", encoding="utf-8") as handle:
            json.dump({"question_ids": ["999"]}, handle)  # "42" is not in the pilot set

        await main_async(
            self.cleaned_path, self.pilot_ids_path, self.tree_path, self.image_dir, self.output_file
        )

        mock_call.assert_not_called()
        self.assertFalse(os.path.exists(self.output_file))


def test_build_gemini_payload_shape():
    payload = build_gemini_payload("YmFzZTY0", "hello")
    parts = payload["contents"][0]["parts"]
    assert parts[0]["inline_data"] == {"mime_type": "image/jpeg", "data": "YmFzZTY0"}
    assert parts[1]["text"] == "hello"


if __name__ == "__main__":
    unittest.main()

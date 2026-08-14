"""
test_evaluate_rollouts_meta.py

Unit tests for the `evaluate_rollouts_meta.py` script. 
It verifies JSON markdown stripping, API mocking, and ensures that the interrupt-resume 
intermediate saving logic correctly handles partially processed data without calling the API redundantly.
"""
import os
import json
import asyncio
import unittest
from unittest.mock import patch, AsyncMock
import tempfile
import sys
import shutil

# Add scripts directory to path to import the module
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "evaluation"))
from evaluate_rollouts_meta import main_async, extract_json_array_from_response, encode_image

class TestEvaluateRolloutsMeta(unittest.IsolatedAsyncioTestCase):
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.input_file = os.path.join(self.test_dir, "input.jsonl")
        self.output_file = os.path.join(self.test_dir, "output.jsonl")
        self.image_dir = os.path.join(self.test_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)
        
        # Create a valid minimal JPEG to avoid PIL errors
        self.dummy_image_path = os.path.join(self.image_dir, "123.jpg")
        from PIL import Image
        img = Image.new('RGB', (100, 100), color = 'red')
        img.save(self.dummy_image_path, format="JPEG")
            
        # Create dummy input data with 2 steps in 1 rollout
        dummy_data = {
            "question_id": "123",
            "rollout_index": 0,
            "question": "What is 1+1?",
            "ground_truth": "2",
            "parsed_steps": ["Step 0: 1+1", "Step 1: equals 2"]
        }
        with open(self.input_file, "w") as f:
            f.write(json.dumps(dummy_data) + "\n")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_extract_json(self):
        # Test markdown stripping and array parsing
        res = extract_json_array_from_response('```json\n[{"step_index": 0, "analysis": "ok", "score": 1}]\n```')
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["score"], 1)
        
        # Test missing codeblocks
        res2 = extract_json_array_from_response('[{"step_index": 1, "score": 0}]')
        self.assertEqual(res2[0]["score"], 0)

    @patch("evaluate_rollouts_meta.call_meta_api")
    async def test_processing_and_intermediate_saving(self, mock_call):
        # Mock the API to return a list of 2 objects
        mock_response = '[{"step_index": 0, "analysis": "ok", "score": 1}, {"step_index": 1, "analysis": "bad", "score": 0}]'
        mock_call.return_value = mock_response
        
        await main_async(self.input_file, self.output_file, self.image_dir, concurrency=2)
        
        self.assertTrue(os.path.exists(self.output_file))
        
        with open(self.output_file, "r") as f:
            lines = f.readlines()
            
        self.assertEqual(len(lines), 1)
        data = json.loads(lines[0])
        self.assertEqual(data["question_id"], "123")
        self.assertEqual(data["rollout_index"], 0)
        self.assertEqual(len(data["evaluations"]), 2)
        self.assertEqual(data["evaluations"][0]["score"], 1)
        self.assertEqual(data["evaluations"][1]["score"], 0)
        
        # Mock should be called exactly once per rollout
        self.assertEqual(mock_call.call_count, 1)
        
    @patch("evaluate_rollouts_meta.call_meta_api")
    async def test_resume_capability(self, mock_call):
        # Simulate that rollout 123_0 is already processed
        with open(self.output_file, "w") as f:
            f.write(json.dumps({"question_id": "123", "rollout_index": 0}) + "\n")
            
        await main_async(self.input_file, self.output_file, self.image_dir, concurrency=2)
        
        # The API should not be called because it was skipped
        self.assertEqual(mock_call.call_count, 0)
        
        with open(self.output_file, "r") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)

if __name__ == "__main__":
    unittest.main()

import os
import json
import asyncio
import unittest
from unittest.mock import patch, AsyncMock
import tempfile
import sys
import shutil

# Add scripts directory to path to import the module
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from evaluate_steps_meta import main_async, extract_json_from_response

class TestEvaluateStepsMeta(unittest.IsolatedAsyncioTestCase):
    
    def setUp(self):
        # Create temporary directories for test files
        self.test_dir = tempfile.mkdtemp()
        self.input_file = os.path.join(self.test_dir, "input.jsonl")
        self.output_file = os.path.join(self.test_dir, "output.jsonl")
        self.image_dir = os.path.join(self.test_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)
        
        # Create dummy image
        self.dummy_image_path = os.path.join(self.image_dir, "123.jpg")
        with open(self.dummy_image_path, "wb") as f:
            f.write(b"dummy image content")
            
        # Create dummy input data
        dummy_data = {
            "question_id": "123",
            "rollout_index": 0,
            "question": "What is 1+1?",
            "ground_truth": "2",
            "parsed_steps": ["Step 1: 1+1", "Step 2: equals 2"]
        }
        with open(self.input_file, "w") as f:
            f.write(json.dumps(dummy_data) + "\n")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_extract_json(self):
        # Test markdown stripping
        res = extract_json_from_response('```json\n{"analysis": "test", "score": 1}\n```')
        self.assertEqual(res["score"], 1)
        
        # Test plain json
        res2 = extract_json_from_response('{"analysis": "test2", "score": 0}')
        self.assertEqual(res2["score"], 0)
        
        # Test malformed
        res3 = extract_json_from_response('this is not json')
        self.assertIsNone(res3)

    @patch("evaluate_steps_meta.call_meta_api")
    async def test_processing_and_intermediate_saving(self, mock_call):
        # Mock the API call to return a valid JSON string
        mock_call.return_value = '{"analysis": "Looks good", "score": 1}'
        
        # Run main process
        await main_async(self.input_file, self.output_file, self.image_dir, concurrency=2)
        
        # Check if output file was created and contains 2 steps
        self.assertTrue(os.path.exists(self.output_file))
        
        with open(self.output_file, "r") as f:
            lines = f.readlines()
            
        self.assertEqual(len(lines), 2)
        
        step_1 = json.loads(lines[0])
        self.assertEqual(step_1["step_index"], 0)
        self.assertEqual(step_1["evaluation"]["score"], 1)
        
        step_2 = json.loads(lines[1])
        self.assertEqual(step_2["step_index"], 1)
        self.assertEqual(step_2["evaluation"]["score"], 1)
        
        # Verify call count (should be 2 steps)
        self.assertEqual(mock_call.call_count, 2)
        
    @patch("evaluate_steps_meta.call_meta_api")
    async def test_resume_capability(self, mock_call):
        # Simulate that step 0 is already in the output file (resuming)
        with open(self.output_file, "w") as f:
            f.write(json.dumps({"question_id": "123", "rollout_index": 0, "step_index": 0}) + "\n")
            
        mock_call.return_value = '{"analysis": "Looks good", "score": 1}'
        
        # Run main process
        await main_async(self.input_file, self.output_file, self.image_dir, concurrency=2)
        
        # The API should only be called 1 time for the remaining step 1
        self.assertEqual(mock_call.call_count, 1)
        
        # Output should now have 2 lines
        with open(self.output_file, "r") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 2)

if __name__ == "__main__":
    unittest.main()

# Violation Test Script

This script tests the `violations_test.json` file against the violation databases (`violations_v2.json` and `violations_v3.json`) using LLM to judge if each test case is correct.

## How It Works

1. **Load Test Cases**: Reads all test cases from `violations_test.json`
2. **Look Up Violations**: For each test case, retrieves the corresponding violation information using the `violation_id`
3. **LLM Judgment**: Uses an LLM to evaluate if the question matches the violation's `canonical_action` and `legal_description`
4. **Generate Report**: Outputs results showing correct/incorrect cases with explanations

## Setup

1. Create a `.env` file in the project root (copy from `.env.example`):
   ```bash
   cp .env.example .env
   ```

2. Configure your LLM settings in `.env`:
   
   **For OpenAI:**
   ```env
   LLM_BACKEND_TYPE=openai
   LLM_MODEL_NAME=gpt-3.5-turbo
   API_KEY=your_openai_api_key
   BASE_URL=https://api.openai.com/v1
   ```
   
   **For vLLM (local server):**
   ```env
   LLM_BACKEND_TYPE=vllm
   LLM_MODEL_NAME=Qwen3-32B
   API_KEY=dummy_key
   BASE_URL=http://localhost:8000/v1
   ```
   
   **For HuggingFace (local model):**
   ```env
   LLM_BACKEND_TYPE=huggingface
   LLM_MODEL_NAME=vinai/phobert-base
   LLM_DEVICE=cpu
   ```

## Usage

Run the test script:

```bash
python scripts/test_violations.py
```

## Output

The script will:
- Print detailed logs for each test case
- Show a summary with accuracy statistics
- List all incorrect cases with explanations
- Save results to `data/test_results.json`

### Example Output

```
================================================================================
Starting Violation Tests
================================================================================
Loaded 50 test cases

================================================================================
Testing TEST_001
Question: Chạy xe ô tô mà không để ý biển báo cấm thì bị phạt bao nhiêu tiền?
Violation ID: D168-A6-C1-Pa
Source: violations_v3.json
Canonical Action: Không tuân thủ tín hiệu biển báo, vạch kẻ đường
Result: ✓ CORRECT
Explanation: Câu hỏi đúng về hành vi không tuân thủ biển báo

================================================================================
TEST SUMMARY
================================================================================
Total Tests: 50
Correct: 45 ✓
Incorrect: 5 ✗
Accuracy: 90.00%
================================================================================
```

## Test File Format

### violations_test.json
```json
[
  {
    "test_id": "TEST_001",
    "question": "Chạy xe ô tô mà không để ý biển báo cấm thì bị phạt bao nhiêu tiền?",
    "violation_id": "D168-A6-C1-Pa"
  }
]
```

### Results File (test_results.json)
```json
{
  "total_tests": 50,
  "correct": 45,
  "incorrect": 5,
  "accuracy": 90.0,
  "results": [
    {
      "test_id": "TEST_001",
      "question": "...",
      "violation_id": "D168-A6-C1-Pa",
      "correct": true,
      "explanation": "...",
      "source": "v3",
      "canonical_action": "...",
      "penalty_info": "400_000 - 600_000 VNĐ"
    }
  ]
}
```

## Features

- ✅ Loops through all test cases
- ✅ Retrieves violation data from both v2 and v3 files
- ✅ Uses LLM to judge correctness based on semantic matching
- ✅ Provides detailed explanations for each judgment
- ✅ Generates comprehensive test report
- ✅ Saves results to JSON file
- ✅ Shows accuracy statistics
- ✅ Lists all correct and incorrect cases

## Troubleshooting

### "Failed to initialize LLM"
- Check your `.env` file exists and has correct values
- Verify API_KEY is valid (for OpenAI/vLLM)
- Ensure vLLM server is running (for vLLM backend)

### "Violation ID not found"
- Check that the `violation_id` in test case exists in either `violations_v2.json` or `violations_v3.json`
- Verify the ID format is correct (e.g., "D168-A6-C1-Pa")

### "Could not extract JSON from LLM response"
- The LLM might be returning unexpected format
- Try adjusting the prompt or using a different model
- Check LLM logs for more details

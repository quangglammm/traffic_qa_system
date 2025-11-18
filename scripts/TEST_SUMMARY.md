# Violation Testing System - Summary

## Overview

A comprehensive testing system has been created to validate the `violations_test.json` file against the violation databases using LLM-based semantic matching.

## Created Files

### 1. Main Test Script
**File**: `scripts/test_violations.py`

A full-featured testing script that:
- Loads 50 test cases from `violations_test.json`
- Retrieves violation data from `violations_v2.json` and `violations_v3.json`
- Uses LLM to semantically judge if questions match violations
- Generates detailed reports with explanations
- Saves results to `data/test_results.json`

**Key Features**:
- ✅ Automatic violation lookup by ID
- ✅ Smart source detection (v2 vs v3)
- ✅ LLM-based semantic judgment
- ✅ Detailed logging for each test case
- ✅ Comprehensive summary statistics
- ✅ JSON output for further analysis

### 2. Configuration Example
**File**: `.env.example`

Template configuration file with settings for:
- OpenAI API integration
- vLLM local server setup
- HuggingFace local models
- Other system configurations

### 3. Documentation
**File**: `scripts/README_TEST.md`

Complete documentation including:
- How the system works
- Setup instructions
- Configuration examples
- Usage guide
- Output format
- Troubleshooting tips

### 4. Quick Start Script
**File**: `scripts/run_test.sh`

Bash script for easy testing:
```bash
./scripts/run_test.sh
```

## How to Use

### Step 1: Configure LLM
Create a `.env` file from the example:
```bash
cp .env.example .env
```

Edit `.env` with your LLM configuration:
```env
LLM_BACKEND_TYPE=openai
LLM_MODEL_NAME=gpt-3.5-turbo
API_KEY=your_api_key_here
BASE_URL=https://api.openai.com/v1
```

### Step 2: Run Tests
Option A - Direct Python:
```bash
python scripts/test_violations.py
```

Option B - Quick start script:
```bash
./scripts/run_test.sh
```

### Step 3: Review Results
- Check console output for detailed logs
- Review `data/test_results.json` for full results
- Analyze accuracy and incorrect cases

## Test Data Structure

### Input: violations_test.json
- 50 test cases
- Each has: test_id, question, violation_id

### Reference Data:
- `violations_v2.json`: Decree 100/2019/NĐ-CP violations
- `violations_v3.json`: Decree 168/2024/NĐ-CP violations

### Output: test_results.json
```json
{
  "total_tests": 50,
  "correct": 45,
  "incorrect": 5,
  "accuracy": 90.0,
  "results": [...]
}
```

## LLM Judgment Process

For each test case, the LLM receives:
1. **User Question**: The question from the test case
2. **Canonical Action**: The violation's standard action description
3. **Legal Description**: Detailed legal text
4. **Additional Context**: Penalty info, detailed descriptions

The LLM evaluates:
- Does the question semantically match the violation?
- Returns: `correct` (bool) and `explanation` (string)

## Example Test Case Flow

```
TEST_001: "Chạy xe ô tô mà không để ý biển báo cấm thì bị phạt bao nhiêu tiền?"
↓
Look up violation: D168-A6-C1-Pa
↓
Found in: violations_v3.json
↓
Canonical Action: "Không tuân thủ tín hiệu biển báo, vạch kẻ đường"
Penalty: 400,000 - 600,000 VNĐ
↓
LLM Judges: ✓ CORRECT
Explanation: "Câu hỏi đúng về hành vi không tuân thủ biển báo"
```

## Benefits

1. **Automated Testing**: No manual checking needed
2. **Semantic Understanding**: LLM understands context, not just keywords
3. **Detailed Feedback**: Each judgment includes explanation
4. **Scalable**: Easy to add more test cases
5. **Configurable**: Works with multiple LLM backends
6. **Traceable**: Full audit trail in JSON output

## Next Steps

1. **Configure your LLM** in `.env` file
2. **Run the tests** using the script
3. **Review results** to identify issues
4. **Iterate**: Fix incorrect mappings and retest

## Troubleshooting

### Common Issues:

**Missing .env file**
```bash
cp .env.example .env
# Edit .env with your settings
```

**Invalid API Key**
- Verify your API_KEY in .env
- Check API key has proper permissions

**vLLM Server Not Running**
```bash
# Start vLLM server first
python -m vllm.entrypoints.openai.api_server --model <model_name>
```

**Import Errors**
```bash
# Install dependencies
pip install -r requirements.txt
```

## Support

For issues or questions:
1. Check `scripts/README_TEST.md` for detailed docs
2. Review error logs in console output
3. Verify LLM configuration in .env
4. Check that all data files exist

---

**Created**: November 17, 2025
**Test Cases**: 50
**Violation Sources**: violations_v2.json, violations_v3.json

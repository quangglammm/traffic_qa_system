# End-to-End System Testing

Complete end-to-end testing framework for the Traffic QA System that validates the entire pipeline from question to answer.

## What It Tests

The E2E test script validates:

1. **Query Processing**: Question parsing and intent detection
2. **Retrieval System**: Vector search and candidate ranking
3. **Knowledge Graph**: Violation detail retrieval
4. **LLM Generation**: Answer generation and violation selection
5. **Overall Accuracy**: Whether the system returns the correct violation

## Metrics Tracked

### Overall Performance
- **Exact Match Accuracy**: % of questions where retrieved violation_id matches expected
- **Retrieval Success Rate**: % of questions where expected violation appears in candidates
- **Average Response Time**: Time taken per question
- **Total Test Time**: Total execution time

### Retrieval Metrics
- **Mean Reciprocal Rank (MRR)**: Average of 1/rank for expected violations
- **Recall@1**: % of expected violations that appear as top result
- **Recall@3**: % of expected violations in top 3 results
- **Recall@5**: % of expected violations in top 5 results

### Per-Question Tracking
- Expected violation ID vs Retrieved violation ID
- Rank of expected violation in retrieval candidates
- Similarity score of expected violation
- Top 5 retrieval candidates
- Response time
- Generated answer and citation

## Installation

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

## Configuration

Make sure your `.env` file is properly configured:

```env
# LLM Configuration
LLM_BACKEND_TYPE=openai
LLM_MODEL_NAME=gpt-3.5-turbo
API_KEY=your_api_key

# Vector Store
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=traffic_violations

# Knowledge Graph
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASS=your_password

# Embedding
EMBEDDING_BACKEND_TYPE=openai
EMBEDDING_MODEL_NAME=text-embedding-ada-002
```

## Usage

### Run All Tests

```bash
python scripts/test_system_e2e.py
```

### Run Limited Tests (Quick Check)

```bash
# Test first 10 questions only
python scripts/test_system_e2e.py --max-tests 10
```

### Custom Output Location

```bash
python scripts/test_system_e2e.py --output results/my_test_run.json
```

### Combined Options

```bash
python scripts/test_system_e2e.py --max-tests 20 --output results/quick_test.json
```

## Output Files

### 1. JSON Results (`data/e2e_test_results.json`)

Complete results with all metrics:

```json
{
  "test_info": {
    "timestamp": "2025-11-17T10:30:00",
    "total_tests": 50
  },
  "overall_metrics": {
    "exact_match_accuracy": 85.0,
    "exact_matches": 42,
    "retrieval_success_rate": 92.0,
    "retrieval_successes": 46,
    "avg_response_time_seconds": 2.3
  },
  "retrieval_metrics": {
    "mean_reciprocal_rank": 0.8234,
    "recall_at_1": 78.0,
    "recall_at_3": 88.0,
    "recall_at_5": 92.0
  },
  "results": [
    {
      "test_id": "TEST_001",
      "question": "...",
      "expected_violation_id": "D168-A6-C1-Pa",
      "retrieved_violation_id": "D168-A6-C1-Pa",
      "exact_match": true,
      "response_time_seconds": 2.1,
      "answer": "...",
      "citation": "...",
      "retrieval_metrics": {
        "expected_in_candidates": true,
        "expected_rank": 1,
        "expected_similarity": 0.8765,
        "top_5_candidates": [...]
      }
    }
  ]
}
```

### 2. CSV Results (`data/e2e_test_results.csv`)

Simplified CSV for Excel/spreadsheet analysis:

```csv
test_id,question,expected_violation_id,retrieved_violation_id,exact_match,expected_rank,expected_similarity,response_time_seconds
TEST_001,"...",D168-A6-C1-Pa,D168-A6-C1-Pa,True,1,0.8765,2.1
```

## Understanding the Results

### Success Criteria

✅ **Perfect Match**: `exact_match = True` and `expected_rank = 1`
- System correctly identified and selected the right violation

✅ **Good Match**: `exact_match = True` and `expected_rank <= 3`
- Expected violation was in top-3, LLM correctly selected it

⚠️ **Retrieval Issue**: `exact_match = False` but `expected_in_candidates = True`
- Retrieval found the right violation, but LLM selected wrong one
- May need LLM prompt tuning

❌ **Complete Miss**: `exact_match = False` and `expected_in_candidates = False`
- Retrieval didn't find expected violation
- May need embedding model improvement or query parsing fixes

### Example Output

```
================================================================================
STARTING END-TO-END SYSTEM TEST
================================================================================
Loaded 50 test cases

================================================================================
Testing TEST_001
Question: Chạy xe ô tô mà không để ý biển báo cấm thì bị phạt bao nhiêu tiền?
Expected Violation ID: D168-A6-C1-Pa
Retrieval: Found 10 candidates
Expected violation at rank 1 with similarity 0.8765
Retrieved Violation ID: D168-A6-C1-Pa
Match: ✓ CORRECT
Response Time: 2.1s

...

================================================================================
TEST SUMMARY
================================================================================

Test Run: 2025-11-17T10:30:00
Total Tests: 50

--- OVERALL PERFORMANCE ---
Exact Match Accuracy: 85.0% (42/50)
Retrieval Success Rate: 92.0% (46/50)
Average Response Time: 2.3s
Total Test Time: 115.0s

--- RETRIEVAL METRICS ---
Mean Reciprocal Rank (MRR): 0.8234
Recall@1: 78.0%
Recall@3: 88.0%
Recall@5: 92.0%

--- FAILED CASES ---
Total Failed: 8

✗ TEST_015: Đỗ xe ô tô che khuất đèn giao thông bị phạt mức nào?
  Expected: D168-A6-C3-Pđ
  Retrieved: D168-A6-C2-Pd
  Note: Expected violation was at rank 3
  Top candidate: D168-A6-C2-Pd (sim: 0.8234)

--- SUCCESSFUL CASES ---
Total Successful: 42
  Rank 1: 39 cases
  Rank 2-3: 3 cases
  Rank 4-5: 0 cases
  Rank 6+: 0 cases
```

## Analyzing Results

### 1. Check Overall Accuracy

High exact match accuracy (>80%) indicates good system performance.

### 2. Compare Retrieval vs Final Selection

If `retrieval_success_rate >> exact_match_accuracy`:
- Retrieval is good, but LLM selection needs improvement
- Consider adjusting LLM prompt for violation selection

If `retrieval_success_rate ≈ exact_match_accuracy`:
- System is well-balanced
- Focus on improving retrieval (embeddings, query parsing)

### 3. Examine Failed Cases

Look at the detailed results for failed cases:
- If expected violation has low rank → improve query parsing or embeddings
- If expected violation at rank 1 but wrong selection → improve LLM selection prompt
- If expected violation not in candidates → check if it exists in database

### 4. Performance Analysis

- Response time >5s → consider optimization
- High MRR (>0.8) → good retrieval quality
- High Recall@1 (>70%) → excellent first-result accuracy

## Troubleshooting

### "Failed to initialize container"
Check your `.env` file:
- Verify API keys are valid
- Ensure Neo4j is running
- Check ChromaDB directory exists

### "Test file not found"
Ensure `data/violations_test.json` exists in your project

### Slow Performance
- Use `--max-tests 10` for quick checks
- Consider using faster LLM models
- Check network latency to API services

### Low Accuracy
1. Verify your vector store is populated: Check `chroma_db` directory
2. Ensure Neo4j has violation data
3. Review your embedding model configuration
4. Check LLM model quality

## Integration with CI/CD

Add to your testing pipeline:

```bash
# Run quick smoke test
python scripts/test_system_e2e.py --max-tests 10

# Run full test suite
python scripts/test_system_e2e.py

# Check accuracy threshold
python -c "
import json
with open('data/e2e_test_results.json') as f:
    results = json.load(f)
    accuracy = results['overall_metrics']['exact_match_accuracy']
    if accuracy < 80:
        exit(1)  # Fail if accuracy below threshold
"
```

## Advanced Usage

### Custom Test Set

Create your own test file following the format:

```json
[
  {
    "test_id": "CUSTOM_001",
    "question": "Your question here",
    "violation_id": "Expected-Violation-ID"
  }
]
```

Then run:

```bash
python scripts/test_system_e2e.py --test-file path/to/your/test.json
```

### Programmatic Usage

```python
from scripts.test_system_e2e import SystemE2ETester
from src.presentation.di_container import Container

# Initialize
container = Container()
tester = SystemE2ETester(container)

# Test single question
result = tester.test_single_question({
    'test_id': 'TEST_001',
    'question': 'Your question',
    'violation_id': 'Expected-ID'
})

print(f"Match: {result['exact_match']}")
print(f"Retrieved: {result['retrieved_violation_id']}")
```

## Next Steps

1. **Run baseline test**: Get initial accuracy metrics
2. **Identify weak areas**: Review failed cases
3. **Iterate improvements**: 
   - Tune embeddings
   - Improve query parsing
   - Adjust LLM prompts
4. **Re-test**: Validate improvements
5. **Monitor**: Track accuracy over time

---

**Created**: November 17, 2025  
**Test Cases**: 50 questions from violations_test.json  
**Output**: JSON + CSV results with comprehensive metrics

# End-to-End System Testing - Complete Guide

## Overview

A comprehensive end-to-end testing framework has been created to validate your entire Traffic QA System. This tests the complete pipeline from user question to final answer, tracking retrieval accuracy, system performance, and answer quality.

## 🎯 What Gets Tested

### Complete Pipeline
```
User Question 
    ↓
Query Parsing (LLM extracts intent & entities)
    ↓
Vector Embedding (Question → Vector)
    ↓
Retrieval (Find top-K similar violations)
    ↓
Knowledge Graph Enrichment (Get full violation details)
    ↓
LLM Selection (Choose best violation from candidates)
    ↓
Answer Generation (Create natural language response)
    ↓
Final Answer (with violation_id, citation, fine info)
```

### Key Validation
- ✅ Does the system retrieve the expected violation?
- ✅ Is the expected violation in the top-K candidates?
- ✅ Does the LLM select the correct violation?
- ✅ What's the retrieval rank of expected violation?
- ✅ How fast is the system?

## 📁 Created Files

### 1. Main Test Script
**File**: `scripts/test_system_e2e.py` (500+ lines)

Comprehensive testing framework that:
- Loads 50 test cases from `violations_test.json`
- Runs each question through the complete system
- Tracks retrieval candidates and ranking
- Compares retrieved violation_id vs expected violation_id
- Calculates detailed metrics (accuracy, MRR, Recall@K)
- Generates JSON and CSV reports

**Key Features**:
- ✅ Full pipeline testing
- ✅ Retrieval analysis (tracks if expected violation is found)
- ✅ Ranking metrics (MRR, Recall@1/3/5)
- ✅ Performance tracking (response time)
- ✅ Detailed failure analysis
- ✅ Structured output (JSON + CSV)

### 2. Quick Start Script
**File**: `scripts/run_e2e_test.sh`

Bash wrapper for easy testing:
```bash
./scripts/run_e2e_test.sh              # Run all tests
./scripts/run_e2e_test.sh --quick      # Quick test (10 questions)
./scripts/run_e2e_test.sh --max-tests 20  # Test first 20
```

### 3. Documentation
**File**: `scripts/README_E2E.md`

Complete guide including:
- How the testing works
- Metrics explanation
- Usage examples
- Result interpretation
- Troubleshooting
- Integration with CI/CD

## 🚀 Quick Start

### Prerequisites
1. Ensure `.env` is configured with your LLM and database settings
2. Make sure ChromaDB is populated with violations
3. Verify Neo4j is running with violation data

### Run Tests

**Option 1: Direct Python**
```bash
python scripts/test_system_e2e.py
```

**Option 2: Quick Start Script**
```bash
# Full test (all 50 questions)
./scripts/run_e2e_test.sh

# Quick test (first 10 questions)
./scripts/run_e2e_test.sh --quick

# Custom number of tests
./scripts/run_e2e_test.sh --max-tests 25
```

**Option 3: With Custom Output**
```bash
python scripts/test_system_e2e.py --output results/my_test.json
```

## 📊 Understanding Results

### Metrics Explained

#### 1. Exact Match Accuracy
**What**: Percentage of questions where `retrieved_violation_id == expected_violation_id`

**Good**: >80%  
**Acceptable**: 70-80%  
**Needs Work**: <70%

#### 2. Retrieval Success Rate
**What**: Percentage of questions where expected violation appears in retrieval candidates

**Good**: >90%  
**Acceptable**: 80-90%  
**Needs Work**: <80%

#### 3. Mean Reciprocal Rank (MRR)
**What**: Average of 1/rank for expected violations

**Good**: >0.8  
**Acceptable**: 0.6-0.8  
**Needs Work**: <0.6

**Example**: If expected violation is at rank 1, 2, 1, 3 in 4 tests:
MRR = (1/1 + 1/2 + 1/1 + 1/3) / 4 = 0.7083

#### 4. Recall@K
**What**: Percentage of expected violations in top-K results

- **Recall@1**: Expected violation is the top result
- **Recall@3**: Expected violation in top 3
- **Recall@5**: Expected violation in top 5

**Target**: Recall@3 > 85%

### Example Output

```
================================================================================
TEST SUMMARY
================================================================================

Test Run: 2025-11-17T10:30:00
Total Tests: 50

--- OVERALL PERFORMANCE ---
Exact Match Accuracy: 84.0% (42/50)
Retrieval Success Rate: 94.0% (47/50)
Average Response Time: 2.3s
Total Test Time: 115.0s

--- RETRIEVAL METRICS ---
Mean Reciprocal Rank (MRR): 0.8456
Recall@1: 76.0%
Recall@3: 90.0%
Recall@5: 94.0%

--- FAILED CASES ---
Total Failed: 8

✗ TEST_015: Đỗ xe ô tô che khuất đèn giao thông bị phạt mức nào?
  Expected: D168-A6-C3-Pđ
  Retrieved: D168-A6-C2-Pd
  Note: Expected violation was at rank 3
  Top candidate: D168-A6-C2-Pd (sim: 0.8234)
```

### Interpreting Failures

**Scenario 1**: Expected violation at rank 1, but wrong ID retrieved
- **Issue**: LLM selection problem
- **Fix**: Improve LLM prompt for violation selection

**Scenario 2**: Expected violation at rank 5+
- **Issue**: Retrieval/ranking problem
- **Fix**: Improve embeddings or query parsing

**Scenario 3**: Expected violation NOT in candidates
- **Issue**: Fundamental retrieval failure
- **Fix**: Check embeddings, verify data in vector store

## 📈 Output Files

### 1. JSON Results (`data/e2e_test_results.json`)

Complete structured results:

```json
{
  "test_info": {
    "timestamp": "2025-11-17T10:30:00",
    "total_tests": 50
  },
  "overall_metrics": {
    "exact_match_accuracy": 84.0,
    "exact_matches": 42,
    "retrieval_success_rate": 94.0,
    "avg_response_time_seconds": 2.3
  },
  "retrieval_metrics": {
    "mean_reciprocal_rank": 0.8456,
    "recall_at_1": 76.0,
    "recall_at_3": 90.0,
    "recall_at_5": 94.0
  },
  "results": [
    {
      "test_id": "TEST_001",
      "question": "Chạy xe ô tô mà không để ý biển báo cấm...",
      "expected_violation_id": "D168-A6-C1-Pa",
      "retrieved_violation_id": "D168-A6-C1-Pa",
      "exact_match": true,
      "response_time_seconds": 2.1,
      "answer": "Nếu bạn điều khiển ô tô...",
      "citation": "Điểm a, Khoản 1, Điều 6...",
      "fine": {
        "min_vnd": "400000",
        "max_vnd": "600000"
      },
      "retrieval_metrics": {
        "expected_in_candidates": true,
        "expected_rank": 1,
        "expected_similarity": 0.8765,
        "top_5_candidates": [
          {"violation_id": "D168-A6-C1-Pa", "similarity": 0.8765},
          {"violation_id": "D168-A6-C1-Pb", "similarity": 0.7543}
        ]
      }
    }
  ]
}
```

### 2. CSV Results (`data/e2e_test_results.csv`)

For Excel/spreadsheet analysis:
- Easy sorting and filtering
- Pivot tables
- Quick visualization
- Share with non-technical team members

## 🔍 Analysis Workflow

### Step 1: Run Baseline Test
```bash
./scripts/run_e2e_test.sh
```

### Step 2: Check Overall Metrics
- Is Exact Match Accuracy acceptable?
- Is Retrieval Success Rate high?
- Compare the two: if retrieval >> accuracy, focus on LLM selection

### Step 3: Analyze Failed Cases
Review the failed cases section:
- Group by failure type (retrieval vs selection)
- Look for patterns (specific violation types failing?)
- Check similarity scores

### Step 4: Deep Dive
Open `data/e2e_test_results.json`:
- Sort by `exact_match: false`
- Check `retrieval_metrics.expected_rank`
- Review `top_5_candidates` to see what was retrieved

### Step 5: Make Improvements
Based on findings:
- **Low retrieval success**: Improve embeddings or query parsing
- **High retrieval, low accuracy**: Tune LLM selection prompt
- **Specific patterns failing**: Add more training data or rules

### Step 6: Re-test
```bash
./scripts/run_e2e_test.sh
```

Compare new results with baseline.

## 🎓 Best Practices

### Regular Testing
- Run tests after any system changes
- Track metrics over time
- Set minimum thresholds (e.g., accuracy > 80%)

### Gradual Improvement
- Don't expect 100% accuracy immediately
- Focus on highest-impact failures first
- Iterate in small steps

### Data Quality
- Ensure test cases are valid
- Verify expected violation IDs exist in database
- Keep test set representative of real usage

### Performance Monitoring
- Track response times
- Set performance budgets
- Optimize slow components

## 🚨 Troubleshooting

### Test Fails to Start

**Error**: "Failed to initialize container"
```bash
# Check .env configuration
cat .env

# Verify Neo4j is running
docker ps | grep neo4j

# Check ChromaDB directory
ls -la chroma_db/
```

### Low Accuracy Results

**Scenario**: Accuracy < 60%
1. Check if vector store is populated
2. Verify test violation IDs exist in database
3. Review query parsing quality
4. Test individual components

### Slow Performance

**Scenario**: Response time > 5s per question
1. Use smaller test set: `--max-tests 10`
2. Check API latency
3. Optimize embedding model
4. Consider caching

### CSV Not Generated

**Issue**: Only JSON created, no CSV
- Check file permissions
- Verify Python csv module available
- Review error logs

## 📚 Additional Resources

### Related Scripts
- `scripts/test_violations.py`: LLM-based validation of test cases
- `scripts/import_data.py`: Import violations into database
- `scripts/run_test.sh`: Quick test runner

### Configuration Files
- `.env`: System configuration
- `requirements.txt`: Python dependencies
- `docker-compose.yml`: Infrastructure setup

## 🎯 Success Criteria

### Minimum Viable Performance
- Exact Match Accuracy: >75%
- Retrieval Success Rate: >85%
- Recall@3: >80%
- MRR: >0.7
- Avg Response Time: <3s

### Production Ready
- Exact Match Accuracy: >85%
- Retrieval Success Rate: >95%
- Recall@3: >90%
- MRR: >0.8
- Avg Response Time: <2s

### Excellent Performance
- Exact Match Accuracy: >90%
- Retrieval Success Rate: >98%
- Recall@3: >95%
- MRR: >0.85
- Avg Response Time: <1.5s

---

**Created**: November 17, 2025  
**Test Coverage**: 50 end-to-end scenarios  
**Output Formats**: JSON (detailed) + CSV (analysis)  
**Metrics**: Accuracy, MRR, Recall@K, Performance

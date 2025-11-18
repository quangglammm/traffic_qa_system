#!/bin/bash

# Quick start script for E2E system testing

echo "=========================================="
echo "End-to-End System Test - Quick Start"
echo "=========================================="
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found!"
    echo "Please create .env file with your configuration."
    echo "See .env.example for reference."
    exit 1
fi

echo "✅ Found .env file"
echo ""

# Check if test file exists
if [ ! -f data/violations_test.json ]; then
    echo "❌ Test file not found: data/violations_test.json"
    exit 1
fi

echo "✅ Found test file"
echo ""

# Parse command line arguments
MAX_TESTS=""
OUTPUT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            MAX_TESTS="--max-tests 10"
            echo "🚀 Quick mode: Testing first 10 questions only"
            shift
            ;;
        --max-tests)
            MAX_TESTS="--max-tests $2"
            echo "🚀 Limited mode: Testing first $2 questions"
            shift 2
            ;;
        --output)
            OUTPUT="--output $2"
            echo "📁 Custom output: $2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick              Run quick test (first 10 questions)"
            echo "  --max-tests N        Test first N questions"
            echo "  --output FILE        Save results to FILE"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                   # Run all tests"
            echo "  $0 --quick           # Quick test (10 questions)"
            echo "  $0 --max-tests 20    # Test first 20 questions"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo ""
echo "Starting E2E system test..."
echo "This will test the complete pipeline: Query → Retrieval → Generation"
echo ""

# Run the test script
python scripts/test_system_e2e.py $MAX_TESTS $OUTPUT

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "✅ Test completed successfully!"
    echo "=========================================="
    echo ""
    echo "Results saved to:"
    if [ -z "$OUTPUT" ]; then
        echo "  - JSON: data/e2e_test_results.json"
        echo "  - CSV:  data/e2e_test_results.csv"
    else
        OUTPUT_FILE=$(echo $OUTPUT | cut -d' ' -f2)
        echo "  - JSON: $OUTPUT_FILE"
        CSV_FILE="${OUTPUT_FILE%.json}.csv"
        echo "  - CSV:  $CSV_FILE"
    fi
    echo ""
    echo "Next steps:"
    echo "  1. Review the test summary above"
    echo "  2. Check detailed results in JSON file"
    echo "  3. Analyze failed cases to identify issues"
    echo "  4. Open CSV in Excel/Sheets for easy analysis"
else
    echo "=========================================="
    echo "❌ Test failed or interrupted"
    echo "=========================================="
    echo ""
    echo "Please check the error messages above."
fi

echo ""

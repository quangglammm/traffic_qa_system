"""
Script to test violations_test.json against violations_v2.json and violations_v3.json
Uses LLM to judge if the answer matches the question based on canonical_action and legal_description
"""

import json
import os
import sys
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.infrastructure.adapters.llm_adapter import GeneralLLMAdapter
from src.infrastructure.config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ViolationTester:
    """Test violations using LLM to judge correctness"""
    
    def __init__(self, llm_adapter: GeneralLLMAdapter):
        self.llm = llm_adapter
        self.violations_v2 = {}
        self.violations_v3 = {}
        
    def load_violation_data(self, v2_path: str, v3_path: str):
        """Load violations data from v2 and v3 JSON files"""
        logger.info("Loading violations data...")
        
        with open(v2_path, 'r', encoding='utf-8') as f:
            v2_data = json.load(f)
            self.violations_v2 = {item['id']: item for item in v2_data}
            
        with open(v3_path, 'r', encoding='utf-8') as f:
            v3_data = json.load(f)
            self.violations_v3 = {item['id']: item for item in v3_data}
            
        logger.info(f"Loaded {len(self.violations_v2)} violations from v2")
        logger.info(f"Loaded {len(self.violations_v3)} violations from v3")
        
    def get_violation_by_id(self, violation_id: str) -> Optional[Dict]:
        """Get violation data by ID from v2 or v3"""
        if violation_id in self.violations_v3:
            return self.violations_v3[violation_id]
        elif violation_id in self.violations_v2:
            return self.violations_v2[violation_id]
        else:
            return None
            
    def judge_with_llm(self, question: str, canonical_action: str, legal_description: str, 
                      detailed_description: str = "", penalty_info: str = "") -> Tuple[bool, str]:
        """
        Use LLM to judge if the question matches the violation information
        
        Args:
            question: The test question
            canonical_action: The canonical action from violation data
            legal_description: The legal description from violation data
            detailed_description: Additional detailed description
            penalty_info: Penalty information
            
        Returns:
            Tuple of (is_correct: bool, explanation: str)
        """
        prompt = f"""Bạn là một chuyên gia về luật giao thông Việt Nam. Nhiệm vụ của bạn là đánh giá xem câu hỏi của người dùng có khớp với thông tin vi phạm được cung cấp hay không.

Câu hỏi: {question}

Thông tin vi phạm:
- Hành vi vi phạm: {canonical_action}
- Mô tả pháp lý: {legal_description}
{f"- Chi tiết: {detailed_description}" if detailed_description else ""}
{f"- Mức phạt: {penalty_info}" if penalty_info else ""}

Hãy phân tích xem câu hỏi có đang hỏi về hành vi vi phạm này không? 
Trả lời theo định dạng JSON:
{{
    "correct": true/false,
    "explanation": "Giải thích ngắn gọn tại sao đúng hoặc sai"
}}

JSON:"""

        try:
            response = self.llm._generate_text(prompt, max_length=512)
            
            # Extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
                return result.get("correct", False), result.get("explanation", "No explanation provided")
            else:
                logger.warning(f"Could not extract JSON from LLM response: {response}")
                return False, "Failed to parse LLM response"
                
        except Exception as e:
            logger.error(f"Error in LLM judgment: {e}")
            return False, f"Error: {str(e)}"
            
    def format_penalty_info(self, penalty: Dict) -> str:
        """Format penalty information into a readable string"""
        if not penalty:
            return ""
            
        fine_min = penalty.get('fine_min_vnd', '')
        fine_max = penalty.get('fine_max_vnd', '')
        
        if fine_min and fine_max:
            return f"{fine_min} - {fine_max} VNĐ"
        elif fine_min:
            return f"Từ {fine_min} VNĐ"
        elif fine_max:
            return f"Đến {fine_max} VNĐ"
        return ""
        
    def test_single_case(self, test_case: Dict) -> Dict:
        """Test a single test case"""
        test_id = test_case.get('test_id')
        question = test_case.get('question')
        violation_id = test_case.get('violation_id')
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing {test_id}")
        logger.info(f"Question: {question}")
        logger.info(f"Violation ID: {violation_id}")
        
        # Get violation data
        violation = self.get_violation_by_id(violation_id)
        
        if not violation:
            logger.error(f"Violation ID {violation_id} not found!")
            return {
                'test_id': test_id,
                'question': question,
                'violation_id': violation_id,
                'correct': False,
                'explanation': f"Violation ID {violation_id} not found in database",
                'source': 'N/A'
            }
        
        # Determine source
        source = 'v3' if violation_id in self.violations_v3 else 'v2'
        
        # Extract violation info
        canonical_action = violation.get('canonical_action', '')
        legal_description = violation.get('legal_description', '')
        detailed_description = violation.get('detailed_description', '')
        penalty_info = self.format_penalty_info(violation.get('penalty', {}))
        
        logger.info(f"Source: violations_{source}.json")
        logger.info(f"Canonical Action: {canonical_action}")
        
        # Judge with LLM
        is_correct, explanation = self.judge_with_llm(
            question, 
            canonical_action, 
            legal_description,
            detailed_description,
            penalty_info
        )
        
        logger.info(f"Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
        logger.info(f"Explanation: {explanation}")
        
        return {
            'test_id': test_id,
            'question': question,
            'violation_id': violation_id,
            'correct': is_correct,
            'explanation': explanation,
            'source': source,
            'canonical_action': canonical_action,
            'penalty_info': penalty_info
        }
        
    def run_all_tests(self, test_file_path: str) -> Dict:
        """Run all tests and generate report"""
        logger.info("="*80)
        logger.info("Starting Violation Tests")
        logger.info("="*80)
        
        # Load test cases
        with open(test_file_path, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
            
        logger.info(f"Loaded {len(test_cases)} test cases")
        
        results = []
        correct_count = 0
        incorrect_count = 0
        
        # Test each case
        for test_case in test_cases:
            result = self.test_single_case(test_case)
            results.append(result)
            
            if result['correct']:
                correct_count += 1
            else:
                incorrect_count += 1
        
        # Generate summary
        total = len(results)
        accuracy = (correct_count / total * 100) if total > 0 else 0
        
        summary = {
            'total_tests': total,
            'correct': correct_count,
            'incorrect': incorrect_count,
            'accuracy': accuracy,
            'results': results
        }
        
        return summary
        
    def print_summary(self, summary: Dict):
        """Print test summary"""
        logger.info("\n" + "="*80)
        logger.info("TEST SUMMARY")
        logger.info("="*80)
        logger.info(f"Total Tests: {summary['total_tests']}")
        logger.info(f"Correct: {summary['correct']} ✓")
        logger.info(f"Incorrect: {summary['incorrect']} ✗")
        logger.info(f"Accuracy: {summary['accuracy']:.2f}%")
        logger.info("="*80)
        
        # Print incorrect cases
        if summary['incorrect'] > 0:
            logger.info("\nINCORRECT TEST CASES:")
            logger.info("-"*80)
            for result in summary['results']:
                if not result['correct']:
                    logger.info(f"\n{result['test_id']}: {result['question']}")
                    logger.info(f"  Violation ID: {result['violation_id']}")
                    logger.info(f"  Expected: {result['canonical_action']}")
                    logger.info(f"  Explanation: {result['explanation']}")
        
        # Print correct cases
        logger.info("\nCORRECT TEST CASES:")
        logger.info("-"*80)
        for result in summary['results']:
            if result['correct']:
                logger.info(f"✓ {result['test_id']}: {result['question']}")
                

def main():
    """Main function to run the tests"""
    # Paths
    data_dir = project_root / "data"
    test_file = data_dir / "violations_test.json"
    v2_file = data_dir / "violations_v2.json"
    v3_file = data_dir / "violations_v3.json"
    
    # Check if files exist
    for file_path in [test_file, v2_file, v3_file]:
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return
    
    # Initialize LLM
    logger.info("Initializing LLM...")
    logger.info(f"Backend: {settings.LLM_BACKEND_TYPE}")
    logger.info(f"Model: {settings.LLM_MODEL_NAME}")
    
    try:
        llm_adapter = GeneralLLMAdapter(
            backend_type=settings.LLM_BACKEND_TYPE,
            model_name=settings.LLM_MODEL_NAME,
            device=settings.LLM_DEVICE,
            api_key=settings.API_KEY,
            base_url=settings.BASE_URL
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        logger.error("Please check your configuration in .env file")
        return
    
    # Create tester
    tester = ViolationTester(llm_adapter)
    
    # Load violation data
    tester.load_violation_data(str(v2_file), str(v3_file))
    
    # Run tests
    summary = tester.run_all_tests(str(test_file))
    
    # Print summary
    tester.print_summary(summary)
    
    # Save results to file
    output_file = data_dir / "test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

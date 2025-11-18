# filename: run_tests.py
"""
Chạy bộ test 50 câu hỏi thực tế để đo độ chính xác của hệ thống Traffic QA
→ Dùng hoàn toàn container đã cấu hình sẵn (không khởi tạo lại bất cứ thứ gì)
→ In kết quả chi tiết + lưu file báo cáo
"""

import json
import asyncio
from pathlib import Path
from datetime import datetime

from src.presentation.di_container import Container
from src.application.use_cases.ask_question_use_case import AskQuestionUseCase

container = Container()
container.wire(modules=[__name__])

# Get use case from container
ask_question_use_case = container.ask_question_use_case()


async def run_traffic_qa_tests():
    # Đọc file test
    test_file = Path("data/violations_test.json")
    if not test_file.exists():
        print("Không tìm thấy file violations_test.json")
        return

    with open(test_file, "r", encoding="utf-8") as f:
        tests = json.load(f)

    print(f"BẮT ĐẦU KIỂM THỬ HỆ THỐNG TRAFFIC QA")
    print(f"Thời gian: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"Số câu hỏi: {len(tests)}\n")
    print("-" * 80)

    correct = 0
    results = []

    for idx, item in enumerate(tests, start=1):
        question = item["question"]
        expected_id = item["violation_id"]

        print(f"[{idx:02d}/50] {question}")

        # Gọi hệ thống (đây là call thật 100% như khi user hỏi trên Gradio)
        response = ask_question_use_case.execute(question)

        predicted_id = "NOT_FOUND"
        if response.violation_found and response.violation_found.id:
            predicted_id = response.violation_found.id.strip()

        is_correct = predicted_id == expected_id.strip()
        status = "ĐÚNG" if is_correct else "SAI"

        if is_correct:
            correct += 1

        print(f"    Dự đoán : {predicted_id}")
        print(f"    Đáp án  : {expected_id} → {status}\n")

        results.append({
            "STT": idx,
            "question": question,
            "expected": expected_id,
            "predicted": predicted_id,
            "correct": is_correct,
            "answer": response.answer,
            "citation": response.citation or ""
        })

    # === BÁO CÁO TỔNG KẾT ===
    accuracy = correct / len(tests) * 100
    print("=" * 80)
    print("                   KẾT QUẢ KIỂM THỬ HOÀN CHỈNH")
    print("=" * 80)
    print(f"Tổng câu hỏi       : {len(tests)}")
    print(f"Số câu đúng        : {correct}")
    print(f"Số câu sai         : {len(tests) - correct}")
    print(f"ĐỘ CHÍNH XÁC       : {accuracy:.2f}%")
    print("=" * 80)

    # In ra các câu sai để dễ debug
    wrongs = [r for r in results if not r["correct"]]
    if wrongs:
        print(f"\nCÁC CÂU BỊ SAI ({len(wrongs)}):")
        for r in wrongs:
            print(f"• {r['STT']:02d}. Dự đoán: {r['predicted']} | Đúng: {r['expected']}")
            print(f"   Câu hỏi: {r['question']}\n")

    # Lưu kết quả chi tiết
    output_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nBáo cáo chi tiết đã lưu: {output_file}")
    print("Chạy xong! Chúc bạn đạt ≥ 95% accuracy")


if __name__ == "__main__":
    asyncio.run(run_traffic_qa_tests())
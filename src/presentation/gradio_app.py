"""
Gradio UI for Traffic Violation QA System.

This module provides an interactive chat interface for testing
the traffic violation question answering system.
"""
import gradio as gr
import logging
from typing import List, Tuple
from src.presentation.di_container import Container
from src.application.use_cases.ask_question_use_case import AskQuestionUseCase

logger = logging.getLogger(__name__)


class TrafficQAGradioApp:
    """Gradio application for Traffic QA System"""
    
    def __init__(self, use_case: AskQuestionUseCase):
        """
        Initialize Gradio app.
        
        Args:
            use_case: AskQuestionUseCase instance for processing queries
        """
        self.use_case = use_case
        logger.info("Initialized Gradio app")
    
    def process_message(
        self,
        message: str,
        history: List[Tuple[str, str]]
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Process a user message and update chat history.
        
        Args:
            message: User's question
            history: Chat history as list of (user_msg, bot_msg) tuples
        
        Returns:
            Tuple of (empty string for textbox, updated history)
        """
        if not message or not message.strip():
            return "", history
        
        try:
            # Process the query
            response = self.use_case.execute(message)
            
            # Format the response with citations
            answer = response.answer
            # if response.citation:
            #     answer += "\n\n📚 **Căn cứ pháp lý:**"
            #     answer += f"\n• {response.citation}"
            
            # Update history
            history.append((message, answer))
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            error_msg = f"Đã xảy ra lỗi khi xử lý câu hỏi: {str(e)}"
            history.append((message, error_msg))
        
        return "", history
    
    def clear_history(self) -> List:
        """Clear chat history"""
        return []
    
    def create_interface(self) -> gr.Blocks:
        """
        Create Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(
            title="Hệ thống Tra cứu Vi phạm Giao thông",
            theme=gr.themes.Soft()
        ) as demo:
            gr.Markdown(
                """
                # 🚦 Hệ thống Tra cứu Vi phạm Giao thông
                
                Hỏi đáp về các mức phạt, điều luật và hình phạt bổ sung liên quan đến vi phạm giao thông.
                
                **Ví dụ câu hỏi:**
                - Xe máy vượt đèn đỏ bị phạt bao nhiêu?
                - Ô tô quá tốc độ 20km/h ở nội thành Hà Nội bị phạt thế nào?
                - Điều luật nào quy định về không đội mũ bảo hiểm?
                """
            )
            
            chatbot = gr.Chatbot(
                label="Lịch sử trò chuyện",
                height=500,
                show_label=True,
                show_copy_button=False
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Câu hỏi của bạn",
                    placeholder="Nhập câu hỏi về vi phạm giao thông...",
                    lines=2,
                    scale=4
                )
                submit_btn = gr.Button("Gửi", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("Xóa lịch sử", variant="secondary")
            
            gr.Markdown(
                """
                ---
                💡 **Lưu ý:** Hệ thống sử dụng AI để phân tích câu hỏi và tra cứu thông tin. 
                Kết quả chỉ mang tính chất tham khảo.
                """
            )
            
            # Event handlers
            submit_btn.click(
                fn=self.process_message,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            
            msg.submit(
                fn=self.process_message,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            
            clear_btn.click(
                fn=self.clear_history,
                outputs=[chatbot]
            )
        
        return demo
    
    def launch(self, **kwargs):
        """
        Launch the Gradio app.
        
        Args:
            **kwargs: Additional arguments to pass to gr.Blocks.launch()
        """
        demo = self.create_interface()
        demo.launch(**kwargs)


def create_app() -> TrafficQAGradioApp:
    """
    Create and configure the Gradio app with dependency injection.
    
    Returns:
        Configured TrafficQAGradioApp instance
    """
    # Initialize container
    container = Container()
    container.wire(modules=[__name__])
    
    # Get use case from container
    use_case = container.ask_question_use_case()
    
    return TrafficQAGradioApp(use_case)


if __name__ == "__main__":
    # Create and launch app
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

from domain.models import QueryResponse
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from src.presentation.di_container import Container
from src.application.use_cases.ask_question_use_case import AskQuestionUseCase
from dependency_injector.wiring import inject, Provide

app = FastAPI()
container = Container()
app.container = container # Gắn container vào app

class QueryRequest(BaseModel):
    question: str

# API endpoint
@app.post("/ask", response_model=QueryResponse) # Dùng model từ Domain
@inject # Tự động tiêm dependency
def ask_endpoint(
    request: QueryRequest,
    use_case: AskQuestionUseCase = Depends(Provide[Container.ask_question_use_case])
):
    # Lớp Presentation chỉ việc gọi `execute` của use case
    response = use_case.execute(request.question)
    return response

# (Có thể thêm endpoint để chạy import_data)

if __name__ == "__main__":
    import uvicorn
    # Phải gọi wire để DI hoạt động
    container.wire(modules=[__name__]) 
    uvicorn.run(app, host="0.0.0.0", port=8000)
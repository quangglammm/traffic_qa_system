"""
Question Answering API endpoints.

This module defines the REST API endpoints for the traffic violation
question answering system.
"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from dependency_injector.wiring import inject, Provide
from src.domain.models import QueryResponse
from src.application.use_cases.ask_question_use_case import AskQuestionUseCase
from src.presentation.di_container import Container

router = APIRouter(tags=["QA"])


class QueryRequest(BaseModel):
    """Request model for question answering endpoint."""
    question: str = Field(..., description="Natural language question about traffic violations")


@router.post("/ask", response_model=QueryResponse, summary="Ask a question about traffic violations")
@inject
def ask_endpoint(
    request: QueryRequest,
    use_case: AskQuestionUseCase = Depends(Provide[Container.ask_question_use_case])
) -> QueryResponse:
    """
    Process a natural language question about traffic violations.

    This endpoint:
    1. Analyzes the query to extract intent and entities
    2. Performs semantic search to find similar violations
    3. Filters and ranks results using knowledge graph
    4. Retrieves detailed violation information
    5. Generates a natural language response with citations

    Args:
        request: Query request containing the user's question
        use_case: Injected AskQuestionUseCase instance

    Returns:
        QueryResponse containing the answer and citation information
    """
    return use_case.execute(request.question)


"""
Main application entry point for the Traffic QA System.

This module initializes the FastAPI application and wires dependency injection.
"""
from fastapi import FastAPI
from src.presentation.di_container import Container
from src.presentation.api.qa_endpoint import router

# Initialize FastAPI application
app = FastAPI(
    title="Traffic QA System",
    description="Question answering system for traffic violations",
    version="1.0.0"
)

# Initialize dependency injection container
container = Container()
app.container = container

# Wire dependency injection on application startup
@app.on_event("startup")
def startup_event():
    """Wire dependency injection for modules using @inject decorator."""
    container.wire(modules=[__name__, "src.presentation.api.qa_endpoint"])

@app.on_event("shutdown")
def shutdown_event():
    """Unwire dependency injection on application shutdown."""
    container.unwire()

# Register API routes
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
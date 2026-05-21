from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(title="Real-Time SLR MVP", version="0.1.0")
    app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
    app.include_router(router)
    return app


app = create_app()

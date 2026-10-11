"""One resident model, one generation at a time, with no per-request reload."""
import logging
from contextlib import asynccontextmanager
from copy import deepcopy
from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from bitext_support.data.customer_support import system_prompt
from bitext_support.evaluation.customer_support import INTENTS, parse_answer
from bitext_support.inference.predictor import Predictor, resolve_model, validate_generation
from bitext_support.inference.schemas import AnalyzeRequest, AnalyzeResponse, ChatRequest, ChatResponse, HealthResponse

logger = logging.getLogger(__name__)


def create_app(config, kind="base", training_run=None, system_prompt="", *, customer_support=False,
               predictor_factory=Predictor):
    config = deepcopy(config)
    validate_generation(config["generation"])
    resolve_model(config, kind, training_run)
    if not system_prompt.strip():
        raise ValueError("System prompt must contain non-whitespace text")
    if customer_support and system_prompt.strip() != system_prompt_for_customer_support():
        raise ValueError("Customer-support API requires the approved customer-support system prompt")
    generation_lock = Lock()

    @asynccontextmanager
    async def lifespan(app):
        try:
            app.state.predictor = predictor_factory(config, kind, training_run, system_prompt)
        except Exception as error:
            logger.exception("Model initialization failed")
            raise RuntimeError(f"Model initialization failed: {error}") from error
        try:
            yield
        finally:
            app.state.predictor = None

    app = FastAPI(title="Customer Support API" if customer_support else "UltraFeedback Inference API",
                  version="0.1.0", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    def health():
        predictor = app.state.predictor
        return {"status": "ready", "kind": kind, "model": predictor.model_config["name_or_path"],
                "max_input_tokens": config["generation"]["max_input_tokens"],
                "max_new_tokens": config["generation"]["max_new_tokens"]}

    @app.post("/chat", response_model=ChatResponse,
              responses={422: {"description": "Invalid input or token limit exceeded"},
                         503: {"description": "Model is generating another answer"},
                         500: {"description": "Generation failed"}})
    def chat(request: ChatRequest):
        if request.max_new_tokens and request.max_new_tokens > config["generation"]["max_new_tokens"]:
            raise HTTPException(422, detail="max_new_tokens exceeds the server limit")
        if not generation_lock.acquire(blocking=False):
            raise HTTPException(503, detail="Model is busy; retry after the current request completes",
                                headers={"Retry-After": "2"})
        request_id = uuid4().hex
        try:
            options = request.model_dump(exclude_none=True, exclude={"prompt"})
            result = app.state.predictor.generate(request.prompt, **options)
            return {"request_id": request_id, "kind": kind, "answer": result["answer"],
                    "input_tokens": result["input_tokens"], "output_tokens": result["output_tokens"],
                    "generation_seconds": result["timing_seconds"]["generate"],
                    "finish_reason": "length" if result["hit_token_limit"] and not result["ended_with_eos"] else "stop"}
        except ValueError as error:
            raise HTTPException(422, detail=str(error)) from error
        except Exception as error:
            logger.exception("Generation failed for request %s", request_id)
            raise HTTPException(500, detail={"message": "Generation failed; see server logs",
                                             "request_id": request_id}) from error
        finally:
            generation_lock.release()

    if customer_support:
        @app.post("/analyze", response_model=AnalyzeResponse,
                  responses={422: {"description": "Invalid input or token limit exceeded"},
                             502: {"description": "Model returned invalid or ungrounded JSON"},
                             503: {"description": "Model is busy"}})
        def analyze(request: AnalyzeRequest):
            if not generation_lock.acquire(blocking=False):
                raise HTTPException(503, detail="Model is busy; retry after the current request completes",
                                    headers={"Retry-After": "2"})
            request_id = uuid4().hex
            try:
                result = app.state.predictor.generate(request.message, max_new_tokens=96, do_sample=False)
                parsed, error = parse_answer(result["answer"])
                if error or (parsed["order_id"] is not None and
                             (not parsed["order_id"] or parsed["order_id"] not in request.message)):
                    raise HTTPException(502, detail={"message": "Model returned invalid or ungrounded JSON",
                                                     "request_id": request_id})
                return {"request_id": request_id, "intent": parsed["intent"],
                        "order_id": parsed["order_id"],
                        "generation_seconds": result["timing_seconds"]["generate"]}
            except HTTPException:
                raise
            except ValueError as error:
                raise HTTPException(422, detail=str(error)) from error
            except Exception as error:
                logger.exception("Customer analysis failed for request %s", request_id)
                raise HTTPException(500, detail={"message": "Generation failed; see server logs",
                                                 "request_id": request_id}) from error
            finally:
                generation_lock.release()

    return app


def system_prompt_for_customer_support():
    return system_prompt(INTENTS)

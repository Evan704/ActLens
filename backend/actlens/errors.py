"""Domain errors. They carry the HTTP status the API reports them with, but know nothing about FastAPI, so the model
manager can be driven from a notebook or a test without a web framework in the loop."""
from __future__ import annotations


class ActLensError(Exception):
    status_code = 500

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


class BadRequest(ActLensError):
    status_code = 400


class NotFound(ActLensError):
    status_code = 404


class Conflict(ActLensError):
    status_code = 409

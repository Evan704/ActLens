"""Optional access-token gate for a server that is reachable from outside (e.g. a tunnel from Colab)."""
from __future__ import annotations

import hmac

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, RedirectResponse

TOKEN_COOKIE = "actlens_token"


def install_token_auth(app: FastAPI, token: str) -> None:
    """Open `/?token=...` once: the token is stored in an HttpOnly cookie and the URL is cleaned up.
    Scripts can send `Authorization: Bearer <token>` instead."""

    def valid(candidate: str | None) -> bool:
        return bool(candidate) and hmac.compare_digest(candidate.encode(), token.encode())

    @app.middleware("http")
    async def require_token(request: Request, call_next):
        bearer = request.headers.get("authorization", "")
        if valid(request.cookies.get(TOKEN_COOKIE)) or valid(bearer.removeprefix("Bearer ").strip()):
            return await call_next(request)
        if request.method == "GET" and valid(request.query_params.get("token")):
            url = request.url.remove_query_params("token")
            resp = RedirectResponse(url.path + (f"?{url.query}" if url.query else ""), status_code=303)
            # Lax, not Strict: the link is usually opened from another site (Colab), and a Strict cookie is
            # not sent on the redirect that follows a cross-site navigation.
            https = "https" in (request.url.scheme, request.headers.get("x-forwarded-proto", ""))
            resp.set_cookie(TOKEN_COOKIE, token, httponly=True, samesite="lax", secure=https)
            return resp
        return PlainTextResponse("ActLens: missing or invalid access token. Open the full link printed "
                                 "by the server (it ends in ?token=...).", status_code=401)

"""Cache only versioned public assets; compress only non-sensitive UI resources."""

from starlette.datastructures import QueryParams
from starlette.middleware.gzip import GZipMiddleware
from starlette.staticfiles import StaticFiles

from gmxbuilder import VERSION


class VersionedStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        version = QueryParams(scope.get("query_string", b"").decode()).get("v")
        versioned = version == VERSION or path.startswith("vendor/")
        if response.status_code in {200, 304}:
            response.headers["Cache-Control"] = (
                "private, max-age=31536000, immutable" if versioned else "private, no-cache"
            )
        return response


class CatalogCompression:
    def __init__(self, app):
        self.app = app
        self.compressed = GZipMiddleware(app, minimum_size=512, compresslevel=6)

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        compress = (
            path.startswith("/static/") and path.endswith((".js", ".css", ".svg"))
        ) or path in {
            "/api/options",
            "/api/task-types",
            "/api/lipid-library-list",
            "/api/coarse-grained/capabilities",
        }
        await (self.compressed if compress else self.app)(scope, receive, send)

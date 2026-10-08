"""Test-only early JavaScript error collection, served under the normal CSP."""

SCRIPT_PATH = "/__test_browser_errors.js"
SCRIPT = b"""
window.__browserErrors = [];
window.__browserRequests = [];
const originalFetch = window.fetch.bind(window);
window.fetch = async (...args) => {
  const record = {url: String(args[0]), started: performance.now(), completed: false};
  window.__browserRequests.push(record);
  try {
    const response = await originalFetch(...args);
    record.status = response.status;
    return response;
  } catch(error) {
    record.error = String(error);
    throw error;
  } finally {
    record.completed = true;
    record.duration = performance.now() - record.started;
  }
};
window.addEventListener('error', event => {
  window.__browserErrors.push(String(event.message || 'resource error') + ' ' +
    String(event.filename || event.target?.src || ''));
}, true);
window.addEventListener('unhandledrejection', event => {
  window.__browserErrors.push('Unhandled rejection: ' + String(event.reason));
});
const originalConsoleError = console.error.bind(console);
console.error = (...values) => {
  window.__browserErrors.push('console.error: ' + values.map(String).join(' '));
  originalConsoleError(...values);
};
"""


class BrowserInstrumentation:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        if scope["path"] == SCRIPT_PATH:
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/javascript")],
                }
            )
            return await send({"type": "http.response.body", "body": SCRIPT})
        if scope["path"] != "/":
            return await self.app(scope, receive, send)
        messages = []

        async def collect(message):
            messages.append(message)

        await self.app(scope, receive, collect)
        start = messages[0]
        body = b"".join(m.get("body", b"") for m in messages[1:])
        body = body.replace(
            b"<head>", b'<head><script src="' + SCRIPT_PATH.encode() + b'"></script>', 1
        )
        start["headers"] = [(k, v) for k, v in start["headers"] if k.lower() != b"content-length"]
        start["headers"].append((b"content-length", str(len(body)).encode()))
        await send(start)
        await send({"type": "http.response.body", "body": body})

"""Focused helpers extracted from the Web server entrypoint.

The public FastAPI application remains in :mod:`gmxbuilder.web.server`.  This
subpackage keeps security, resource-boundary, and scientific-preview helpers
together beside ``server.py`` instead of adding more unrelated modules directly
under :mod:`gmxbuilder.web`.
"""

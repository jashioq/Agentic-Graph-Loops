"""The ticket workflow: decompose a request into tickets and drive them to merged.

Layer: workflows. Composes the core modules; nothing in `core` may import this.

No re-exports. The modules here are imported by their own names — `from
agl.workflows.tickets.models import Ticket` — so there is one import path per
name rather than two that can disagree about what is public.
"""

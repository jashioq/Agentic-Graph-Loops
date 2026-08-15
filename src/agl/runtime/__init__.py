"""Reusable functionality a workflow composes: pure, or built from core.

Layer: runtime. Sits between `workflows` and `core`. Nothing here reaches
outside AGL on its own terms — a runtime module is either pure or does its
outside work through a core connector it was handed. Workflows import from this
package; core never does.
"""

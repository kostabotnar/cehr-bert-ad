"""Shared, testable building blocks for the AD CEHR-BERT pipeline.

Every step script (``scripts/ad_pipeline/NN_*.py``) is a thin entrypoint that wires
command-line arguments to functions in this package and writes a :class:`StepReport`.

Design rules:
  * No heavy dependency (``datasets``, ``transformers``, ``torch``, ``cehrbert``,
    ``cehrbert_data``) is imported at module import time. They are imported lazily
    inside the functions that need them so that pure validation steps run in a
    minimal environment and the test suite can import everything.
"""

from .reporting import Check, StepReport

__all__ = ["Check", "StepReport"]

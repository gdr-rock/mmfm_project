"""Import smoke tests for the scaffold package."""

from src.data import datasets, segmentation, transforms
from src.eval import metrics, retrieval, vpa
from src.models import bridges, critic, jepa_wrapper, planner_llm
from src.planning import scoring, selection


def test_core_imports() -> None:
    assert datasets is not None
    assert segmentation is not None
    assert transforms is not None
    assert bridges is not None
    assert critic is not None
    assert jepa_wrapper is not None
    assert planner_llm is not None
    assert scoring is not None
    assert selection is not None
    assert metrics is not None
    assert retrieval is not None
    assert vpa is not None

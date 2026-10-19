import pytest

from urdu_sentiment.data import clean_table, split_dataset
from urdu_sentiment.synthetic import make_synthetic


@pytest.fixture(scope="session")
def synthetic_frame():
    """SYNTHETIC template sentences. Used for pipeline tests only."""
    frame, _ = clean_table(make_synthetic(240, seed=1))
    return frame


@pytest.fixture(scope="session")
def splits(synthetic_frame):
    return split_dataset(synthetic_frame, val_size=0.15, test_size=0.25, seed=3)

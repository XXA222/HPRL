from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from freqtrade.persistence.hedge_models import create_hedge_tables


@pytest.fixture()
def engine():
    value = create_engine("sqlite+pysqlite:///:memory:")
    create_hedge_tables(value)
    try:
        yield value
    finally:
        value.dispose()


@pytest.fixture()
def session_factory(engine):
    return sessionmaker(bind=engine, expire_on_commit=False)

"""Market-data store. Prices are total-return adjusted at ingest, so the
backtester needs no separate dividend bookkeeping. Every row carries the
date it describes; research code must only ever query `date <= as_of`.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import Column, Date, Float, String, create_engine
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class DailyBar(Base):
    __tablename__ = "daily_bars"
    ticker = Column(String, primary_key=True)
    date = Column(Date, primary_key=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)


def get_engine(db_path: str):
    return create_engine(f"sqlite:///{db_path}")


def init_db(engine) -> None:
    Base.metadata.create_all(engine)

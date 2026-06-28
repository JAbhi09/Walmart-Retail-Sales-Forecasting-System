"""
Input validation helpers for agent entry points.
"""
import pandas as pd
from typing import List, Optional


def validate_dataframe(
    df,
    required_columns: List[str],
    name: str,
    date_columns: Optional[List[str]] = None,
) -> None:
    """
    Validate a DataFrame at an agent entry point.

    Raises ValueError with a clear message if:
    - df is not a pandas DataFrame (e.g. None, dict, list)
    - Any required columns are absent
    - DataFrame is empty
    - Any specified date column is entirely null / unparseable
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            f"'{name}' must be a pandas DataFrame, got {type(df).__name__}."
        )

    if df.empty:
        raise ValueError(f"'{name}' DataFrame is empty.")

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"'{name}' is missing required column(s): {missing}. "
            f"Present columns: {list(df.columns)}."
        )

    if date_columns:
        for col in date_columns:
            if col not in df.columns:
                continue
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.isna().all():
                raise ValueError(
                    f"'{name}' date column '{col}' is entirely null or unparseable as dates."
                )

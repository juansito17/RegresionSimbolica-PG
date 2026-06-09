"""Input parsing helpers shared by AlphaSymbolic UI tabs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ParsedData:
    x: np.ndarray
    y: np.ndarray
    error: Optional[str] = None


def _parse_numeric_line(line: str) -> list[float]:
    return [float(v) for v in line.replace(",", " ").split()]


def parse_input_data(x_str: str, y_str: str) -> ParsedData:
    """Parse 1D or multivariable X/Y strings from the UI."""
    try:
        x_text = (x_str or "").strip()
        y_text = (y_str or "").strip()
        if not x_text or not y_text:
            return ParsedData(np.array([]), np.array([]), "Ingresa valores para X e Y.")

        multivar = "\n" in x_text or ";" in x_text
        if multivar:
            rows = [row.strip() for row in x_text.replace(";", "\n").splitlines() if row.strip()]
            x_rows = [_parse_numeric_line(row) for row in rows]
            if len({len(row) for row in x_rows}) != 1:
                return ParsedData(np.array([]), np.array([]), "Todas las filas de X deben tener el mismo número de variables.")
            x = np.asarray(x_rows, dtype=np.float64)
        else:
            x = np.asarray(_parse_numeric_line(x_text), dtype=np.float64)

        y = np.asarray(_parse_numeric_line(y_text.replace(";", " ")), dtype=np.float64)

        if len(x) != len(y):
            return ParsedData(x, y, f"Cantidad de muestras X ({len(x)}) != Y ({len(y)}).")
        if len(y) == 0:
            return ParsedData(x, y, "Ingresa al menos una muestra.")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            return ParsedData(x, y, "Los datos deben ser números finitos.")
        return ParsedData(x, y, None)
    except Exception as exc:
        return ParsedData(np.array([]), np.array([]), f"Error parseando datos: {exc}")


def load_csv_to_strings(file_obj) -> Tuple[Optional[str], Optional[str]]:
    """Load CSV/TXT file where the last column is Y and previous columns are X."""
    if file_obj is None:
        return None, None
    try:
        try:
            df = pd.read_csv(file_obj.name, sep=None, engine="python", header=None)
        except Exception:
            df = pd.read_csv(file_obj.name, header=None)

        if df.shape[1] < 2:
            return None, "Error: el archivo debe tener al menos dos columnas (X..., Y)."

        numeric_df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all").dropna(axis=1, how="all")
        if numeric_df.empty or numeric_df.shape[1] < 2:
            return None, "Error: el archivo debe contener columnas numéricas para X e Y."

        first_row_has_text = numeric_df.iloc[0].isna().any()
        remaining_rows_are_numeric = len(numeric_df) > 1 and numeric_df.iloc[1:].notna().all().all()
        if first_row_has_text and remaining_rows_are_numeric:
            numeric_df = numeric_df.iloc[1:]

        if numeric_df.isna().any().any():
            return None, "Error: todas las columnas de datos deben ser numéricas. Si usas encabezados, colócalos solo en la primera fila."

        x_values = numeric_df.iloc[:, :-1].to_numpy(dtype=float)
        y_values = numeric_df.iloc[:, -1].to_numpy(dtype=float)
        if x_values.shape[1] == 1:
            x_str = ", ".join(f"{v:g}" for v in x_values.flatten())
        else:
            x_str = "\n".join(" ".join(f"{v:g}" for v in row) for row in x_values)
        y_str = ", ".join(f"{v:g}" for v in y_values.flatten())
        return x_str, y_str
    except Exception as exc:
        return None, f"Error leyendo CSV: {exc}"

import numpy as np

from AlphaSymbolic.ui.data_io import load_csv_to_strings, parse_input_data
from AlphaSymbolic.ui.app_search import generate_example


def test_parse_1d_comma_and_space_separators():
    parsed = parse_input_data("1, 2 3,4", "2, 4 6,8")

    assert parsed.error is None
    assert parsed.x.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert parsed.y.tolist() == [2.0, 4.0, 6.0, 8.0]


def test_parse_multivariable_rows():
    parsed = parse_input_data("1 2\n3 4\n5 6", "3, 7, 11")

    assert parsed.error is None
    assert parsed.x.shape == (3, 2)
    assert parsed.y.shape == (3,)


def test_parse_rejects_mismatched_lengths():
    parsed = parse_input_data("1, 2, 3", "1, 2")

    assert parsed.error
    assert "Cantidad de muestras" in parsed.error


def test_generate_example_lineal_is_parseable():
    x_str, y_str = generate_example("lineal")
    parsed = parse_input_data(x_str, y_str)

    assert parsed.error is None
    assert np.allclose(parsed.y, 2 * parsed.x + 3)


def test_generate_example_trig_matches_displayed_x_values():
    x_str, y_str = generate_example("trig")
    parsed = parse_input_data(x_str, y_str)

    assert parsed.error is None
    assert np.allclose(parsed.y, np.sin(parsed.x), atol=1e-6)


def test_load_csv_without_header_keeps_first_row(tmp_path):
    csv_path = tmp_path / "linear.csv"
    csv_path.write_text("1,5\n2,7\n3,9\n", encoding="utf-8")

    x_str, y_str = load_csv_to_strings(type("Upload", (), {"name": str(csv_path)})())

    assert x_str == "1, 2, 3"
    assert y_str == "5, 7, 9"


def test_load_csv_with_header_uses_last_column_as_y(tmp_path):
    csv_path = tmp_path / "linear_header.csv"
    csv_path.write_text("x,y\n1,5\n2,7\n3,9\n", encoding="utf-8")

    x_str, y_str = load_csv_to_strings(type("Upload", (), {"name": str(csv_path)})())

    assert x_str == "1, 2, 3"
    assert y_str == "5, 7, 9"


def test_load_csv_multivariable_formats_x_by_rows(tmp_path):
    csv_path = tmp_path / "multi.csv"
    csv_path.write_text("x0,x1,y\n1,2,3\n4,5,9\n", encoding="utf-8")

    x_str, y_str = load_csv_to_strings(type("Upload", (), {"name": str(csv_path)})())

    assert x_str == "1 2\n4 5"
    assert y_str == "3, 9"

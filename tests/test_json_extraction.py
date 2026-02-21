"""Tests for the improved JSON extraction logic."""

from services.planner.planner import VisualizationPlanner


def _planner():
    """Create a planner instance without calling __init__."""
    p = VisualizationPlanner.__new__(VisualizationPlanner)
    return p


def test_deeply_nested_json_with_surrounding_text():
    p = _planner()
    response = (
        'Here is the visualization spec:\n'
        '{\n'
        '  "chart_type": "bar",\n'
        '  "x": "Country",\n'
        '  "y": "Price",\n'
        '  "aggregate": {"func": "sum", "group_by": ["Country"]},\n'
        '  "filters": [],\n'
        '  "transformations": [],\n'
        '  "options": {"title": "Top 10 Countries by Sales", "sort": "descending", "limit": 10},\n'
        '  "explain": "Bar chart of top 10 countries by total sales"\n'
        '}'
    )
    result = p._extract_json_from_response(response)
    assert result["chart_type"] == "bar"
    assert result["options"]["limit"] == 10
    print("Test 1 (nested JSON with surrounding text): PASS")


def test_markdown_code_block():
    p = _planner()
    response = (
        'Sure! Here is your chart spec:\n'
        '```json\n'
        '{"chart_type": "line", "x": "Date", "y": "Revenue", "aggregate": null, '
        '"filters": [], "transformations": [], "options": {"title": "Revenue over time"}, '
        '"explain": "Line chart"}\n'
        '```\n'
        'Hope this helps!'
    )
    result = p._extract_json_from_response(response)
    assert result["chart_type"] == "line"
    print("Test 2 (markdown code block): PASS")


def test_clean_json():
    p = _planner()
    response = '{"chart_type": "scatter", "x": "A", "y": "B", "filters": [], "transformations": [], "options": {}, "explain": "test"}'
    result = p._extract_json_from_response(response)
    assert result["chart_type"] == "scatter"
    print("Test 3 (clean JSON): PASS")


def test_trailing_comma_cleanup():
    p = _planner()
    response = '{"chart_type": "pie", "x": "Category", "y": "Amount", "filters": [], "transformations": [], "options": {"title": "Distribution",}, "explain": "test"}'
    result = p._extract_json_from_response(response)
    assert result["chart_type"] == "pie"
    print("Test 4 (trailing comma cleanup): PASS")


def test_triple_nested_objects():
    p = _planner()
    response = (
        'I analyzed the data. Here:\n\n'
        '{"chart_type": "bar", "x": "Country", "y": "Total", '
        '"aggregate": {"func": "sum", "group_by": ["Country"]}, '
        '"filters": [{"column": "Year", "op": ">=", "value": 2020}], '
        '"transformations": [], '
        '"options": {"title": "Sales by Country", "height": 600, "width": 900, "sort": "descending", "limit": 10}, '
        '"explain": "Top 10 countries"}\n\n'
        'This shows the top 10 countries.'
    )
    result = p._extract_json_from_response(response)
    assert result["chart_type"] == "bar"
    assert result["aggregate"]["func"] == "sum"
    assert result["filters"][0]["op"] == ">="
    assert result["options"]["limit"] == 10
    print("Test 5 (triple nested with filters + aggregate + options): PASS")


if __name__ == "__main__":
    test_deeply_nested_json_with_surrounding_text()
    test_markdown_code_block()
    test_clean_json()
    test_trailing_comma_cleanup()
    test_triple_nested_objects()
    print("\nAll JSON extraction tests passed!")

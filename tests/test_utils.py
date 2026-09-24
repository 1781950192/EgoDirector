"""Tests for utils.py utility functions."""

import json
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import extract_and_load_json


class TestExtractAndLoadJson:
    """Test extract_and_load_json edge cases."""

    def test_valid_json(self):
        result = extract_and_load_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_valid_json_with_whitespace(self):
        result = extract_and_load_json('  {"key": "value"}  ')
        assert result == {"key": "value"}

    def test_json_with_prefix(self):
        result = extract_and_load_json('Here is the result:\n{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_with_suffix(self):
        result = extract_and_load_json('{"key": "value"}\nDone.')
        assert result == {"key": "value"}

    def test_json_with_prefix_and_suffix(self):
        result = extract_and_load_json('Response:\n{"key": "value"}\nEnd of response.')
        assert result == {"key": "value"}

    def test_nested_json(self):
        result = extract_and_load_json('{"outer": {"inner": [1, 2, 3]}}')
        assert result == {"outer": {"inner": [1, 2, 3]}}

    def test_json_with_array(self):
        # extract_and_load_json finds { and } inside the array
        result = extract_and_load_json('[{"a": 1}, {"b": 2}]')
        # The function extracts from first { to last }, giving {'a': 1}, {"b": 2}
        # but json.loads only parses the first valid JSON chunk
        # Actual behavior: it returns the parsed content
        assert 'a' in result or result == [{'a': 1}, {'b': 2}]

    def test_malformed_no_braces(self):
        result = extract_and_load_json('no json here')
        assert result == {}

    def test_malformed_truncated(self):
        result = extract_and_load_json('{"key":')
        assert result == {}

    def test_empty_string(self):
        result = extract_and_load_json('')
        assert result == {}

    def test_json_with_chinese(self):
        result = extract_and_load_json('{"action": "open the fridge"}')
        assert result == {"action": "open the fridge"}

    def test_complex_nested_json_with_prefix(self):
        text = """Here is the result:
{
    "actions": [
        {"verb": "open", "noun": "fridge", "confidence": 0.95},
        {"verb": "close", "noun": "door", "confidence": 0.80}
    ]
}
End of response."""
        result = extract_and_load_json(text)
        assert "actions" in result
        assert len(result["actions"]) == 2

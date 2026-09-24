"""Tests for context_manager.py functions."""

import pytest
import json
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_manager import extract_noun_probabilities, update_json_file, update_verb_noun_json


class TestExtractNounProbabilities:
    """Test noun-to-verb probability extraction."""

    def test_basic_extraction(self):
        """Test extracting probabilities for known nouns."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                'fridge': {'open': 0.8, 'close': 0.2},
                'door': {'open': 0.6, 'close': 0.4},
            }, f)
            f.flush()
            result = extract_noun_probabilities(['fridge', 'door'], f.name)

        assert result == {
            'fridge': {'open': 0.8, 'close': 0.2},
            'door': {'open': 0.6, 'close': 0.4},
        }

    def test_unknown_noun(self):
        """Test that unknown nouns return empty dict."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({'fridge': {'open': 0.8}}, f)
            f.flush()
            result = extract_noun_probabilities(['fridge', 'unknown_noun'], f.name)

        assert result['fridge'] == {'open': 0.8}
        assert result['unknown_noun'] == {}

    def test_empty_noun_list(self):
        """Test with empty input list."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({'fridge': {'open': 0.8}}, f)
            f.flush()
            result = extract_noun_probabilities([], f.name)

        assert result == {}


class TestUpdateJsonFile:
    """Test update_json_file (list format)."""

    def test_create_new_file(self):
        """Test creating a new JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filename = f.name

        try:
            new_data = [
                {"key": "fridge", "generated_text": "A large appliance for storing food", "frequency": 1},
            ]
            update_json_file(filename, new_data)

            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            assert len(data) == 1
            assert data[0]["key"] == "fridge"
        finally:
            os.unlink(filename)

    def test_append_existing_file(self):
        """Test appending to an existing JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filename = f.name

        try:
            # Create initial file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump([{"key": "fridge", "generated_text": "existing", "frequency": 1}], f)

            new_data = [{"key": "door", "generated_text": "new entry", "frequency": 1}]
            update_json_file(filename, new_data)

            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            assert len(data) == 2
            assert data[1]["key"] == "door"
        finally:
            os.unlink(filename)

    def test_invalid_json_recreates(self):
        """Test that invalid JSON is recreated."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filename = f.name

        try:
            with open(filename, 'w') as f:
                f.write("not valid json")

            new_data = [{"key": "test", "generated_text": "recovered", "frequency": 1}]
            update_json_file(filename, new_data)

            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            assert len(data) == 1
            assert data[0]["key"] == "test"
        finally:
            os.unlink(filename)


class TestUpdateVerbNounJson:
    """Test update_verb_noun_json (dict format)."""

    def test_create_new_dict_file(self):
        """Test creating a new dict-format JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filename = f.name

        try:
            new_entries = [
                {"fridge": ["open", "close"]},
                {"door": ["push", "pull"]},
            ]
            update_verb_noun_json(filename, new_entries)

            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            assert data["fridge"] == ["open", "close"]
            assert data["door"] == ["push", "pull"]
        finally:
            os.unlink(filename)

    def test_update_existing_dict(self):
        """Test updating existing entries."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filename = f.name

        try:
            # Create initial file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({"fridge": ["open"]}, f)

            new_entries = [
                {"fridge": ["open", "close", "check"]},
                {"door": ["push"]},
            ]
            update_verb_noun_json(filename, new_entries)

            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # "fridge" should be overwritten with new verbs
            assert data["fridge"] == ["open", "close", "check"]
            assert data["door"] == ["push"]
        finally:
            os.unlink(filename)

    def test_non_dict_recreates(self):
        """Test that non-dict JSON is recreated as dict."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filename = f.name

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump([{"wrong": "format"}], f)  # list, not dict

            new_entries = [{"fridge": ["open"]}]
            update_verb_noun_json(filename, new_entries)

            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            assert isinstance(data, dict)
            assert data["fridge"] == ["open"]
        finally:
            os.unlink(filename)

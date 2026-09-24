"""Tests for pipeline.py frame selection and result writing."""

import pytest
import sys
import os
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import select_sub_frames, ResultWriter


class TestSelectSubFrames:
    """Test uniform frame sub-selection."""

    def test_select_all(self):
        """If target >= available, return all."""
        indices = list(range(32))
        result = select_sub_frames(indices, 32)
        assert result == indices

        result = select_sub_frames(indices, 50)
        assert result == indices

    def test_32_to_16(self):
        """32 -> 16 should pick evenly spaced frames (np.linspace rounding)."""
        indices = list(range(32))
        result = select_sub_frames(indices, 16)
        assert len(result) == 16
        # np.linspace(0, 31, 16) rounds to these actual values
        expected = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 31]
        assert result == expected

    def test_32_to_8(self):
        """32 -> 8 should pick evenly spaced frames."""
        indices = list(range(32))
        result = select_sub_frames(indices, 8)
        assert len(result) == 8
        expected = [0, 4, 8, 13, 17, 22, 26, 31]
        assert result == expected

    def test_32_to_4(self):
        """32 -> 4 should pick quarters."""
        indices = list(range(32))
        result = select_sub_frames(indices, 4)
        assert len(result) == 4
        expected = [0, 10, 20, 31]
        assert result == expected

    def test_16_to_4(self):
        """16 -> 4 should pick quarters."""
        indices = list(range(16))
        result = select_sub_frames(indices, 4)
        assert len(result) == 4
        expected = [0, 5, 10, 15]
        assert result == expected

    def test_32_to_1(self):
        """32 -> 1 should pick first frame."""
        indices = list(range(32))
        result = select_sub_frames(indices, 1)
        assert len(result) == 1
        assert result[0] == 0

    def test_preserves_order(self):
        """Selected frames should be in ascending order."""
        indices = list(range(64))
        result = select_sub_frames(indices, 8)
        assert result == sorted(result)

    def test_non_sequential_indices(self):
        """Should work with non-sequential indices (e.g., video frame numbers)."""
        # Simulate frame indices from a 100-frame video
        indices = np.linspace(0, 99, num=32, dtype=int).tolist()
        result = select_sub_frames(indices, 8)
        assert len(result) == 8
        # First and last should be preserved
        assert result[0] == 0
        assert result[-1] == 99


class TestResultWriter:
    """Test CSV result writing."""

    def test_write_result_basic(self):
        """Test basic result writing format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            output_path = f.name

        try:
            ResultWriter.init_output_file(output_path)

            data = {'narration_id': 'test_001'}
            predicted = {
                'actions': [
                    {'verb': 'open', 'noun': 'fridge', 'action': 'open_fridge', 'confidence': 0.9},
                    {'verb': 'close', 'noun': 'door', 'action': 'close_door', 'confidence': 0.7},
                ]
            }
            selected_noun_keys = ['fridge', 'door']

            ResultWriter.write_result(output_path, data, predicted, selected_noun_keys, 'test_001')

            with open(output_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            assert len(lines) == 2  # header + 1 data line
            assert lines[0] == ResultWriter.HEADER
            parts = lines[1].strip().split(',')
            assert parts[0] == 'test_001'
            assert 'open;close' in parts[1]  # verbs (sorted by confidence desc)
            assert 'fridge;door' in parts[2]  # nouns
        finally:
            os.unlink(output_path)

    def test_write_result_empty_actions(self):
        """Test writing when no actions are found."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            output_path = f.name

        try:
            ResultWriter.init_output_file(output_path)

            data = {'narration_id': 'test_002'}
            predicted = {'actions': []}
            selected_noun_keys = []

            ResultWriter.write_result(output_path, data, predicted, selected_noun_keys, 'test_002')

            with open(output_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            parts = lines[1].strip().split(',')
            assert parts[1] == 'no verb'
            assert parts[2] == 'no noun'
            # Empty list -> join gives empty string
            assert parts[5] == ''
        finally:
            os.unlink(output_path)

    def test_load_processed_ids(self):
        """Test loading already-processed IDs."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            output_path = f.name

        try:
            ResultWriter.init_output_file(output_path)

            # Write some fake results
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write("id_001,x,x,x,x,x\n")
                f.write("id_002,x,x,x,x,x\n")
                f.write("id_003,x,x,x,x,x\n")

            processed = ResultWriter.load_processed_ids(output_path)
            assert processed == {'id_001', 'id_002', 'id_003'}
        finally:
            os.unlink(output_path)

    def test_load_processed_ids_empty_file(self):
        """Test loading from non-existent file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            output_path = f.name

        try:
            processed = ResultWriter.load_processed_ids(output_path)
            assert processed == set()
        finally:
            os.unlink(output_path)

    def test_init_output_file_existing_with_header(self):
        """Test init doesn't overwrite existing file with content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            output_path = f.name

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("existing content\n")

            ResultWriter.init_output_file(output_path)

            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()

            assert 'existing content' in content
        finally:
            os.unlink(output_path)

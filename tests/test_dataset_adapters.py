"""Tests for dataset_adapters.py label parsing."""

import pytest
import os
import tempfile
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_adapters import GTEAAdapter, EGTEAAdapter


class TestGTEAAdapter:
    """Test GTEA label file parsing."""

    def test_parse_verb_noun_only(self):
        """Only verb+noun format should be kept."""
        with tempfile.TemporaryDirectory() as tmpdir:
            label_file = os.path.join(tmpdir, 'video1.txt')
            with open(label_file, 'w', encoding='utf-8') as f:
                f.write("<take><bottle> (100-200)\n")
                f.write("<put><cup> (250-350)\n")

            adapter = GTEAAdapter(tmpdir, '/fake/videos')
            data = adapter.load_data()

            assert len(data) == 2
            assert data[0]['verb'] == 'take'
            assert data[0]['noun'] == 'bottle'
            assert data[0]['action'] == 'take_bottle'
            assert data[0]['start_frame'] == 100
            assert data[0]['stop_frame'] == 200

    def test_skip_noun_only(self):
        """Single <noun> annotations should be skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            label_file = os.path.join(tmpdir, 'video1.txt')
            with open(label_file, 'w', encoding='utf-8') as f:
                f.write("<bottle> (100-200)\n")        # noun only — should be skipped
                f.write("<take><bottle> (250-350)\n")  # verb+noun — should be kept

            adapter = GTEAAdapter(tmpdir, '/fake/videos')
            data = adapter.load_data()

            assert len(data) == 1
            assert data[0]['verb'] == 'take'

    def test_narration_id_format(self):
        """narration_id should be video_start_stop."""
        with tempfile.TemporaryDirectory() as tmpdir:
            label_file = os.path.join(tmpdir, 'video1.txt')
            with open(label_file, 'w', encoding='utf-8') as f:
                f.write("<take><bottle> (100-200)\n")

            adapter = GTEAAdapter(tmpdir, '/fake/videos')
            data = adapter.load_data()

            nid = GTEAAdapter.make_narration_id(data[0])
            assert nid == 'video1_100_200'

    def test_video_path_resolution(self):
        """Video path should be videos_base/video_name.mp4."""
        adapter = GTEAAdapter('/fake/labels', '/fake/videos')
        data = {'video_id': 'test_video'}

        # File doesn't exist, so should return None
        assert adapter.get_video_path(data) is None

    def test_multiple_files(self):
        """Should parse all .txt files in the labels directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                label_file = os.path.join(tmpdir, f'video{i}.txt')
                with open(label_file, 'w', encoding='utf-8') as f:
                    f.write(f"<take><bottle> ({i*100}-{i*100+100})\n")

            adapter = GTEAAdapter(tmpdir, '/fake/videos')
            data = adapter.load_data()

            assert len(data) == 3


class TestEGTEAAdapter:
    """Test EGTEA label file parsing."""

    def test_parse_split_file(self):
        """Each line should be video_id + action_label."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("A01_V01-P01-01 cut\n")
            f.write("A01_V01-P01-02 pour\n")
            f.write("A01_V01-P01-03 stir\n")
            f.flush()
            label_path = f.name

        try:
            adapter = EGTEAAdapter(label_path, '/fake/videos')
            data = adapter.load_data()

            assert len(data) == 3
            assert data[0]['video_id'] == 'A01_V01-P01-01'
            assert data[0]['action_label'] == 'cut'
            assert data[0]['narration_id'] == 'A01_V01-P01-01'
        finally:
            os.unlink(label_path)

    def test_empty_lines_skipped(self):
        """Empty lines should be ignored."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("A01_V01-P01-01 cut\n")
            f.write("\n")
            f.write("   \n")
            f.write("A01_V01-P01-02 pour\n")
            f.flush()
            label_path = f.name

        try:
            adapter = EGTEAAdapter(label_path, '/fake/videos')
            data = adapter.load_data()

            assert len(data) == 2
        finally:
            os.unlink(label_path)

    def test_missing_action_label(self):
        """Line without action_label should default to 'unknown'."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("A01_V01-P01-01\n")
            f.flush()
            label_path = f.name

        try:
            adapter = EGTEAAdapter(label_path, '/fake/videos')
            data = adapter.load_data()

            assert data[0]['action_label'] == 'unknown'
        finally:
            os.unlink(label_path)

    def test_video_path_subdir(self):
        """Video path should use first 3 dash-parts as subdir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # video_id 'A01_V01-P01-01' splits on '-' into ['A01_V01', 'P01', '01']
            # subdir = 'A01_V01-P01-01', path = tmpdir/A01_V01-P01-01/A01_V01-P01-01.mp4
            subdir = os.path.join(tmpdir, 'A01_V01-P01-01')
            os.makedirs(subdir)
            video_path = os.path.join(subdir, 'A01_V01-P01-01.mp4')
            with open(video_path, 'w') as f:
                f.write('')

            adapter = EGTEAAdapter('/fake/labels.txt', tmpdir)
            data = {'video_id': 'A01_V01-P01-01'}

            result = adapter.get_video_path(data)
            assert result == video_path

    def test_video_path_short_id(self):
        """Video ID with less than 3 parts should return None."""
        adapter = EGTEAAdapter('/fake/labels.txt', '/fake/videos')
        data = {'video_id': 'short'}
        assert adapter.get_video_path(data) is None

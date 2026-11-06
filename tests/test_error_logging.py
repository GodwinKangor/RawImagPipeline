"""Verify failures are logged and stop the pipeline cleanly."""

from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from raw_pipeline.pipeline import run_pipeline
from raw_pipeline.preprocessing import get_bayer_layout


class ErrorLoggingTests(unittest.TestCase):
    def test_module_validation_logs_and_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "errors.log"
            terminal = io.StringIO()
            with patch("raw_pipeline.pipeline.main", side_effect=lambda config: get_bayer_layout("invalid")):
                with redirect_stderr(terminal):
                    status = run_pipeline("config.json", log)
            self.assertEqual(status, 1)
            self.assertIn("Pattern must be", log.read_text())
            self.assertEqual(log.read_text().count("ERROR:"), 1)
            self.assertNotIn("Traceback", terminal.getvalue())

    def test_failure_logs_without_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "errors.log"
            terminal = io.StringIO()
            with patch("raw_pipeline.pipeline.main", side_effect=ValueError("Invalid Bayer pattern")):
                with redirect_stderr(terminal):
                    status = run_pipeline("config.json", log)
            self.assertEqual(status, 1)
            self.assertIn("Invalid Bayer pattern", log.read_text())
            self.assertIn("Invalid Bayer pattern", terminal.getvalue())
            self.assertNotIn("Traceback", terminal.getvalue())

    def test_success_has_no_error_log(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "errors.log"
            with patch("raw_pipeline.pipeline.main", return_value={}):
                self.assertEqual(run_pipeline("config.json", log), 0)
            self.assertFalse(log.exists())

    def test_unwritable_log_still_reports_error(self):
        with tempfile.TemporaryDirectory() as directory:
            terminal = io.StringIO()
            with patch("raw_pipeline.pipeline.main", side_effect=ValueError("Invalid input")):
                with redirect_stderr(terminal):
                    status = run_pipeline("config.json", Path(directory))
            self.assertEqual(status, 1)
            self.assertIn("Invalid input", terminal.getvalue())
            self.assertIn("Could not write error log", terminal.getvalue())

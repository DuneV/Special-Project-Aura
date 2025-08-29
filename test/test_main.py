# tests/test_main.py
import unittest
from unittest.mock import patch
from ..main import main

class TestMain(unittest.TestCase):

    @patch("your_main_module.G1CameraCapture")
    def test_main_runs_without_error(self, mock_capture_class):
        # Mock para evitar abrir MuJoCo
        mock_capture = mock_capture_class.return_value
        mock_capture.run_video_capture.return_value = None

        try:
            main()  # Debería ejecutarse sin errores
        except Exception as e:
            self.fail(f"main() falló con excepción: {e}")
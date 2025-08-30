# tests/test_g1_camera_capture.py

import os
import unittest
import tempfile
import numpy as np

from ..g1cameraCapture import G1CameraCapture 

# TODO: @DuneV necesitas actualizar los test para hacerlo para todo el batch de modelos. 

try:
    from ..config import Config
    MODEL_PATH = Config().model_path
except:
    MODEL_PATH = "../model/scene_objects.xml" 

class TestG1CameraCapture(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Se ejecuta una vez antes de todos los tests"""
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modelo no encontrado: {MODEL_PATH}")
        cls.capture = G1CameraCapture(MODEL_PATH)

    def test_initialization(self):
        """Prueba que el modelo se cargó correctamente"""
        self.assertIsNotNone(self.capture.model)
        self.assertIsNotNone(self.capture.data)
        self.assertGreater(self.capture.model.nbody, 0)
        self.assertGreater(self.capture.model.ncam, 0)

    def test_camera_detection(self):
        """Prueba que haya al menos una cámara"""
        self.assertGreater(self.capture.model.ncam, 0, "No se detectaron cámaras en el modelo")

    def test_looking_pose_application(self):
        """Prueba que se pueda aplicar la pose 'looking'"""
        result = self.capture.set_looking_pose()
        self.assertTrue(result, "No se pudo aplicar la pose 'looking'")

        # Verifica que el joint waist_pitch_joint tenga el valor correcto
        joint_id = self.capture.model.mj_name2id(mujoco.mjtObj.mjOBJ_JOINT, "waist_pitch_joint")
        qpos_addr = self.capture.model.jnt_qposadr[joint_id]
        current_angle = self.capture.data.qpos[qpos_addr]
        self.assertAlmostEqual(current_angle, 0.7, delta=1e-3, msg="El ángulo no es el esperado")

    def test_capture_rgb_without_viewer(self):
        """Prueba captura de imagen RGB sin abrir viewer"""
        rgb = self.capture.capture_camera_image(camera_name="head_camera", save_image=False)
        self.assertIsInstance(rgb, np.ndarray)
        self.assertEqual(rgb.shape, (720, 1280, 3))
        self.assertIn(rgb.dtype, [np.uint8])

    def test_capture_depth_without_viewer(self):
        """Prueba captura de profundidad sin viewer"""
        depth = self.capture.capture_depth_image(camera_name="head_camera", save_image=False)
        self.assertIsInstance(depth, np.ndarray)
        self.assertEqual(depth.shape, (720, 1280))
        self.assertIn(depth.dtype, [np.float32, np.float64])

    def test_video_capture_short(self):
        """Prueba grabación de video corto (1 segundo)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_output = self.capture.output_dir
            self.capture.output_dir = tmpdir

            try:
                self.capture.run_video_capture(
                    duration_seconds=1,
                    fps=10,
                    camera_name="head_camera",
                    apply_gravity=True,
                    use_sitting_pose=False
                )

                # Verifica que los archivos se hayan creado
                rgb_video = os.path.join(tmpdir, "video_rgb_head_camera.mp4")
                depth_video = os.path.join(tmpdir, "video_depth_head_camera.mp4")

                self.assertTrue(os.path.exists(rgb_video), "No se generó el video RGB")
                self.assertTrue(os.path.exists(depth_video), "No se generó el video Depth")

                # Verifica tamaño mínimo razonable
                self.assertGreater(os.path.getsize(rgb_video), 1024, "El archivo RGB es muy pequeño")
                self.assertGreater(os.path.getsize(depth_video), 1024, "El archivo Depth es muy pequeño")

            finally:
                self.capture.output_dir = original_output

    def test_capture_loop_short(self):
        """Prueba bucle de captura corto (3 imágenes)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_output = self.capture.output_dir
            self.capture.output_dir = tmpdir

            try:
                self.capture.run_capture_loop(
                    num_images=3,
                    capture_interval=0.1,
                    camera_name="head_camera",
                    apply_gravity=False,
                    use_sitting_pose=False
                )

                # Verifica imágenes guardadas
                for i in range(3):
                    rgb_path = os.path.join(tmpdir, f"camera_head_camera_{i:04d}.png")
                    depth_path = os.path.join(tmpdir, f"depth_color_head_camera_{i:04d}.png")
                    self.assertTrue(os.path.exists(rgb_path), f"No se guardó RGB {i}")
                    self.assertTrue(os.path.exists(depth_path), f"No se guardó Depth {i}")

            finally:
                self.capture.output_dir = original_output


if __name__ == '__main__':
    unittest.main()
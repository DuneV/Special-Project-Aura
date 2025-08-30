
from simulationPipeline import G1SimulationPipeline
from g1cameraCapture import G1CameraCapture
from interactive import G1InteractivePoseEditor
from models_vision.vision_recognition import VideoRecognition
from config import *

MODEL = "./models_vision/yolov8m.pt"

# OWL-ViT https://medium.com/@Mert.A/zero-shot-object-detection-with-owl-vit-and-huggingface-cbf04a509904

def process_video(path, filepath_out, model):
    '''
       Funcion para procesar el video de reconocimiento 
    '''
    recognizer = VideoRecognition(
        input_video_path = path,
        output_video_path = filepath_out,
        model = model
    )
    recognizer.runtime()

def main():
    """Función principal - Fase A completa"""
    print("=" * 60)
    print("UNITREE G1 - FASE A: Base física y sensorial")
    print("=" * 60)

    global MODEL
    config = Config()

    # Configuración del modelo con config
    model_path = config.model_path
    model_free = config.model_path_free
    path_read = config.file_read
    path_write = config.file_write

    try:
        """ pipeline = G1SimulationPipeline(model_path)
        pipeline.run_simulation(duration=20.0, enable_walking=True) """
        capture_system = G1CameraCapture(model_path)
        """ capture_system.run_capture_loop(
            num_images=5, 
            capture_interval=1.0,
            camera_name="head_camera",
            apply_gravity=True,
            use_sitting_pose=False  # False = usa looking_pose
        ) """
        capture_system.run_video_capture(duration_seconds=10, fps=30, camera_name="head_camera")
        """ interactive_ = G1InteractivePoseEditor(model_free)
        interactive_.run_interactive_editor() """

        process_success = process_video(path=path_read, filepath_out=path_write, model=MODEL)
        if process_success:
            print(f'Proceso realizado con exito con modelo {MODEL}')

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

from simulationPipeline import G1SimulationPipeline
from g1cameraCapture import G1CameraCapture
from interactive import G1InteractivePoseEditor

from config import *

def process_video():
    '''
        
    '''
    return 

def main():
    """Función principal - Fase A completa"""
    print("=" * 60)
    print("UNITREE G1 - FASE A: Base física y sensorial")
    print("=" * 60)

    config = Config()

    # Configuración del modelo con config
    model_path = config.model_path
    model_free = config.model_path_free
    
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
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
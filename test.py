import mujoco
import mujoco.viewer
import time

# Ruta al modelo (ajusta si es necesario)
xml_path = "/home/krita/mujoco_test_bed/model/humanoid.xml"

# Cargar modelo
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Crear ventana de visualización
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Activar sincronización automática
    viewer.sync()

    # Bucle de simulación
    start_time = time.time()
    while viewer.is_running() and time.time() - start_time < 10:  # 10 segundos
        mujoco.mj_step(model, data)  # Avanzar un paso de física
        viewer.sync()  # Actualizar la ventana
        time.sleep(0.01)  # Control de velocidad (100 Hz)
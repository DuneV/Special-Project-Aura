import mujoco
import mujoco.viewer
import numpy as np
import time
from typing import Dict, Tuple, Optional

class G1SimulationPipeline:
    
    
    # Metodo constructor
    
    def __init__(self, model_path: str):
        """Pipeline completo para simulación del G1 - Fase A """
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Configurar renderer para sensores RGB-D
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        
        # Inicializar sistema de teleoperación
        self.init_teleop()
        
        # Verificar modelo
        self.verify_model()
        
        print(f"✓ Modelo cargado: {self.model.nq} DOF, {self.model.nbody} cuerpos")
    
    # Verificación de modelo

    def verify_model(self):
        """Verificar dinámica y contactos del modelo"""
        print("\n=== VERIFICACIÓN DEL MODELO ===")
        # numero de coordenadas generalizada por acticulación

        print(f"Grados de libertad: {self.model.nq}")
        
        # numero de actuadores
        print(f"Actuadores: {self.model.nu}")
        
        # numero de sensores

        print(f"Sensores: {self.model.nsensor}")
        
        # numero de cuerpos rigidos

        print(f"Cuerpos: {self.model.nbody}")

        # numero de formas visuales de colision
        print(f"Geometrías: {self.model.ngeom}")
        
        # Test de estabilidad inicial
        initial_pos = self.data.qpos.copy()
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        
        pos_drift = np.linalg.norm(self.data.qpos - initial_pos)
        print(f"Deriva de posición en 100 pasos: {pos_drift:.4f}")
        
        # Resetear el robot a su posicion inicial
        mujoco.mj_resetData(self.model, self.data)
    
    # Iniciar teleoperación

    def init_teleop(self):
        """Inicializar sistema de teleoperación (teclado por ahora)"""
        self.teleop_commands = np.zeros(self.model.nu)
        self.control_mode = "joint"  # "joint" o "cartesian"
        
        print("✓ Teleoperación inicializada (controles de teclado)")
        print("  Controles: WASD (caminar), QE (rotar), IJKL (brazos)")
    
    # Obtener datos de sensores

    def get_sensor_data(self) -> Dict:
        """Obtener todos los datos sensoriales"""
        sensor_data = {}
        
        # 1. RGB-D de cámaras
        rgb_data = self.get_rgb_cameras()
        depth_data = self.get_depth_cameras()
        
        # 2. Estados articulares
        joint_data = {
            'positions': self.data.qpos.copy(),
            'velocities': self.data.qvel.copy(),
            'accelerations': self.data.qacc.copy(),
            'torques': self.data.qfrc_applied.copy()
        }
        
        # 3. Fuerzas de contacto
        contact_data = self.get_contact_forces()
        
        # 4. Sensores específicos (si existen en el modelo)
        mujoco_sensors = self.get_mujoco_sensors()
        
        sensor_data.update({
            'rgb': rgb_data,
            'depth': depth_data,
            'joints': joint_data,
            'contacts': contact_data,
            'sensors': mujoco_sensors,
            'timestamp': time.time()
        })
        
        return sensor_data
    
    def get_rgb_cameras(self) -> Dict:
        """Capturar imágenes RGB de las cámaras disponibles"""
        rgb_images = {}
        
        # Buscar cámaras en el modelo
        camera_names = []
        for i in range(self.model.ncam):
            cam_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if cam_name:
                camera_names.append(cam_name)
        
        # Si no hay cámaras definidas, usar vista por defecto
        if not camera_names:
            camera_names = ['default']
        
        for cam_name in camera_names:
            try:
                if cam_name == 'default':
                    self.renderer.update_scene(self.data)
                else:
                    self.renderer.update_scene(self.data, camera=cam_name)
                
                rgb = self.renderer.render()
                rgb_images[cam_name] = rgb
                
            except Exception as e:
                print(f"Error capturando {cam_name}: {e}")
        
        return rgb_images
    
    def get_depth_cameras(self) -> Dict:
        """Capturar mapas de profundidad"""
        depth_images = {}
        
        # Por simplicidad, simulamos depth como gradiente de distancia
        # En implementación real usarías el renderer de depth de MuJoCo
        for cam_name in ['default']:
            # Placeholder para depth - implementar con MuJoCo depth rendering
            depth_images[cam_name] = np.random.uniform(0.1, 10.0, (480, 640))
        
        return depth_images
    
    def get_contact_forces(self) -> Dict:
        """Obtener fuerzas de contacto"""
        contacts = {
            'n_contacts': self.data.ncon,
            'contact_list': [],
            'total_force': 0.0
        }
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # Calcular fuerza de contacto de forma segura
            try:
                # Usar la fuerza de contacto directamente desde los datos de contacto
                contact_force = np.zeros(6)  # 6D wrench (force + torque)
                mujoco.mj_contactForce(self.model, self.data, i, contact_force)
                force_magnitude = np.linalg.norm(contact_force[:3])  # Solo la componente de fuerza
                
            except Exception as e:
                # Fallback: calcular magnitud aproximada desde geometría
                force_magnitude = np.linalg.norm(contact.dist) if hasattr(contact, 'dist') else 0.0
            
            # Validar índices antes de usarlos
            geom1_valid = 0 <= contact.geom1 < self.model.ngeom
            geom2_valid = 0 <= contact.geom2 < self.model.ngeom
            
            contact_info = {
                'geom1': contact.geom1 if geom1_valid else -1,
                'geom2': contact.geom2 if geom2_valid else -1,
                'pos': contact.pos.copy(),
                'force': force_magnitude,
                'friction': getattr(contact, 'friction', [0.0, 0.0, 0.0])
            }
            
            contacts['contact_list'].append(contact_info)
            contacts['total_force'] += force_magnitude
        
        return contacts
    
    def get_mujoco_sensors(self) -> Dict:
        """Leer sensores definidos en el modelo MJCF"""
        sensors = {}
        
        if self.model.nsensor > 0:
            for i in range(self.model.nsensor):
                sensor_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
                if sensor_name:
                    sensor_start = self.model.sensor_adr[i]
                    sensor_dim = self.model.sensor_dim[i]
                    sensor_data = self.data.sensordata[sensor_start:sensor_start + sensor_dim]
                    sensors[sensor_name] = sensor_data.copy()
        
        return sensors
    
    def process_teleop_input(self, viewer) -> np.ndarray:
        """Procesar input de teleoperación (teclado por ahora)"""
        commands = np.zeros(self.model.nu)
        
        # Obtener estado del teclado a través del viewer
        # Nota: Esta es una implementación simplificada
        # En la práctica necesitarías integrar con GLFW callbacks o pygame
        
        # Mapeo básico de teclas a articulaciones
        # Este es un ejemplo - necesitas adaptar a la configuración específica del G1
        
        # Simular algunos comandos básicos
        t = time.time()
        
        # Patrón de caminata básico (sinusoidal)
        if hasattr(self, 'walking_enabled') and self.walking_enabled:
            frequency = 1.0  # Hz
            amplitude = 0.3  # rad
            
            # Ejemplo para articulaciones de piernas (adaptar índices según G1)
            for i in range(min(6, self.model.nu)):  # Primeras 6 articulaciones
                commands[i] = amplitude * np.sin(2 * np.pi * frequency * t + i * np.pi/3)
        
        return commands
    
    def apply_control_commands(self, commands: np.ndarray):
        """Aplicar comandos de control al robot"""
        # Limitar comandos a rangos seguros
        commands = np.clip(commands, -1.0, 1.0)
        
        # Aplicar al modelo
        if len(commands) == self.model.nu:
            self.data.ctrl[:] = commands
        else:
            print(f"Warning: Command size {len(commands)} != actuator count {self.model.nu}")
    
    def print_sensor_summary(self, sensor_data: Dict):
        """Imprimir resumen de datos sensoriales"""
        print(f"\r=== SENSORES (t={sensor_data['timestamp']:.2f}) ===", end="")
        print(f" | Cámaras RGB: {len(sensor_data['rgb'])}", end="")
        print(f" | Contactos: {sensor_data['contacts']['n_contacts']}", end="")
        print(f" | Fuerza total: {sensor_data['contacts']['total_force']:.2f}N", end="")
        
        if sensor_data['sensors']:
            print(f" | Sensores MJ: {len(sensor_data['sensors'])}", end="")
    


    
    def run_simulation(self, duration: float = 30.0, enable_walking: bool = False):
        """Ejecutar simulación completa con pipeline sensorial y teleoperación"""
        print(f"Iniciando simulación por {duration}s...")
        print("Controles: Presiona 'W' para activar caminata, 'S' para parar")
        
        self.walking_enabled = enable_walking
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.sync()
            
            start_time = time.time()
            last_sensor_print = 0
            
            while viewer.is_running() and (time.time() - start_time) < duration:
                # 1. Procesar teleoperación
                teleop_commands = self.process_teleop_input(viewer)
                
                # 2. Aplicar comandos
                self.apply_control_commands(teleop_commands)
                
                # 3. Step de física
                mujoco.mj_step(self.model, self.data)
                
                # 4. Obtener datos sensoriales (cada 0.1s para no saturar)
                current_time = time.time()
                if current_time - last_sensor_print > 0.1:
                    sensor_data = self.get_sensor_data()
                    self.print_sensor_summary(sensor_data)
                    last_sensor_print = current_time
                
                # 5. Actualizar visualización
                viewer.sync()
                
                # Control de frecuencia
                time.sleep(0.005)  # ~200 FPS máx
        
        print(f"\n✓ Simulación completada en {time.time() - start_time:.2f}s")
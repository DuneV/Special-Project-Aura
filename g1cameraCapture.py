import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import imageio
import cv2
import matplotlib.pyplot as plt

class G1CameraCapture:
    def __init__(self, model_path: str):
        """Inicializa el modelo y el renderizador para capturar RGB-D"""
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Configuración visual mejorada
        self.model.vis.global_.offwidth = 1280
        self.model.vis.global_.offheight = 720

        self.renderer = mujoco.Renderer(self.model, 
                                        height=720, width=1280)
    
        
        self.model.vis.rgba.fog = [0.8, 0.8, 0.8, 1]  # Niebla suave
        
        self.model.opt.gravity[0] = 0.0    # X
        self.model.opt.gravity[1] = 0.0    # Y  
        self.model.opt.gravity[2] = -9.81  # Z (hacia abajo)
        
        
        self.model.opt.timestep = 0.001        # Timestep más pequeño para mejor estabilidad
        self.model.opt.iterations = 100       # Más iteraciones para mejor convergencia
        self.model.opt.tolerance = 1e-10      # Tolerancia más estricta
        
        self.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON  # Solver Newton
        
        self.depth_near = 0.01  # 1cm mínimo
        self.depth_far = 50.0   # 50m máximo
        self.depth_scale = 1000.0
        self.model.vis.quality.offsamples = 8  # Más muestras para antialiasing
        self.model.vis.quality.shadowsize = 2048
        self.output_dir = "g1_camera_data"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Modelo cargado: {self.model.nq} DOF, {self.model.nbody} cuerpos")
        print(f"Gravedad activada: {self.model.opt.gravity}")
        print(f"Timestep: {self.model.opt.timestep}s")
        
        print("Cámaras disponibles:")
        for i in range(self.model.ncam):
            cam_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            print(f"  - {cam_name} (id: {i})")
    
    def set_looking_pose(self):
        """Configura el robot en una pose fija mirando hacia abajo, moviendo solo el waist_pitch_joint"""
        print("Aplicando pose fija: 'looking_pose' (waist_pitch_joint = 0.5 rad)")

        # Reiniciar la configuración del robot a posición neutral
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # Buscar el ID de la articulación
        joint_name = "waist_pitch_joint"
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        
        if joint_id == -1:
            print(f"Articulación '{joint_name}' no encontrada en el modelo.")
            print("Articulaciones disponibles:")
            for i in range(self.model.njnt):
                jnt_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if jnt_name and "waist" in jnt_name.lower():
                    print(f"  - {jnt_name} (id: {i})")
            return False

        # Obtener la posición en el vector qpos
        qpos_addr = self.model.jnt_qposadr[joint_id]

        # Aplicar el ángulo deseado
        target_angle = 0.7  # radianes (28.6 grados hacia abajo)
        self.data.qpos[qpos_addr] = target_angle

        # Actualizar cinemática para que MuJoCo calcule las nuevas posiciones
        mujoco.mj_forward(self.model, self.data)

        print(f"✓ Pose aplicada: {joint_name} = {target_angle:.3f} rad ({np.degrees(target_angle):.1f}°)")
        print(f"  Posición qpos[{qpos_addr}] = {self.data.qpos[qpos_addr]:.3f}")
        
        return True

    def set_sitting_pose(self):
        """Configura el robot en posición sentada con estabilización"""
        print("Configurando posición sentada...")
        
        # Reset a keyframe 0 si existe
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        
        # Ajustar articulaciones para posición sentada
        joint_targets = {
            "left_hip_pitch_joint": -2.5,
            "left_knee_joint": -0.8,
            "left_ankle_pitch_joint": 0.6,
            "right_hip_pitch_joint": -2.5,
            "right_knee_joint": -0.8,
            "right_ankle_pitch_joint": 0.6,
            "waist_pitch_joint": 0.4,
            "left_shoulder_pitch_joint": -0.7,
            "left_elbow_pitch_joint": -0.3,
            "right_shoulder_pitch_joint": -0.7,
            "right_elbow_pitch_joint": -0.3
        }
        
        # Aplicar posición sentada
        applied_joints = 0
        for joint_name, value in joint_targets.items():
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                joint_qposadr = self.model.jnt_qposadr[joint_id]
                self.data.qpos[joint_qposadr] = value
                applied_joints += 1
                print(f"  ✓ {joint_name} = {value:.3f} rad")
            else:
                print(f"Articulación '{joint_name}' no encontrada")
        
        # Posicionar el robot a altura adecuada para sentarse
        self.data.qpos[2] = 0.5  # Altura Z ligeramente mayor
        mujoco.mj_forward(self.model, self.data)
        
        print(f"Robot configurado en posición sentada ({applied_joints}/{len(joint_targets)} articulaciones aplicadas)")
        return applied_joints > 0
    
    def stabilize_objects(self, steps=200):
        """Permite que los objetos caigan y se estabilicen con gravedad"""
        print(f"Aplicando gravedad y estabilizando objetos ({steps} pasos)...")
        
        for i in range(steps):
            mujoco.mj_step(self.model, self.data)
            
            # Mostrar progreso cada 50 pasos
            if (i + 1) % 50 == 0:
                print(f"   Paso {i+1}/{steps}")
        
        print("Objetos estabilizados por gravedad")
    
    def capture_camera_image(self, camera_name="head_camera", save_image=True, image_index=0):
        """Captura lo que ve una cámara específica"""
        # Obtener ID de la cámara
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        
        if camera_id == -1:
            print(f"Cámara '{camera_name}' no encontrada. Usando cámara 0.")
            camera_id = 0
    
        self.renderer.update_scene(self.data, camera=camera_id)
        rgb = self.renderer.render()
        
        # Guardar imagen
        if save_image:
            rgb_path = os.path.join(self.output_dir, f"camera_{camera_name}_{image_index:04d}.png")
            imageio.imwrite(rgb_path, rgb)
            print(f"Imagen RGB guardada: {rgb_path}")
        
        return rgb
    
    def capture_depth_image(self, camera_name="head_camera", save_image=True, image_index=0):
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id == -1:
            print(f"Cámara '{camera_name}' no encontrada. Usando cámara 0.")
            camera_id = 0

        try:
            # Actualizar escena
            self.renderer.update_scene(self.data, camera=camera_id)
            
            # Activar renderizado de profundidad
            self.renderer.enable_depth_rendering()
            
            # Renderizar profundidad (valores en metros)
            depth = self.renderer.render()
            
            if depth is None:
                print("Error: No se pudo renderizar la profundidad")
                return None
                
            # Filtrar valores fuera de rango
            depth = np.where(depth > self.depth_far, self.depth_far, depth)
            depth = np.where(depth < self.depth_near, self.depth_near, depth)
            
            # Guardar datos si es necesario
            if save_image:
                # 1. Guardar datos brutos de profundidad (float32)
                depth_raw_path = os.path.join(self.output_dir, f"depth_raw_{camera_name}_{image_index:04d}.npy")
                np.save(depth_raw_path, depth.astype(np.float32))
                
                # 2. Crear visualización mejorada con escala logarítmica
                depth_log = np.log1p(depth - self.depth_near)
                depth_max_log = np.log1p(self.depth_far - self.depth_near)
                depth_vis = (depth_log / depth_max_log * 255).astype(np.uint8)
                
                # Aplicar colormap para mejor visualización
                depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                
                # 3. Guardar visualización a color
                depth_color_path = os.path.join(self.output_dir, f"depth_color_{camera_name}_{image_index:04d}.png")
                cv2.imwrite(depth_color_path, depth_color)
                
                # 4. Guardar visualización en escala de grises
                depth_gray_path = os.path.join(self.output_dir, f"depth_gray_{camera_name}_{image_index:04d}.png")
                cv2.imwrite(depth_gray_path, depth_vis)
                
                print(f"Profundidad guardada: {depth_raw_path}")
                print(f"Visualización: {depth_color_path}")
                
                # Mostrar estadísticas detalladas
                valid_pixels = np.sum((depth > self.depth_near) & (depth < self.depth_far))
                total_pixels = depth.size
                coverage = (valid_pixels / total_pixels) * 100
                
                print(f"Estadísticas de profundidad:")
                print(f"  Rango: {depth.min():.3f}m - {depth.max():.3f}m")
                print(f"  Media: {depth.mean():.3f}m, Mediana: {np.median(depth):.3f}m")
                print(f"  Cobertura válida: {coverage:.1f}%")
                
                # Crear histograma de profundidad
                plt.figure(figsize=(10, 6))
                plt.hist(depth.flatten(), bins=100, range=(self.depth_near, self.depth_far))
                plt.xlabel('Profundidad (m)')
                plt.ylabel('Frecuencia')
                plt.title(f'Histograma de Profundidad - {camera_name}')
                hist_path = os.path.join(self.output_dir, f"depth_hist_{camera_name}_{image_index:04d}.png")
                plt.savefig(hist_path)
                plt.close()
                print(f"Histograma: {hist_path}")
            
            return depth
            
        except Exception as e:
            print(f"Error en captura de profundidad: {e}")
            return None
        
        finally:
            # Asegurarse de desactivar el modo profundidad
            self.renderer.disable_depth_rendering()
    
    def print_current_joint_states(self):
        """Imprime el estado actual de todas las articulaciones para debug"""
        print("\n=== ESTADO ACTUAL DE ARTICULACIONES ===")
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name:
                qpos_addr = self.model.jnt_qposadr[i]
                current_value = self.data.qpos[qpos_addr]
                print(f"{joint_name}: {current_value:.4f} rad ({np.degrees(current_value):.2f}°)")
        print("=======================================\n")
    
    def run_capture_loop(self, num_images=10, capture_interval=0.5, camera_name="head_camera", 
                        apply_gravity=True, use_sitting_pose=False):
        """Bucle principal para capturar múltiples imágenes"""
        print(f"Iniciando captura de {num_images} imágenes...")
        print(f"Cámara: {camera_name}")
        print(f"Aplicar gravedad: {apply_gravity}")
        print(f"Pose: {'Sentada' if use_sitting_pose else 'Looking Down'}")
        
        # Configurar la pose
        pose_applied = self.set_looking_pose()
            
        if not pose_applied:
            print("No se pudo aplicar la pose correctamente.")
        
        # Debug: mostrar estado de las articulaciones después de aplicar pose
        """ self.print_current_joint_states() """
        
        """ if apply_gravity:
            self.stabilize_objects(steps=300) """
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1      # Sombras
            viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 0   # Sin wireframe
            viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1  # Reflejos
            viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1      # Skybox
            viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1         # Niebla
            
            # Configuración de luces mejorada
            viewer.user_scn.nlight = min(8, viewer.user_scn.nlight)  # Máximo 8 luces
            
            viewer.sync()
            time.sleep(2.0)  # Más tiempo de estabilización visual
            
            for i in range(num_images):
                print(f"\nCapturando imagen {i+1}/{num_images}...")
                
                # MINI-SIMULACIÓN ENTRE CAPTURAS
                if apply_gravity and i > 0:
                    print("Aplicando pequeños pasos de física...")
                    for _ in range(10):  # Pequeños pasos entre capturas
                        mujoco.mj_step(self.model, self.data)
                
                # Capturar RGB
                rgb = self.capture_camera_image(camera_name, save_image=True, image_index=i)
                
                # Capturar Depth
                try:
                    depth = self.capture_depth_image(camera_name, save_image=True, image_index=i)
                    
                    # Estadísticas de profundidad
                    if depth is not None:
                        print(f"Profundidad: min={depth.min():.3f}m, max={depth.max():.3f}m, mean={depth.mean():.3f}m")
                        
                except Exception as e:
                    print(f"Error capturando depth para imagen {i}: {e}")
                
                # Esperar antes de la siguiente captura
                time.sleep(capture_interval)
                viewer.sync()
            
            print("\n¡Captura completada con éxito!")
            print(f"Archivos guardados en: {self.output_dir}")

    def run_video_capture(self, duration_seconds=10, fps=30, camera_name="head_camera", 
                     apply_gravity=True, use_sitting_pose=False):
        """Graba video durante los segundos especificados"""
        print(f"Iniciando grabación de video por {duration_seconds} segundos...")
        print(f"Cámara: {camera_name}, FPS: {fps}")
        print(f"Aplicar gravedad: {apply_gravity}")
        print(f"Pose: {'Sentada' if use_sitting_pose else 'Looking Down'}")
        
        KP = 100.0  # Ganancia proporcional
        KD = 10.0   # Ganancia derivativa
        target_angle = 0.7
        # Configurar la pose
        pose_applied = self.set_looking_pose()
        target_joint = "waist_pitch_joint"
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, target_joint)
        qpos_addr = self.model.jnt_qposadr[joint_id]
        qvel_addr = self.model.jnt_dofadr[joint_id] 
        actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{target_joint}_motor")
        has_actuator = actuator_id >= 0
        if not pose_applied:
            print("No se pudo aplicar la pose correctamente.")
        
        # Configurar video writers
        video_path_rgb = os.path.join(self.output_dir, f"video_rgb_{camera_name}.mp4")
        video_path_depth = os.path.join(self.output_dir, f"video_depth_{camera_name}.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer_rgb = cv2.VideoWriter(video_path_rgb, fourcc, fps, (1280, 720))
        video_writer_depth = cv2.VideoWriter(video_path_depth, fourcc, fps, (1280, 720))
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Configuración visual
            viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
            viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 0
            viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
            viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
            viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
            
            viewer.sync()
            time.sleep(1.0)
            
            total_frames = int(duration_seconds * fps)
            frame_interval = 1.0 / fps
            
            print(f" Grabando {total_frames} frames...")
            
            for frame in range(total_frames):
                start_time = time.time()
                """ current_angle = self.data.qpos[qpos_addr]
                current_velocity = self.data.qvel[qvel_addr]
                error = target_angle - current_angle
                d_error = -current_velocity
                error = target_angle - current_angle
                d_error = -current_velocity  # derivada del error """
                pose_applied = self.set_looking_pose() # apagar para mejorar fisicas
                """ control_torque = KP * error + KD * d_error
                if has_actuator:
                    self.data.ctrl[actuator_id] = control_torque
                else:
                    # Si no hay actuator, aplicar fuerza directamente
                    self.data.qfrc_applied[qvel_addr] = control_torque """
                # Simular física si está habilitada
                if apply_gravity:
                    mujoco.mj_step(self.model, self.data)
                
                # Capturar RGB
                rgb = self.capture_camera_image(camera_name, save_image=False, image_index=frame)
                rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # OpenCV usa BGR
                video_writer_rgb.write(rgb_bgr)
                
                # Capturar Depth
                try:
                    depth = self.capture_depth_image(camera_name, save_image=False, image_index=frame)
                    if depth is not None:
                        # Convertir depth a visualización colorizada
                        depth_log = np.log1p(depth - self.depth_near)
                        depth_max_log = np.log1p(self.depth_far - self.depth_near)
                        depth_vis = (depth_log / depth_max_log * 255).astype(np.uint8)
                        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                        video_writer_depth.write(depth_color)
                except Exception as e:
                    print(f"Error en depth frame {frame}: {e}")
                
                # Control de timing para mantener FPS
                elapsed = time.time() - start_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                
                # Progreso cada segundo
                if (frame + 1) % fps == 0:
                    print(f"   {(frame + 1) // fps}/{duration_seconds} segundos grabados")
                
                viewer.sync()
            
            # Cerrar video writers
            video_writer_rgb.release()
            video_writer_depth.release()
            
            print(f"\n¡Grabación completada!")
            print(f"Video RGB: {video_path_rgb}")
            print(f"Video Depth: {video_path_depth}")

import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import imageio
import glfw

class G1InteractivePoseEditor:
    def __init__(self, model_path: str):
        """Inicializa el editor interactivo de poses"""
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        self.output_dir = "g1_pose_configs"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configuración inicial
        self.selected_joint_idx = 0
        self.joint_increment = 0.05  # Radianes por paso
        self.freeze_mode = False
        self.saved_poses = {}
        
        # Lista de articulaciones disponibles (excluyendo la base flotante)
        self.joint_names = []
        for i in range(self.model.njnt):
            jnt_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if jnt_name and self.model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
                self.joint_names.append(jnt_name)
        
        print(f"Encontradas {len(self.joint_names)} articulaciones:")
        for i, name in enumerate(self.joint_names):
            print(f"{i}: {name}")
    
    def get_joint_qpos_addr(self, joint_name):
        """Obtiene la dirección en qpos para una articulación"""
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id != -1:
            return self.model.jnt_qposadr[joint_id]
        return -1
    
    def adjust_joint(self, joint_name, delta):
        """Ajusta una articulación por un incremento/delta"""
        qpos_addr = self.get_joint_qpos_addr(joint_name)
        if qpos_addr != -1:
            current_value = self.data.qpos[qpos_addr]
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            joint_range = self.model.jnt_range[joint_id]
            
            # Aplicar límites de rango si existen
            new_value = current_value + delta
            if joint_range[0] != joint_range[1]:  # Si tiene rango definido
                new_value = np.clip(new_value, joint_range[0], joint_range[1])
            
            self.data.qpos[qpos_addr] = new_value
            print(f"Ajustando {joint_name}: {current_value:.3f} → {new_value:.3f}")
    
    def freeze_current_pose(self):
        """Congela la posición actual desactivando la gravedad"""
        self.model.opt.gravity[2] = 0  # Gravedad cero en Z
        self.freeze_mode = True
        print("✓ Posición congelada (gravedad desactivada)")
    
    def unfreeze_pose(self):
        """Reactiva la gravedad"""
        self.model.opt.gravity[2] = -9.81  # Gravedad normal
        self.freeze_mode = False
        print("✓ Gravedad reactivada")
    
    def save_current_pose(self, pose_name):
        """Guarda la configuración actual de articulaciones"""
        pose_config = {}
        for joint_name in self.joint_names:
            qpos_addr = self.get_joint_qpos_addr(joint_name)
            if qpos_addr != -1:
                pose_config[joint_name] = float(self.data.qpos[qpos_addr])
        
        self.saved_poses[pose_name] = pose_config
        
        # Guardar en archivo
        filename = os.path.join(self.output_dir, f"{pose_name}.npz")
        np.savez(filename, **pose_config)
        print(f"✓ Pose '{pose_name}' guardada en {filename}")
        
        return pose_config
    
    def load_pose(self, pose_name):
        """Carga una configuración de articulaciones guardada"""
        filename = os.path.join(self.output_dir, f"{pose_name}.npz")
        if os.path.exists(filename):
            pose_config = np.load(filename)
            for joint_name in pose_config.files:
                qpos_addr = self.get_joint_qpos_addr(joint_name)
                if qpos_addr != -1:
                    self.data.qpos[qpos_addr] = pose_config[joint_name]
            print(f"✓ Pose '{pose_name}' cargada")
            mujoco.mj_forward(self.model, self.data)
        else:
            print(f"❌ Pose '{pose_name}' no encontrada")
    
    def run_interactive_editor(self):
        """Ejecuta el editor interactivo de poses"""
        print("\n" + "="*50)
        print("EDITOR INTERACTIVO DE POSES - ROBOT G1")
        print("="*50)
        print("CONTROLES:")
        print("  ← → : Seleccionar articulación")
        print("  ↑ ↓ : Ajustar articulación seleccionada")
        print("  F   : Congelar/descongelar posición")
        print("  S   : Guardar pose actual")
        print("  L   : Cargar pose guardada")
        print("  ESC : Salir")
        print("="*50)
        
        # Inicializar GLFW para capturar input
        if not glfw.init():
            return
        
        # Crear ventana
        window = glfw.create_window(640, 480, "G1 Pose Editor", None, None)
        if not window:
            glfw.terminate()
            return
        
        glfw.make_context_current(window)
        
        # Inicializar viewer de MuJoCo
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            print("Editor listo. Usa las teclas para ajustar la pose.")
            
            while not glfw.window_should_close(window) and viewer.is_running():
                # Procesar eventos de teclado
                glfw.poll_events()
                
                # Leer estado de teclas
                keys = {
                    glfw.KEY_LEFT: glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS,
                    glfw.KEY_RIGHT: glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS,
                    glfw.KEY_UP: glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS,
                    glfw.KEY_DOWN: glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS,
                    glfw.KEY_F: glfw.get_key(window, glfw.KEY_F) == glfw.PRESS,
                    glfw.KEY_S: glfw.get_key(window, glfw.KEY_S) == glfw.PRESS,
                    glfw.KEY_L: glfw.get_key(window, glfw.KEY_L) == glfw.PRESS,
                    glfw.KEY_ESCAPE: glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS,
                }
                
                # Cambiar articulación seleccionada
                if keys[glfw.KEY_LEFT]:
                    self.selected_joint_idx = (self.selected_joint_idx - 1) % len(self.joint_names)
                    print(f"Articulación seleccionada: {self.joint_names[self.selected_joint_idx]}")
                    time.sleep(0.2)  # Debounce
                
                if keys[glfw.KEY_RIGHT]:
                    self.selected_joint_idx = (self.selected_joint_idx + 1) % len(self.joint_names)
                    print(f"Articulación seleccionada: {self.joint_names[self.selected_joint_idx]}")
                    time.sleep(0.2)  # Debounce
                
                # Ajustar articulación
                current_joint = self.joint_names[self.selected_joint_idx]
                if keys[glfw.KEY_UP]:
                    self.adjust_joint(current_joint, self.joint_increment)
                    time.sleep(0.1)  # Debounce
                
                if keys[glfw.KEY_DOWN]:
                    self.adjust_joint(current_joint, -self.joint_increment)
                    time.sleep(0.1)  # Debounce
                
                # Congelar/descongelar
                if keys[glfw.KEY_F]:
                    if self.freeze_mode:
                        self.unfreeze_pose()
                    else:
                        self.freeze_current_pose()
                    time.sleep(0.5)  # Debounce
                
                # Guardar pose
                if keys[glfw.KEY_S]:
                    pose_name = input("Nombre para esta pose: ")
                    if pose_name:
                        self.save_current_pose(pose_name)
                    time.sleep(0.5)  # Debounce
                
                # Cargar pose
                if keys[glfw.KEY_L]:
                    pose_name = input("Nombre de la pose a cargar: ")
                    if pose_name:
                        self.load_pose(pose_name)
                    time.sleep(0.5)  # Debounce
                
                # Salir
                if keys[glfw.KEY_ESCAPE]:
                    break
                
                # Actualizar simulación si no está congelada
                if not self.freeze_mode:
                    mujoco.mj_step(self.model, self.data)
                
                # Actualizar visualización
                viewer.sync()
                time.sleep(0.01)
        
        glfw.terminate()
        print("Editor cerrado.")

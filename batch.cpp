#include <iostream>
#include <cstdlib>
#include <string>
#include <thread>
#include <chrono>
#include <filesystem>

namespace fs = std::filesystem;

class MuJoCoAutomator {
private:
    std::string pythonScriptPath;
    std::string dinoGroundingPath;
    std::string modelPath;
    int numImages;
    double captureInterval;

public:
    MuJoCoAutomator(const std::string& pyScript, 
                   const std::string& dinoPath,
                   const std::string& model,
                   int images = 20,
                   double interval = 0.3)
        : pythonScriptPath(pyScript), 
          dinoGroundingPath(dinoPath),
          modelPath(model),
          numImages(images),
          captureInterval(interval) {}

    bool checkFileExists(const std::string& path) {
        if (!fs::exists(path)) {
            std::cerr << "Error: Archivo no encontrado - " << path << std::endl;
            return false;
        }
        return true;
    }

    int runPythonScript() {
        std::cout << "🔧 Ejecutando script de MuJoCo..." << std::endl;
        
        std::string command = "python3 " + pythonScriptPath + " " + modelPath + 
                             " --num_images " + std::to_string(numImages) +
                             " --capture_interval " + std::to_string(captureInterval);
        
        std::cout << "Comando: " << command << std::endl;
        
        int result = std::system(command.c_str());
        
        if (result == 0) {
            std::cout << "Script de MuJoCo ejecutado exitosamente" << std::endl;
        } else {
            std::cerr << "Error al " << std::endl;
        }
        
        return result;
    }

    int runDinoGrounding() {
        std::cout << "🦖 Ejecutando DINO Grounding..." << std::endl;
        
        std::string command = "python3 " + dinoGroundingPath + " --input_dir g1_camera_data";
        
        std::cout << "Comando: " << command << std::endl;
        
        int result = std::system(command.c_str());
        
        if (result == 0) {
            std::cout << "DINO Grounding ejecutado exitosamente" << std::endl;
        } else {
            std::cerr << "Error al ejecutar DINO Grounding" << std::endl;
        }
        
        return result;
    }

    void run() {
        std::cout << "Iniciando automatización de MuJoCo + DINO Grounding" << std::endl;
        std::cout << "=====================================================" << std::endl;

        // Verificar que los archivos existen
        if (!checkFileExists(pythonScriptPath) || 
            !checkFileExists(dinoGroundingPath) || 
            !checkFileExists(modelPath)) {
            std::cerr << "No se pueden encontrar los archivos necesarios" << std::endl;
            return;
        }

        // Ejecutar MuJoCo
        std::cout << "\nFase 1: Captura de imágenes con MuJoCo" << std::endl;
        std::cout << "=====================================================" << std::endl;
        
        int mujocoResult = runPythonScript();
        if (mujocoResult != 0) {
            std::cerr << "Deteniendo ejecución debido a error en MuJoCo" << std::endl;
            return;
        }

        // Pequeña pausa entre procesos
        std::this_thread::sleep_for(std::chrono::seconds(2));

        // Ejecutar DINO Grounding
        std::cout << "\nFase 2: Procesamiento con DINO Grounding" << std::endl;
        std::cout << "=====================================================" << std::endl;
        
        int dinoResult = runDinoGrounding();
        if (dinoResult != 0) {
            std::cerr << "Error en DINO Grounding" << std::endl;
            return;
        }

        std::cout << "\nProceso completado exitosamente!" << std::endl;
        std::cout << "=====================================================" << std::endl;
    }
};

int main(int argc, char* argv[]) {
    std::string pythonScript = "mujoco_capture.py";
    std::string dinoScript = "dino_grounding.py";
    std::string modelPath = "g1_with_floor.xml";
    int numImages = 20;
    double interval = 0.3;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--script" && i + 1 < argc) {
            pythonScript = argv[++i];
        } else if (arg == "--dino" && i + 1 < argc) {
            dinoScript = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            modelPath = argv[++i];
        } else if (arg == "--num_images" && i + 1 < argc) {
            numImages = std::stoi(argv[++i]);
        } else if (arg == "--interval" && i + 1 < argc) {
            interval = std::stod(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Uso: " << argv[0] << " [opciones]" << std::endl;
            std::cout << "Opciones:" << std::endl;
            std::cout << "  --script <path>    Script Python de MuJoCo (default: mujoco_capture.py)" << std::endl;
            std::cout << "  --dino <path>      Script de DINO Grounding (default: dino_grounding.py)" << std::endl;
            std::cout << "  --model <path>     Archivo XML del modelo (default: g1_with_floor.xml)" << std::endl;
            std::cout << "  --num_images <N>   Número de imágenes a capturar (default: 20)" << std::endl;
            std::cout << "  --interval <sec>   Intervalo entre capturas (default: 0.3)" << std::endl;
            std::cout << "  --help             Mostrar esta ayuda" << std::endl;
            return 0;
        }
    }
    MuJoCoAutomator automator(pythonScript, dinoScript, modelPath, numImages, interval);
    automator.run();

    return 0;
}
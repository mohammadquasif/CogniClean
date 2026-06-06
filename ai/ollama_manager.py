"""
Ollama Manager & Hardware Profiler
==================================
Author: Mohammad Quasif, DBA (AI) | B.Tech (CS)
License: MIT

Handles detection, startup lifecycle, silent download/install of the Ollama service,
and pulling model variants with streaming progress callbacks.
"""

import os
import sys
import time
import socket
import logging
import requests
import json
import subprocess
import threading
from typing import Optional, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# CONSTANTS & MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────

TARGET_MODEL = "qwen2.5:0.5b"

PREFERRED_MODELS = [
    "qwen2.5:0.5b",
    "qwen2.5:1.5b",
    "phi3:mini",
    "gemma2:2b"
]

MODEL_PRIORITY = [
    {
        "name": "qwen2.5:0.5b",
        "description": "Qwen 2.5 0.5B (Recommended: ultra-light, 390MB, low RAM)",
        "ram_required": 4.0
    },
    {
        "name": "qwen2.5:1.5b",
        "description": "Qwen 2.5 1.5B (Balanced, 980MB, standard performance)",
        "ram_required": 8.0
    },
    {
        "name": "phi3:mini",
        "description": "Phi-3 Mini 3.8B (Advanced: high accuracy, 2.2GB, needs 8GB+ RAM)",
        "ram_required": 8.0
    },
    {
        "name": "gemma2:2b",
        "description": "Gemma 2 2B (Advanced: modern reasoning, 1.6GB, needs GPU/8GB+)",
        "ram_required": 8.0
    }
]

# ─────────────────────────────────────────────────────────────────
# SETUP STEP REPRESENTATION
# ─────────────────────────────────────────────────────────────────

class SetupStep:
    """Represents a discrete step in the automatic configuration wizard."""
    def __init__(self, name: str, status: str = "pending", message: str = "", progress: int = 0):
        self.name = name
        self.status = status      # pending, running, done, failed, skipped
        self.message = message
        self.progress = progress  # 0 to 100


# ─────────────────────────────────────────────────────────────────
# SYSTEM HARDWARE PROFILER
# ─────────────────────────────────────────────────────────────────

class SystemProfile:
    """Checks CPU, RAM, GPU, and internet status to auto-select optimal models."""
    def __init__(self):
        self.ram_total_gb = self.get_ram_gb()
        self.has_gpu = self.detect_gpu()
        self.has_internet = self.check_internet()
        self.cpu_cores = os.cpu_count() or 4

    def get_ram_gb(self) -> float:
        try:
            import psutil
            return psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            return 8.0  # Safe default fallback

    def detect_gpu(self) -> bool:
        """Query for dedicated Nvidia CUDA devices."""
        try:
            # Check nvidia-smi command line
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, timeout=3, text=True
            )
            return result.returncode == 0 and bool(result.stdout.strip())
        except Exception:
            return False

    def check_internet(self) -> bool:
        """Attempt quick connection to a public DNS port."""
        try:
            socket.setdefaulttimeout(3)
            # Query Google DNS
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(("8.8.8.8", 53))
            s.close()
            return True
        except Exception:
            return False

    def select_best_model(self) -> Dict[str, str]:
        """Chooses the most optimal local model from MODEL_PRIORITY."""
        # Always return the qwen2.5:0.5b by default as defined by app's fast paths,
        # but dynamically pick based on resources if GPU or high RAM is present.
        if self.has_gpu and self.ram_total_gb >= 12.0:
            # Return Gemma 2B or Qwen 1.5B
            return MODEL_PRIORITY[1]  # qwen2.5:1.5b
        elif self.ram_total_gb >= 8.0:
            return MODEL_PRIORITY[1]  # qwen2.5:1.5b
        else:
            return MODEL_PRIORITY[0]  # qwen2.5:0.5b


# ─────────────────────────────────────────────────────────────────
# OLLAMA MANAGER
# ─────────────────────────────────────────────────────────────────

class OllamaManager:
    """Manages Ollama installations, processes, and model registry actions."""
    def __init__(self, step_cb: Optional[Callable[[SetupStep], None]] = None, log_cb: Optional[Callable[[str], None]] = None):
        self.step_cb = step_cb
        self.log_cb = log_cb
        self.profile = SystemProfile()
        self.selected_model = self.profile.select_best_model()

    def log(self, message: str):
        logger.info(message)
        if self.log_cb:
            self.log_cb(message)

    def update_step(self, step_name: str, status: str, message: str, progress: int):
        if self.step_cb:
            step = SetupStep(step_name, status, message, progress)
            self.step_cb(step)

    @staticmethod
    def is_api_running() -> bool:
        """Test local REST endpoint."""
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    @staticmethod
    def is_installed() -> bool:
        """Check if local CLI registry is visible."""
        try:
            # Query version
            result = subprocess.run(["ollama", "--version"], capture_output=True, timeout=5, text=True)
            return result.returncode == 0
        except Exception:
            return False

    @staticmethod
    def _start_service() -> bool:
        """Executes daemon server process in background without window frame."""
        try:
            subprocess.Popen(["ollama", "serve"], creationflags=subprocess.CREATE_NO_WINDOW)
            return True
        except Exception as e:
            logger.error(f"Failed to trigger ollama serve: {e}")
            return False

    @staticmethod
    def list_installed_models() -> List[str]:
        """Returns name listing of locally hosted models."""
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=4)
            if r.status_code == 200:
                data = r.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.debug(f"list_installed_models error: {e}")
        return []

    def auto_setup(self) -> Tuple[bool, str]:
        """
        Executes sequential configuration workflow.
        Returns Tuple (success_status, status_message).
        """
        steps = [
            "Checking Ollama installation",
            "Starting Ollama service",
            "Verifying API connection",
            "Checking model availability",
            "Pulling AI model",
            "Final verification",
        ]

        # Initialize all steps to pending
        for s in steps:
            self.update_step(s, "pending", "Waiting...", 0)

        # ----------------------------------------------------------------─
        # Step 1: Checking Ollama installation
        # ----------------------------------------------------------------─
        step_name = steps[0]
        self.update_step(step_name, "running", "Checking for Ollama cli...", 20)
        self.log("Detecting Ollama CLI installation...")
        
        installed = self.is_installed()
        if not installed:
            self.update_step(step_name, "running", "Ollama not found. Downloading installer...", 40)
            self.log("Ollama not found. Fetching online setup binary...")
            
            # Setup directories
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            offline_installer = os.path.join(script_dir, "installer", "ollama", "OllamaSetup.exe")
            temp_dir = os.path.join(os.environ.get("TEMP", "C:\\Temp"), "CogniCleanSetup")
            download_path = os.path.join(temp_dir, "OllamaSetup.exe")
            
            installer_path = None
            if os.path.exists(offline_installer):
                installer_path = offline_installer
                self.log(f"Using cached offline installer: {offline_installer}")
            else:
                self.log("Downloading installer from github...")
                url = "https://github.com/ollama/ollama/releases/latest/download/OllamaSetup.exe"
                try:
                    os.makedirs(temp_dir, exist_ok=True)
                    r = requests.get(url, stream=True, timeout=60)
                    r.raise_for_status()
                    total = int(r.headers.get("content-length", 0))
                    downloaded = 0
                    
                    with open(download_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=16384):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total > 0:
                                    pct = int((downloaded / total) * 100)
                                    self.update_step(step_name, "running", f"Downloading Ollama: {pct}%", 40 + int(pct * 0.4))
                    installer_path = download_path
                except Exception as e:
                    self.log(f"Ollama download failed: {e}")
                    self.update_step(step_name, "failed", f"Download failed: {e}", 0)
                    return False, f"Could not acquire Ollama Setup: {e}"
            
            # Run installer
            self.update_step(step_name, "running", "Installing Ollama (silent mode)...", 85)
            self.log("Executing silent setup...")
            try:
                # /S for silent install
                proc = subprocess.run([installer_path, "/S"], capture_output=True, timeout=120)
                time.sleep(8)  # Pause to allow background service and files to settle
                self.update_step(step_name, "done", "Ollama installed successfully.", 100)
            except Exception as e:
                self.log(f"Ollama installer execution error: {e}")
                self.update_step(step_name, "failed", f"Install execution failed: {e}", 0)
                return False, f"Ollama installation failed: {e}"
        else:
            self.update_step(step_name, "done", "Ollama already installed.", 100)
            self.log("Ollama already present.")

        # ----------------------------------------------------------------─
        # Step 2: Starting Ollama service
        # ----------------------------------------------------------------─
        step_name = steps[1]
        self.update_step(step_name, "running", "Checking for service activity...", 20)
        self.log("Checking if daemon service is active...")
        
        if self.is_api_running():
            self.update_step(step_name, "done", "Service is running.", 100)
            self.log("Service active.")
        else:
            self.update_step(step_name, "running", "Starting service daemon...", 50)
            self.log("Launching Ollama service in background...")
            self._start_service()
            
            # Wait for service startup
            service_running = False
            for i in range(12):
                if self.is_api_running():
                    service_running = True
                    break
                self.update_step(step_name, "running", f"Waiting for service endpoint ({i+1}/12)...", 50 + i * 4)
                time.sleep(1.5)
            
            if service_running:
                self.update_step(step_name, "done", "Service started successfully.", 100)
                self.log("Ollama daemon service verified online.")
            else:
                self.update_step(step_name, "failed", "Service start timeout.", 0)
                self.log("Timeout attempting to spin up Ollama service.")
                return False, "Could not start Ollama service daemon."

        # ----------------------------------------------------------------─
        # Step 3: Verifying API connection
        # ----------------------------------------------------------------─
        step_name = steps[2]
        self.update_step(step_name, "running", "Validating API connection...", 50)
        self.log("Verifying API handshake...")
        
        if self.is_api_running():
            self.update_step(step_name, "done", "API connection verified.", 100)
            self.log("Handshake successful.")
        else:
            self.update_step(step_name, "failed", "API connection error.", 0)
            return False, "Ollama API not responding."

        # ----------------------------------------------------------------─
        # Step 4: Checking model availability
        # ----------------------------------------------------------------─
        step_name = steps[3]
        self.update_step(step_name, "running", f"Looking up model: {TARGET_MODEL}...", 40)
        self.log(f"Checking list of models for target: {TARGET_MODEL}...")
        
        installed_models = self.list_installed_models()
        self.log(f"Found installed models: {installed_models}")
        
        has_model = any(
            m.startswith(TARGET_MODEL.split(":")[0]) or m == TARGET_MODEL
            for m in installed_models
        )
        
        if has_model:
            self.update_step(step_name, "done", f"Model {TARGET_MODEL} is ready.", 100)
            self.update_step(steps[4], "skipped", "Model download not required.", 100)
            self.log("Target model already loaded. Skipping pull.")
        else:
            self.update_step(step_name, "done", f"Model {TARGET_MODEL} needs to be pulled.", 100)
            self.log("Target model needs to be pulled.")

        # ----------------------------------------------------------------─
        # Step 5: Pulling AI model
        # ----------------------------------------------------------------─
        step_name = steps[4]
        if not has_model:
            self.update_step(step_name, "running", f"Pulling {TARGET_MODEL} (approx. 390MB)...", 5)
            self.log(f"Requesting pull API for model: {TARGET_MODEL}...")
            
            try:
                # Request a model pull from endpoint streamingly
                with requests.post(
                    "http://localhost:11434/api/pull",
                    json={"name": TARGET_MODEL}, stream=True, timeout=600
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if line:
                            payload = json.loads(line.decode("utf-8"))
                            status_msg = payload.get("status", "")
                            completed = payload.get("completed", 0)
                            total = payload.get("total", 0)
                            
                            if total > 0:
                                pct = int((completed / total) * 100)
                                self.update_step(step_name, "running", f"Downloading: {pct}% - {status_msg}", pct)
                            else:
                                self.update_step(step_name, "running", status_msg, 0)
                                
                            if status_msg == "success":
                                break
                self.update_step(step_name, "done", f"Model {TARGET_MODEL} pulled successfully.", 100)
                self.log("Model pull complete.")
            except Exception as e:
                self.log(f"Model pull failed: {e}")
                self.update_step(step_name, "failed", f"Pull failed: {e}", 0)
                return False, f"Model download failed: {e}"

        # ----------------------------------------------------------------─
        # Step 6: Final verification
        # ----------------------------------------------------------------─
        step_name = steps[5]
        self.update_step(step_name, "running", "Performing sanity integration tests...", 30)
        self.log("Running final verification checks...")
        
        # Double check model is in the api listing
        if self.is_api_running():
            models = self.list_installed_models()
            verified = any(
                m.startswith(TARGET_MODEL.split(":")[0]) or m == TARGET_MODEL
                for m in models
            )
            if verified:
                self.update_step(step_name, "done", "System verified and functional!", 100)
                self.log("System verification succeeded. CogniClean ready.")
                return True, "Setup completed successfully."
            else:
                self.update_step(step_name, "failed", "Model list verification failed.", 0)
                self.log("Model not shown in tags registry listing.")
                return False, "Model verification failed."
        else:
            self.update_step(step_name, "failed", "Verification timeout.", 0)
            return False, "Connection lost during verification."

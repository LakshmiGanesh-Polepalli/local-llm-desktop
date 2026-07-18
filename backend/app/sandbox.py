import os
import uuid
import time
import subprocess
import tempfile
from typing import Dict, Any

def execute_local_code(language: str, code: str, timeout: float = 5.0) -> Dict[str, Any]:
    """
    Executes raw string code locally in a secure temporary environment.
    Supports 'python' and 'cpp'.
    """
    file_id = str(uuid.uuid4())[:8]
    
    # 🛡️ FIX 1: Route to OS Temp Directory to prevent Tauri Death Loops
    temp_dir = tempfile.gettempdir()
    
    if language.lower() in ["python", "py"]:
        filename = os.path.join(temp_dir, f"temp_{file_id}.py")
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(code)
            
        start_time = time.perf_counter()
        try:
            # 🛡️ FIX 2: Use "python" for Windows, not "python3"
            result = subprocess.run(
                ["python", filename],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            execution_time = (time.perf_counter() - start_time) * 1000
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time_ms": round(execution_time, 2)
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "", "stderr": f"[Security]: Execution timed out after {timeout}s.", "execution_time_ms": timeout * 1000}
        except FileNotFoundError:
            return {"success": False, "stdout": "", "stderr": "[System]: Python interpreter not found on system path.", "execution_time_ms": 0}
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    elif language.lower() in ["cpp", "c++"]:
        source_file = os.path.join(temp_dir, f"temp_{file_id}.cpp")
        # 🛡️ FIX 3: Windows uses .exe, not .out
        exe_file = os.path.join(temp_dir, f"temp_{file_id}.exe")
        
        with open(source_file, "w", encoding="utf-8") as f:
            f.write(code)
            
        # 1. Compile Phase
        try:
            compile_res = subprocess.run(["g++", "-O3", source_file, "-o", exe_file], capture_output=True, text=True)
            if compile_res.returncode != 0:
                if os.path.exists(source_file): os.remove(source_file)
                return {"success": False, "stdout": "", "stderr": f"Compilation Error:\n{compile_res.stderr}", "execution_time_ms": 0}
        except FileNotFoundError:
            # Catch missing C++ compiler so backend doesn't crash
            return {"success": False, "stdout": "", "stderr": "[System]: g++ compiler not found. Please install MinGW.", "execution_time_ms": 0}
            
        # 2. Run Phase
        start_time = time.perf_counter()
        try:
            # 🛡️ FIX 4: Absolute path execution (No Unix "./" needed)
            result = subprocess.run(
                [exe_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            execution_time = (time.perf_counter() - start_time) * 1000
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time_ms": round(execution_time, 2)
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "", "stderr": f"[Security]: Execution timed out after {timeout}s.", "execution_time_ms": timeout * 1000}
        finally:
            # Clean up both source and executable
            if os.path.exists(source_file): os.remove(source_file)
            if os.path.exists(exe_file): os.remove(exe_file)
            
    return {"success": False, "stdout": "", "stderr": f"Unsupported language: {language}", "execution_time_ms": 0}

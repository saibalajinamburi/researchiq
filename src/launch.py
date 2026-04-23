import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
RUNTIME_DIR = ROOT_DIR / ".runtime"
LOG_DIR = RUNTIME_DIR / "logs"
SERVICES_PATH = RUNTIME_DIR / "services.json"


def free_port(preferred: int) -> int:
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No free port found near {preferred}")


def start_service(name: str, command: list[str], env: dict[str, str] | None = None) -> dict[str, object]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stdout_path = LOG_DIR / f"{name}.out.log"
    stderr_path = LOG_DIR / f"{name}.err.log"
    stdout = stdout_path.open("w", encoding="utf-8")
    stderr = stderr_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=ROOT_DIR,
        stdout=stdout,
        stderr=stderr,
        env=env,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    return {
        "name": name,
        "pid": process.pid,
        "command": command,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_for_port(name: str, port: int, timeout_seconds: int = 30) -> bool:
    started = time.time()
    while time.time() - started < timeout_seconds:
        if port_is_open(port):
            print(f"{name} is listening on port {port}")
            return True
        time.sleep(1)
    print(f"{name} did not open port {port} within {timeout_seconds}s")
    return False


def main() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    api_port = free_port(8000)
    streamlit_port = free_port(8501)
    mlflow_port = free_port(5000)

    api_url = f"http://127.0.0.1:{api_port}"
    streamlit_url = f"http://127.0.0.1:{streamlit_port}"
    mlflow_url = f"http://127.0.0.1:{mlflow_port}"

    env = dict(os.environ)
    env["RESEARCHIQ_API_URL"] = api_url
    env["RESEARCHIQ_MLFLOW_URL"] = mlflow_url

    services = []
    services.append(
        start_service(
            "api",
            [
                sys.executable,
                "-m",
                "uvicorn",
                "src.api:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(api_port),
            ],
            env=env,
        )
    )
    wait_for_port("api", api_port)

    services.append(
        start_service(
            "streamlit",
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "src/streamlit_app.py",
                "--server.address",
                "127.0.0.1",
                "--server.port",
                str(streamlit_port),
                "--server.headless",
                "true",
            ],
            env=env,
        )
    )
    wait_for_port("streamlit", streamlit_port)

    services.append(
        start_service(
            "mlflow",
            [
                sys.executable,
                "-m",
                "mlflow",
                "ui",
                "--backend-store-uri",
                (ROOT_DIR / "mlruns").as_uri(),
                "--default-artifact-root",
                (ROOT_DIR / "mlruns").as_uri(),
                "--host",
                "127.0.0.1",
                "--port",
                str(mlflow_port),
            ],
            env=env,
        )
    )
    wait_for_port("mlflow", mlflow_port)

    links = {
        "streamlit": streamlit_url,
        "api": api_url,
        "api_docs": f"{api_url}/docs",
        "metrics": f"{api_url}/metrics",
        "health": f"{api_url}/health",
        "mlflow": mlflow_url,
    }
    payload = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "links": links,
        "services": services,
    }
    SERVICES_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("ResearchIQ services are starting.")
    for label, url in links.items():
        print(f"{label}: {url}")
    print(f"service registry: {SERVICES_PATH}")
    print(f"logs: {LOG_DIR}")


if __name__ == "__main__":
    main()

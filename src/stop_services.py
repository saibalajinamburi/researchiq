import json
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
SERVICES_PATH = ROOT_DIR / ".runtime" / "services.json"


def stop_pid(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        subprocess.run(
            ["kill", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def main() -> None:
    if not SERVICES_PATH.exists():
        print("No service registry found.")
        return

    payload = json.loads(SERVICES_PATH.read_text(encoding="utf-8"))
    for service in payload.get("services", []):
        pid = service.get("pid")
        name = service.get("name", "service")
        if pid:
            stop_pid(int(pid))
            print(f"Stopped {name} pid={pid}")


if __name__ == "__main__":
    main()

FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends patchelf && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user

ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GLIBC_TUNABLES="glibc.rtld.execstack=2" \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "import glob, os, site, subprocess; libs = glob.glob(os.path.join(site.getusersitepackages(), 'onnxruntime', 'capi', 'onnxruntime_pybind11_state*.so')); [subprocess.run(['patchelf', '--clear-execstack', lib], check=True) for lib in libs]"

COPY --chown=user app.py app.py
COPY --chown=user src src
COPY --chown=user models models
COPY --chown=user reports reports
COPY --chown=user README.md README.md

EXPOSE 7860

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "7860", "--server.headless", "true"]

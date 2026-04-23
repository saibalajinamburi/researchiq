import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

try:
    from onnxruntime.quantization import QuantType, quantize_dynamic
except Exception:
    QuantType = None
    quantize_dynamic = None


def export_onnx(
    model_path: str,
    output_dir: str,
    n_features: int,
    sample_size: int,
    seed: int,
) -> None:
    root_dir = Path(__file__).resolve().parent.parent
    model_file = root_dir / model_path
    export_dir = root_dir / output_dir
    processed_dir = root_dir / "data" / "processed"
    export_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = export_dir / "best_model.onnx"
    int8_path = export_dir / "best_model_int8.onnx"
    metadata_path = export_dir / "phase4_metadata.json"

    print(f"Loading sklearn model: {model_file}", flush=True)
    model = joblib.load(model_file)

    print("Converting sklearn pipeline to ONNX", flush=True)
    initial_types = [("input", FloatTensorType([None, n_features]))]
    classifier = getattr(model, "named_steps", {}).get("classifier")
    options = {id(classifier): {"zipmap": False}} if classifier is not None else None
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_types,
        target_opset=15,
        options=options,
    )
    onnx.save_model(onnx_model, onnx_path)
    print(f"Saved ONNX model: {onnx_path}", flush=True)

    quantized = False
    quantization_error = None
    if quantize_dynamic is not None:
        try:
            print("Attempting ONNX Runtime dynamic INT8 quantization", flush=True)
            quantize_dynamic(
                model_input=str(onnx_path),
                model_output=str(int8_path),
                weight_type=QuantType.QInt8,
            )
            quantized = int8_path.exists()
            print(f"Saved INT8 ONNX model: {int8_path}", flush=True)
        except Exception as exc:
            quantization_error = str(exc)
            print(f"INT8 quantization skipped: {quantization_error}", flush=True)
    else:
        quantization_error = "onnxruntime.quantization is unavailable"
        print(f"INT8 quantization skipped: {quantization_error}", flush=True)

    print("Verifying ONNX predictions against sklearn", flush=True)
    rng = np.random.default_rng(seed)
    X = np.load(processed_dir / "X.npy", mmap_mode="r")
    sample_size = min(sample_size, X.shape[0])
    sample_indices = rng.choice(X.shape[0], size=sample_size, replace=False)
    X_sample = np.asarray(X[sample_indices], dtype=np.float32)

    sklearn_preds = model.predict(X_sample)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_outputs = session.run(None, {input_name: X_sample})
    onnx_preds = np.asarray(onnx_outputs[0]).reshape(-1)
    agreement = float(np.mean(sklearn_preds == onnx_preds))

    int8_agreement = None
    if quantized:
        int8_session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
        int8_outputs = int8_session.run(None, {input_name: X_sample})
        int8_preds = np.asarray(int8_outputs[0]).reshape(-1)
        int8_agreement = float(np.mean(sklearn_preds == int8_preds))

    metadata = {
        "source_model": str(model_file.relative_to(root_dir)),
        "onnx_model": str(onnx_path.relative_to(root_dir)),
        "int8_model": str(int8_path.relative_to(root_dir)) if quantized else None,
        "quantized": quantized,
        "quantization_error": quantization_error,
        "n_features": n_features,
        "verification_sample_size": sample_size,
        "onnx_sklearn_label_agreement": agreement,
        "int8_sklearn_label_agreement": int8_agreement,
        "onnxruntime_providers": ort.get_available_providers(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Verification agreement: ONNX={agreement:.4f}", flush=True)
    if int8_agreement is not None:
        print(f"Verification agreement: INT8={int8_agreement:.4f}", flush=True)
    print(f"Saved metadata: {metadata_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export the Phase 3 sklearn winner to ONNX.")
    parser.add_argument(
        "--model-path",
        default="models/phase3_final/best_model.joblib",
        help="Path to the trained sklearn/joblib model relative to repo root.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/phase4_onnx",
        help="Output directory for ONNX artifacts relative to repo root.",
    )
    parser.add_argument("--n-features", type=int, default=384, help="Embedding feature dimension.")
    parser.add_argument("--sample-size", type=int, default=1000, help="Rows used for export verification.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for verification sampling.")
    args = parser.parse_args()
    export_onnx(
        model_path=args.model_path,
        output_dir=args.output_dir,
        n_features=args.n_features,
        sample_size=args.sample_size,
        seed=args.seed,
    )

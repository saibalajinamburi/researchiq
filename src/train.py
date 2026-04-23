import argparse
import csv
import json
import os
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    import dagshub
except Exception:
    dagshub = None


def log_step(message: str, start_time: float | None = None) -> float:
    now = time.time()
    elapsed = ""
    if start_time is not None:
        elapsed = f" | elapsed={now - start_time:.1f}s"
    print(f"[{time.strftime('%H:%M:%S')}] {message}{elapsed}", flush=True)
    return now


def write_status(root_dir: Path, status: dict) -> None:
    status_path = root_dir / "reports" / "training_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_payload = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **status,
    }
    status_path.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")


def configure_mlflow(
    experiment_name: str,
    use_dagshub: bool = False,
    dagshub_owner: str = "",
    dagshub_repo: str = "",
    allow_local_fallback: bool = True,
) -> None:
    if use_dagshub:
        try:
            if dagshub is None:
                raise RuntimeError(
                    "DagsHub logging requested but `dagshub` package is missing. "
                    "Install with: pip install dagshub"
                )
            if not dagshub_owner or not dagshub_repo:
                raise RuntimeError(
                    "DagsHub logging requested but owner/repo is missing. "
                    "Provide --dagshub-owner and --dagshub-repo."
                )
            dagshub.init(repo_owner=dagshub_owner, repo_name=dagshub_repo, mlflow=True)
        except Exception as exc:
            if not allow_local_fallback:
                raise
            print(
                "DagsHub MLflow initialization failed; continuing with local MLflow. "
                f"Reason: {exc}"
            )

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def build_model_specs(
    seed: int,
    n_classes: int,
    selected_models: list[str],
    n_jobs: int,
    accelerator: str,
    lightgbm_gpu: bool,
) -> dict:
    specs = {}
    selected = set(selected_models)

    if "logreg_ovr" in selected:
        specs["logreg_ovr"] = {
            "estimator": Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=2000,
                            tol=1e-3,
                            class_weight="balanced",
                            solver="lbfgs",
                            n_jobs=n_jobs,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            "param_distributions": {
                "classifier__C": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            },
        }

    if "random_forest" in selected:
        specs["random_forest"] = {
            "estimator": RandomForestClassifier(
                class_weight="balanced_subsample",
                n_jobs=n_jobs,
                random_state=seed,
            ),
            "param_distributions": {
                "n_estimators": [300, 500, 700, 900],
                "max_depth": [None, 20, 30, 40, 50],
                "min_samples_split": [2, 4, 8],
                "min_samples_leaf": [1, 2, 4],
            },
        }

    if "extra_trees" in selected:
        specs["extra_trees"] = {
            "estimator": ExtraTreesClassifier(
                class_weight="balanced",
                n_jobs=n_jobs,
                random_state=seed,
            ),
            "param_distributions": {
                "n_estimators": [300, 500, 700, 900],
                "max_depth": [None, 20, 30, 40, 50],
                "min_samples_split": [2, 4, 8],
                "min_samples_leaf": [1, 2, 4],
            },
        }

    if "xgboost" in selected:
        if XGBClassifier is None:
            raise RuntimeError(
                "xgboost model requested but package is missing. Install with: pip install xgboost"
            )
        xgboost_kwargs = {}
        if accelerator == "cuda":
            xgboost_kwargs["device"] = "cuda"
        specs["xgboost"] = {
            "estimator": XGBClassifier(
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
                tree_method="hist",
                n_jobs=n_jobs,
                random_state=seed,
                **xgboost_kwargs,
            ),
            "param_distributions": {
                "n_estimators": [300, 500, 700],
                "max_depth": [6, 8, 10],
                "learning_rate": [0.03, 0.05, 0.1],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
            },
        }

    if "lightgbm" in selected:
        if LGBMClassifier is None:
            raise RuntimeError(
                "lightgbm model requested but package is missing. Install with: pip install lightgbm"
            )
        lightgbm_kwargs = {}
        if accelerator == "cuda" and lightgbm_gpu:
            lightgbm_kwargs["device_type"] = "gpu"
        specs["lightgbm"] = {
            "estimator": LGBMClassifier(
                objective="multiclass",
                num_class=n_classes,
                class_weight="balanced",
                n_jobs=n_jobs,
                random_state=seed,
                verbose=-1,
                **lightgbm_kwargs,
            ),
            "param_distributions": {
                "n_estimators": [300, 500, 700],
                "num_leaves": [31, 63, 127],
                "max_depth": [-1, 10, 20, 30],
                "learning_rate": [0.03, 0.05, 0.1],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
            },
        }

    if not specs:
        raise RuntimeError("No models selected. Pass --models with at least one valid model key.")

    return specs


def build_quick_model(model, model_name: str):
    if model_name == "logreg_ovr":
        model.set_params(classifier__C=2.0)
    elif model_name in {"random_forest", "extra_trees"}:
        model.set_params(
            n_estimators=300,
            max_depth=30,
            min_samples_split=4,
            min_samples_leaf=2,
        )
    elif model_name == "xgboost":
        model.set_params(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
        )
    elif model_name == "lightgbm":
        model.set_params(
            n_estimators=300,
            num_leaves=63,
            max_depth=20,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
        )
    return model


def model_params(model) -> dict:
    params = model.get_params(deep=False)
    return {
        key: value
        for key, value in params.items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }


def run_training(
    test_size: float,
    seed: int,
    experiment_name: str,
    n_iter: int,
    cv_folds: int,
    selected_models: list[str],
    use_dagshub: bool = False,
    dagshub_owner: str = "",
    dagshub_repo: str = "",
    allow_local_fallback: bool = True,
    output_dir: str = "models",
    summary_path: str = "reports/model_comparison.csv",
    best_model_name: str = "ResearchIQBestClassifier",
    sample_size: int | None = None,
    n_jobs: int = 1,
    accelerator: str = "cpu",
    lightgbm_gpu: bool = False,
    quick_final: bool = False,
) -> None:
    run_started = log_step("Starting ResearchIQ training")
    root_dir = Path(__file__).resolve().parent.parent
    write_status(root_dir, {"stage": "starting", "experiment": experiment_name})
    processed_dir = root_dir / "data" / "processed"
    model_dir = root_dir / output_dir
    comparison_path = root_dir / summary_path
    cache_dir = root_dir / ".cache" / "boost_compute"
    model_dir.mkdir(parents=True, exist_ok=True)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("BOOST_COMPUTE_CACHE_PATH", str(cache_dir))

    log_step("Loading processed arrays")
    write_status(root_dir, {"stage": "loading_data", "experiment": experiment_name})
    X = np.load(processed_dir / "X.npy")
    y = np.load(processed_dir / "y.npy")
    classes = np.load(processed_dir / "classes.npy", allow_pickle=True)
    log_step(f"Loaded X={X.shape}, y={y.shape}, classes={len(classes)}")

    if sample_size is not None:
        log_step(f"Building stratified sample with {sample_size} rows")
        if sample_size < len(classes) * 2:
            raise ValueError("sample_size must leave at least two examples per class.")
        _, X, _, y = train_test_split(
            X,
            y,
            test_size=sample_size,
            random_state=seed,
            stratify=y,
        )
        log_step(f"Sample ready: X={X.shape}, y={y.shape}")

    log_step(f"Creating train/test split with test_size={test_size}")
    write_status(root_dir, {"stage": "splitting_data", "experiment": experiment_name, "rows": int(X.shape[0])})
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    log_step(f"Split ready: train={X_train.shape[0]}, test={X_test.shape[0]}")

    log_step(f"Configuring MLflow experiment: {experiment_name}")
    configure_mlflow(
        experiment_name=experiment_name,
        use_dagshub=use_dagshub,
        dagshub_owner=dagshub_owner,
        dagshub_repo=dagshub_repo,
        allow_local_fallback=allow_local_fallback,
    )
    log_step("Building model specifications")
    model_specs = build_model_specs(
        seed=seed,
        n_classes=len(classes),
        selected_models=selected_models,
        n_jobs=n_jobs,
        accelerator=accelerator,
        lightgbm_gpu=lightgbm_gpu,
    )
    best_name = None
    best_f1 = -1.0
    best_run_id = None
    best_estimator = None
    results = []

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    total_models = len(model_specs)
    for model_index, (model_name, spec) in enumerate(model_specs.items(), start=1):
        model_started = log_step(f"Model {model_index}/{total_models} START: {model_name}")
        write_status(
            root_dir,
            {
                "stage": "training_model",
                "experiment": experiment_name,
                "model": model_name,
                "model_index": model_index,
                "total_models": total_models,
                "quick_final": quick_final,
            },
        )
        model = spec["estimator"]
        param_distributions = spec["param_distributions"]
        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("quick_final", quick_final)
            try:
                if quick_final:
                    log_step(f"[{model_name}] fitting fixed quick-final config")
                    current_best_model = build_quick_model(model, model_name)
                    current_best_model.fit(X_train, y_train)
                    cv_best_score = float("nan")
                    best_params = model_params(current_best_model)
                else:
                    total_fits = n_iter * cv_folds
                    log_step(f"[{model_name}] searching {n_iter} configs x {cv_folds} folds = {total_fits} fits")
                    search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_distributions,
                        n_iter=n_iter,
                        scoring="f1_macro",
                        cv=cv,
                        n_jobs=n_jobs,
                        random_state=seed,
                        verbose=2,
                    )
                    search.fit(X_train, y_train)
                    current_best_model = search.best_estimator_
                    cv_best_score = float(search.best_score_)
                    best_params = search.best_params_
                mlflow.set_tag("accelerator_effective", accelerator)
            except Exception as exc:
                if accelerator != "cuda" or model_name not in {"xgboost", "lightgbm"}:
                    raise
                print(
                    f"[{model_name}] CUDA fit failed; retrying this model on CPU. "
                    f"Reason: {exc}"
                )
                mlflow.set_tag("accelerator_effective", "cpu_fallback")
                mlflow.set_tag("accelerator_fallback_reason", str(exc)[:500])
                if model_name == "xgboost":
                    model.set_params(device="cpu")
                elif model_name == "lightgbm":
                    model.set_params(device_type="cpu")
                if quick_final:
                    current_best_model = build_quick_model(model, model_name)
                    current_best_model.fit(X_train, y_train)
                    cv_best_score = float("nan")
                    best_params = model_params(current_best_model)
                else:
                    search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_distributions,
                        n_iter=n_iter,
                        scoring="f1_macro",
                        cv=cv,
                        n_jobs=n_jobs,
                        random_state=seed,
                        verbose=2,
                    )
                    search.fit(X_train, y_train)
                    current_best_model = search.best_estimator_
                    cv_best_score = float(search.best_score_)
                    best_params = search.best_params_

            log_step(f"[{model_name}] predicting test split")
            preds = current_best_model.predict(X_test)
            macro_f1 = f1_score(y_test, preds, average="macro")
            log_step(f"[{model_name}] scoring complete: macro_f1={macro_f1:.4f}")
            write_status(
                root_dir,
                {
                    "stage": "scored_model",
                    "experiment": experiment_name,
                    "model": model_name,
                    "macro_f1": float(macro_f1),
                    "model_index": model_index,
                    "total_models": total_models,
                },
            )

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("seed", seed)
            mlflow.log_param("n_iter", n_iter)
            mlflow.log_param("cv_folds", cv_folds)
            mlflow.log_param("n_jobs", n_jobs)
            mlflow.log_param("accelerator", accelerator)
            mlflow.log_param("lightgbm_gpu", lightgbm_gpu)
            mlflow.log_param("selected_models", ",".join(selected_models))
            mlflow.log_param("n_samples", int(X.shape[0]))
            mlflow.log_param("n_features", int(X.shape[1]))
            mlflow.log_param("n_classes", int(len(classes)))
            mlflow.log_param("sample_size", sample_size if sample_size is not None else "full")
            mlflow.log_param("use_dagshub", use_dagshub)
            mlflow.log_param("dagshub_owner", dagshub_owner if dagshub_owner else "none")
            mlflow.log_param("dagshub_repo", dagshub_repo if dagshub_repo else "none")
            mlflow.log_param("best_model_name", best_model_name)
            for k, v in best_params.items():
                mlflow.log_param(f"best_param__{k}", v)
            if not np.isnan(cv_best_score):
                mlflow.log_metric("cv_best_macro_f1", cv_best_score)
            mlflow.log_metric("macro_f1", float(macro_f1))
            log_step(f"[{model_name}] logging artifacts to MLflow")
            mlflow.log_text(
                classification_report(
                    y_test,
                    preds,
                    target_names=classes.astype(str),
                    zero_division=0,
                ),
                f"{model_name}_classification_report.txt",
            )
            mlflow.log_text(str(best_params), f"{model_name}_best_params.txt")
            mlflow.sklearn.log_model(current_best_model, artifact_path=f"{model_name}_model")
            mlflow.set_tag("candidate_model", "true")

            run_id = mlflow.active_run().info.run_id
            results.append(
                {
                    "model_name": model_name,
                    "run_id": run_id,
                    "cv_best_macro_f1": cv_best_score,
                    "test_macro_f1": float(macro_f1),
                    "best_params": json.dumps(best_params, sort_keys=True),
                }
            )

            cv_text = "n/a" if np.isnan(cv_best_score) else f"{cv_best_score:.4f}"
            print(
                f"[{model_name}] cv_best_macro_f1={cv_text} | "
                f"test_macro_f1={macro_f1:.4f}",
                flush=True,
            )

            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_name = model_name
                best_run_id = run_id
                best_estimator = current_best_model
        log_step(f"Model {model_index}/{total_models} DONE: {model_name}", model_started)

    results = sorted(results, key=lambda item: item["test_macro_f1"], reverse=True)
    log_step(f"Writing comparison CSV: {comparison_path}")
    write_status(root_dir, {"stage": "writing_comparison", "experiment": experiment_name, "best_so_far": best_name})
    with comparison_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_name",
                "run_id",
                "cv_best_macro_f1",
                "test_macro_f1",
                "best_params",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    if best_estimator is not None:
        log_step(f"Exporting best model: {best_name}")
        write_status(root_dir, {"stage": "exporting_best_model", "experiment": experiment_name, "best_model": best_name})
        best_model_path = model_dir / "best_model.joblib"
        metadata_path = model_dir / "best_model_metadata.json"
        joblib.dump(best_estimator, best_model_path, compress=3)
        metadata = {
            "model_name": best_name,
            "run_id": best_run_id,
            "macro_f1": best_f1,
            "classes": classes.astype(str).tolist(),
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "sample_size": sample_size if sample_size is not None else "full",
            "source_experiment": experiment_name,
            "comparison_path": str(comparison_path.relative_to(root_dir)),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        with mlflow.start_run(run_name="best_model_export"):
            log_step("Logging best-model export run to MLflow")
            mlflow.log_param("best_model_name", best_model_name)
            mlflow.log_param("source_model_name", best_name)
            mlflow.log_param("source_run_id", best_run_id)
            mlflow.log_metric("best_macro_f1", float(best_f1))
            mlflow.log_artifact(str(comparison_path))
            mlflow.log_artifact(str(best_model_path))
            mlflow.log_artifact(str(metadata_path))
            mlflow.sklearn.log_model(best_estimator, artifact_path="best_model")
            mlflow.set_tag("best_model", "true")

    print(f"\nBest model: {best_name} | macro_f1={best_f1:.4f}")
    print(f"Comparison saved to: {comparison_path}")
    print(f"Best model exported to: {model_dir / 'best_model.joblib'}")
    log_step("Training finished", run_started)
    write_status(
        root_dir,
        {
            "stage": "finished",
            "experiment": experiment_name,
            "best_model": best_name,
            "best_macro_f1": float(best_f1),
            "comparison_path": str(comparison_path),
            "model_dir": str(model_dir),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multiple models with hyperparameter search.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--n-iter", type=int, default=10, help="RandomizedSearchCV iterations per model.")
    parser.add_argument("--cv-folds", type=int, default=3, help="Cross-validation folds.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logreg_ovr", "random_forest", "extra_trees", "xgboost", "lightgbm"],
        choices=["logreg_ovr", "random_forest", "extra_trees", "xgboost", "lightgbm"],
        help="Models to train and compare.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="researchiq_phase3",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--use-dagshub",
        action="store_true",
        help="Enable DagsHub-backed MLflow tracking.",
    )
    parser.add_argument(
        "--dagshub-owner",
        type=str,
        default="bsai38337",
        help="DagsHub repo owner/username.",
    )
    parser.add_argument(
        "--dagshub-repo",
        type=str,
        default="my-first-repo",
        help="DagsHub repo name.",
    )
    parser.add_argument(
        "--strict-dagshub",
        action="store_true",
        help="Fail instead of falling back to local MLflow if DagsHub initialization fails.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory for exported best model artifacts.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default="reports/model_comparison.csv",
        help="CSV path for model comparison results.",
    )
    parser.add_argument(
        "--best-model-name",
        type=str,
        default="ResearchIQBestClassifier",
        help="Logical name for the exported winning model.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional stratified sample size for smoke tests. Omit for full training.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers for estimators and RandomizedSearchCV. Use -1 outside restricted shells.",
    )
    parser.add_argument(
        "--accelerator",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Use CUDA-capable estimators where supported. Logistic/RF/ExtraTrees remain CPU models.",
    )
    parser.add_argument(
        "--lightgbm-gpu",
        action="store_true",
        help="Also try LightGBM's OpenCL GPU backend. Off by default because many Windows wheels are CPU-only or cache-restricted.",
    )
    parser.add_argument(
        "--quick-final",
        action="store_true",
        help="Train each selected model once with fixed strong defaults instead of running RandomizedSearchCV.",
    )
    args = parser.parse_args()
    run_training(
        test_size=args.test_size,
        seed=args.seed,
        experiment_name=args.experiment_name,
        n_iter=args.n_iter,
        cv_folds=args.cv_folds,
        selected_models=args.models,
        use_dagshub=args.use_dagshub,
        dagshub_owner=args.dagshub_owner,
        dagshub_repo=args.dagshub_repo,
        allow_local_fallback=not args.strict_dagshub,
        output_dir=args.output_dir,
        summary_path=args.summary_path,
        best_model_name=args.best_model_name,
        sample_size=args.sample_size,
        n_jobs=args.n_jobs,
        accelerator=args.accelerator,
        lightgbm_gpu=args.lightgbm_gpu,
        quick_final=args.quick_final,
    )

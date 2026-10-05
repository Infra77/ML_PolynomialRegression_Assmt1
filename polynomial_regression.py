import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.utils.parallel import Parallel, delayed
warnings.filterwarnings("ignore")

ROLL = "BT2024015"
DATA_DIR = "."
OUT_DIR = "."

MAX_DEGREE = {1: 10, 2: 20}

MODEL_NAMES = ["linear", "ridge", "lasso", "elasticnet"]

SEED = 42
N_FOLDS = 5
N_WORKERS = 4
N_JOBS = 1


# Hyper-parameter search spaces
# Ridge: 10^-4, 10^-3, ..., 10^4
RIDGE_ALPHAS = [10**i for i in range(-4, 5)]

N_PATH_ALPHAS = 60
ENET_L1_RATIOS = [0.05, 0.2, 0.5, 0.8, 0.95]

MAX_ITER = 50000
TOL = 1e-3

# Same folds are used for every model
KFOLD = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


# 1. Data Loading
def load_problem(problem_id):
    train_path = os.path.join(DATA_DIR, f"{ROLL}_train_var{problem_id}.csv")
    test_path = os.path.join(DATA_DIR, f"{ROLL}_test_var{problem_id}.csv")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    feature_names = [c for c in train.columns if c != "y"]

    X_train = train[feature_names].to_numpy(dtype=float)
    y_train = train["y"].to_numpy(dtype=float)

    X_test = test[feature_names].to_numpy(dtype=float)

    return X_train, y_train, X_test, feature_names


# 2. Pipeline construction
def build_pipeline(estimator, degree):
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree, include_bias=False),
        StandardScaler(),
        estimator,
    )


def count_poly_features(n_inputs, degree):
    dummy = np.zeros((2, n_inputs))
    return PolynomialFeatures(degree, include_bias=False).fit(dummy).n_output_features_


# 3. CV estimator helper
def make_cv_estimator(cls, **kwargs):
    try:
        return cls(alphas=N_PATH_ALPHAS, **kwargs)
    except (TypeError, ValueError):
        return cls(n_alphas=N_PATH_ALPHAS, **kwargs)


# 4. Model tuners
def tune_linear(X, y, degree):
    pipe = build_pipeline(LinearRegression(), degree)
    
    scores = cross_val_score(
        pipe, X, y, cv=KFOLD, scoring="neg_mean_squared_error", n_jobs=N_JOBS
    )
    pipe.fit(X, y)

    return (-scores.mean(), pipe, {})


def tune_ridge(X, y, degree):
    search = GridSearchCV(
        build_pipeline(Ridge(), degree),
        param_grid={"ridge__alpha": RIDGE_ALPHAS},
        cv=KFOLD,
        scoring="neg_mean_squared_error",
        n_jobs=N_JOBS,
    )
    search.fit(X, y)

    hyper = {"alpha": float(search.best_params_["ridge__alpha"])}
    return (-search.best_score_, search.best_estimator_, hyper)


def tune_lasso(X, y, degree):
    estimator = make_cv_estimator(
        LassoCV, cv=KFOLD, max_iter=MAX_ITER, tol=TOL, n_jobs=N_JOBS, random_state=SEED
    )
    pipe = build_pipeline(estimator, degree)
    pipe.fit(X, y)
    model = pipe[-1]
    cv_mse = model.mse_path_.mean(axis=-1).min()
    hyper = {"alpha": float(model.alpha_), "nonzero": int((model.coef_ != 0).sum())}
    return (cv_mse, pipe, hyper)


def tune_elasticnet(X, y, degree):
    estimator = make_cv_estimator(
        ElasticNetCV,
        cv=KFOLD,
        l1_ratio=ENET_L1_RATIOS,
        max_iter=MAX_ITER,
        tol=TOL,
        n_jobs=N_JOBS,
        random_state=SEED,
    )
    pipe = build_pipeline(estimator, degree)
    pipe.fit(X, y)
    model = pipe[-1]
    cv_mse = model.mse_path_.mean(axis=-1).min()
    hyper = {
        "alpha": float(model.alpha_),
        "l1_ratio": float(model.l1_ratio_),
        "nonzero": int((model.coef_ != 0).sum()),
    }
    return (cv_mse, pipe, hyper)


TUNERS = {
    "linear": tune_linear,
    "ridge": tune_ridge,
    "lasso": tune_lasso,
    "elasticnet": tune_elasticnet,
}


# 5. Search over (degree, model)
def evaluate_config(X, y, degree, model_name):
    start = time.time()
    cv_mse, pipeline, hyper = TUNERS[model_name](X, y, degree)
    return {
        "degree": degree,
        "model": model_name,
        "n_features": count_poly_features(X.shape[1], degree),
        "cv_mse": float(cv_mse),
        "cv_r2": float(1.0 - cv_mse / y.var()),
        "hyper": hyper,
        "seconds": time.time() - start,
        "pipeline": pipeline,
    }


def run_search(X, y, max_degree):
    results = []
    with Parallel(n_jobs=N_WORKERS, backend="threading") as parallel:
        for degree in range(1, max_degree + 1):
            batch = parallel(
                delayed(evaluate_config)(X, y, degree, model_name)
                for model_name in MODEL_NAMES
            )
            for res in batch:
                print(
                    f"deg {res['degree']:2d} "
                    f"{res['model']:10s} "
                    f"feats={res['n_features']:5d} "
                    f"cv_mse={res['cv_mse']:.6g} "
                    f"cv_r2={res['cv_r2']:.5f} "
                    f"{format_hyper(res['hyper'])} "
                    f"({res['seconds']:.1f}s)",
                    flush=True,
                )
            results.extend(batch)
    return results


# 6. Model selection
def select_best(results):
    best_mse = min(r["cv_mse"] for r in results)
    threshold = best_mse * 1.02

    eligible = [r for r in results if r["cv_mse"] <= threshold]

    best = min(eligible, key=lambda x: x["degree"])
    return best


def best_per_model(results):
    return {
        name: min((r for r in results if r["model"] == name), key=lambda r: r["cv_mse"])
        for name in MODEL_NAMES
    }


# 7. Reporting helpers
def format_hyper(hyper):
    parts = []
    for key, value in hyper.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4g}")
        else:
            parts.append(f"{key}={value}")
    return "{" + ", ".join(parts) + "}"


def results_to_frame(results):
    rows = []
    for r in results:
        row = {k: r[k] for k in ("degree", "model", "n_features", "cv_mse", "cv_r2")}
        row.update(r["hyper"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cv_mse").reset_index(drop=True)


def print_top_configs(results, k=10):
    table = results_to_frame(results).head(k)
    print(f"\nTop {k} configs by CV MSE:")
    print(table.to_string())


def print_best_per_model(results):
    print("\nBest degree for each regressor:")
    for name, r in best_per_model(results).items():
        print(
            f"  {name:10s} "
            f"degree={r['degree']:2d}  "
            f"cv_mse={r['cv_mse']:.6g}  "
            f"cv_r2={r['cv_r2']:.5f}  "
            f"{format_hyper(r['hyper'])}"
        )


def print_best(problem_id, best):
    print("\n" + "=" * 78)
    print(f"BEST MODEL FOR var{problem_id}")
    print("=" * 78)
    print(f"  model        : {best['model']}")
    print(f"  degree       : {best['degree']}")
    print(f"  hyper-params : {format_hyper(best['hyper'])}")
    print(f"  CV MSE       : {best['cv_mse']:.6g}")
    print(f"  CV R2        : {best['cv_r2']:.5f}")


def print_train_metrics(pipeline, X, y):
    pred = pipeline.predict(X)
    print(
        f"  train MSE    : "
        f"{mean_squared_error(y, pred):.6g}\n"
        f"  train R2     : "
        f"{r2_score(y, pred):.5f}"
    )


# 8. Optional results table
def save_results_table(problem_id, results):
    path = os.path.join(OUT_DIR, f"results_var{problem_id}.csv")
    results_to_frame(results).to_csv(path, index=False)
    print(f"saved {path}")


# 9. Optional CV plot
def save_cv_plot(problem_id, results):
    plt.figure(figsize=(7, 4.5))
    for name in MODEL_NAMES:
        rows = sorted(
            (r for r in results if r["model"] == name), key=lambda r: r["degree"]
        )
        plt.plot(
            [r["degree"] for r in rows],
            [r["cv_mse"] for r in rows],
            marker="o",
            label=name,
        )
    plt.yscale("log")
    plt.xlabel("polynomial degree")
    plt.ylabel(f"{N_FOLDS}-fold CV MSE")
    plt.title(f"var{problem_id}: CV MSE by degree and regressor")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"cv_curve_var{problem_id}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"saved {path}")


# 10. Prediction file
def write_submission(problem_id, pipeline, X_test):
    predictions = pipeline.predict(X_test)
    submission = pd.DataFrame({"y": predictions})
    path = os.path.join(OUT_DIR, f"{ROLL}_pred_var{problem_id}.csv")
    submission.to_csv(path, index=False)
    print(
        f"  saved {path} ({len(submission)} rows, columns={list(submission.columns)})"
    )


# 11. One full problem
def solve_problem(problem_id):
    X_train, y_train, X_test, feature_names = load_problem(problem_id)
    print(
        f"\n{'#' * 78}\n"
        f"# var{problem_id}: "
        f"train {X_train.shape}, "
        f"test {X_test.shape}, "
        f"features {feature_names}, "
        f"degrees 1-{MAX_DEGREE[problem_id]}"
        f"\n{'#' * 78}",
        flush=True,
    )
    started = time.time()
    # Search every degree and every model
    results = run_search(X_train, y_train, MAX_DEGREE[problem_id])

    # Choose the overall best configuration
    best = select_best(results)

    print_top_configs(results)

    print_best_per_model(results)

    print_best(problem_id, best)

    print_train_metrics(best["pipeline"], X_train, y_train)

    # Predict test data using the best
    # pipeline, which has already been
    # fitted on the full training data
    write_submission(problem_id, best["pipeline"], X_test)
    print(f"  var{problem_id} finished in {(time.time() - started) / 60:.1f} min")


# 12. Main
def main():
    solve_problem(1)
    solve_problem(2)

if __name__ == "__main__":
    main()

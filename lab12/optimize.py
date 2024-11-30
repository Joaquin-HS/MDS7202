import os
import pickle
import optuna
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
import matplotlib.pyplot as plt
import xgboost as xgb

def optimize_model():
    # Se crean los directorios para los artefactos
    os.makedirs("plots", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Se cargan los datos
    data = pd.read_csv("water_potability.csv")
    X = data.drop("Potability", axis=1)
    y = data["Potability"]

    # Se dividen los datos en entrenamiento y validación
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(trial):
        # Grilla de parámetros del modelo
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }

        # Nombre único para el experimento y el run
        experiment_name = f"XGBoost Experiment: lr_{params['learning_rate']:.3f}"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"Run max_depth={params['max_depth']}"):
            # Se entrena el modelo
            model = XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss")
            model.fit(X_train, y_train)

            # Se valida el modelo
            y_pred = model.predict(X_valid)
            f1 = f1_score(y_valid, y_pred)

            # Se registran parámetros y métricas en MLflow
            mlflow.log_params(params)
            mlflow.log_metric("valid_f1", f1)

            # Se registra el modelo en MLflow
            mlflow.sklearn.log_model(model, "model")

            return f1

    # Se optimiza con Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    # Se guardan los mejores resultados y gráficos
    best_trial = study.best_trial
    best_experiment_name = f"XGBoost Experiment: lr_{best_trial.params['learning_rate']:.3f}"
    experiment = mlflow.get_experiment_by_name(best_experiment_name)
    best_model = get_best_model(experiment.experiment_id)
    
    # Se inicia una run en el mejor experimento
    with mlflow.start_run(run_name="plots", experiment_id=experiment.experiment_id):
        # Gráfico de la importancia de las características del mejor modelo
        importance_fig = plot_feature_importance(best_model)
        importance_path = os.path.join("plots", "feature_importance.png")
        importance_fig.savefig(importance_path)
        
        # Se guardan los gráficos de Optuna e importancia de características en la run del mejor modelo
        mlflow.log_artifact(save_optuna_plot(plot_optimization_history(study), "optimization_history.png"))
        mlflow.log_artifact(save_optuna_plot(plot_param_importances(study), "param_importances.png"))
        mlflow.log_artifact(importance_path)
    
    # Se serializa el mejor modelo
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print(f"Mejor F1-Score: {best_trial.value}")
    print(f"Mejores hiperparámetros: {best_trial.params}")


def save_optuna_plot(plot, filename):
    if isinstance(plot, plt.Axes):
        plot = plot.figure
    
    if isinstance(plot, plt.Figure):
        path = os.path.join("plots", filename)
        plot.savefig(path)
        return path
    else:
        raise ValueError("El argumento 'plot' debe ser una figura de Matplotlib.")


def get_best_model(experiment_id):
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    best_model_id = runs.sort_values("metrics.valid_f1", ascending=False)["run_id"].iloc[0]
    best_model = mlflow.sklearn.load_model(f"runs:/{best_model_id}/model")
    return best_model


def plot_feature_importance(model):
    fig, ax = plt.subplots(figsize=(10, 8))
    xgb.plot_importance(model, importance_type="weight", max_num_features=10, ax=ax)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    optimize_model()
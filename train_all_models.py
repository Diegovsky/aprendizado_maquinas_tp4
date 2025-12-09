import os
import time

# Importing refactored modules
from models import (
    gbte,
    knn,
    random_forest,
    regressao_penalizada,
    prediction_rule_ensemble,
    arvore_decisao,
    modelo_gam,
)
from models.shared_utils import save_results_to_file


def run_all_models():
    """Train all models and save results to txt files."""
    # --- CONFIGURATION ---
    # Set the file path used by all scripts
    CAMINHO_ARQUIVO = "dataset_limpo_completo.tsv"

    # Check if file exists (Optional: remove if you want to run synthetic)
    if not os.path.exists(CAMINHO_ARQUIVO):
        print(f"WARNING: File '{CAMINHO_ARQUIVO}' was not found.")
        print("Scripts will generate synthetic data automatically if supported.\n")
        # Uncomment below to force file existence:
        # return

    # List of algorithms to run
    # Structure: (Module, Entry Point, Extra Arguments Dict)
    algorithms = [
        # (knn, "run", {}),
        # (arvore_decisao, "run", {}),
        # (random_forest, "run", {}),
        # (gbte, "run", {}),
        # # RuleFit can run as classifier
        # (prediction_rule_ensemble, "run", {"mode": "classify"}),
        (modelo_gam, "run", {}),
        # Penalized regression requires target_col
        (regressao_penalizada, "run", {"target_col": "target"}),
    ]

    # Mapping of modules to output file names
    model_names = {
        "knn": "knn",
        "arvore_decisao": "decision_tree",
        "random_forest": "random_forest",
        "gbte": "gbte",
        "prediction_rule_ensemble": "rulefit",
        "modelo_gam": "gam",
        "regressao_penalizada": "penalized_regression",
    }

    for modulo, entry_point, kwargs in algorithms:
        module_name = modulo.__name__.split(".")[-1]
        output_name = model_names.get(module_name, module_name)

        try:
            # Get the function from module using reflection
            func = getattr(modulo, entry_point)

            # Call the function passing the file and any extra arguments
            results = func(file_path=CAMINHO_ARQUIVO, **kwargs)

            if results and isinstance(results, dict) and 'error' not in results:
                # Save results to txt file
                output_file = save_results_to_file(results, output_name)
                print(f"[OK] {module_name} → {output_file}")
            elif results and 'error' in results:
                print(f"[ERROR] {module_name}: {results['error']}")
            else:
                print(f"[ERROR] {module_name}: unexpected results format")

        except AttributeError:
            print(f"[ERROR] Function '{entry_point}' does not exist in {module_name}.")
        except Exception as e:
            print(f"[ERROR] {module_name}: {e}")
            continue

        # Small pause to ensure prints don't mix in buffer
        time.sleep(0.5)


if __name__ == "__main__":
    run_all_models()

import os
import time

# Importing refactored modules
import regressao_penalizada
from models import (
    gbte,
    knn,
    random_forest,
    prediction_rule_ensemble,
    arvore_decisao,
    modelo_gam,
)
from models.shared_utils import save_results_to_file


def run_all_models():
    """Train all models and save results to txt files."""
    # --- CONFIGURATION ---
    # Set the file path used by all scripts
    CAMINHO_ARQUIVO = "datasets/meu_dataset_treino.tsv"
    
    # Check if file exists (Optional: remove if you want to run synthetic)
    if not os.path.exists(CAMINHO_ARQUIVO):
        print(f"WARNING: File '{CAMINHO_ARQUIVO}' was not found.")
        print("Scripts will generate synthetic data automatically if supported.\n")
        # Uncomment below to force file existence:
        # return 

    print(f"=== STARTING UNIFIED TRAINING ===")
    print(f"Target file: {CAMINHO_ARQUIVO}\n")

    # List of algorithms to run
    # Structure: (Module, Entry Point, Extra Arguments Dict)
    algorithms = [
        (knn, "run", {}),
        (arvore_decisao, "run", {}),
        (random_forest, "run", {}),
        (gbte, "run", {}),
        (modelo_gam, "run", {}),
        # Penalized regression requires target_col
        (regressao_penalizada, "run", {"target_col": "target"}),
        # RuleFit can run as classifier
        (prediction_rule_ensemble, "run", {"mode": "classify"}),
    ]

    # Mapping of modules to output file names
    model_names = {
        "knn": "knn",
        "arvore_decisao": "decision_tree",
        "random_forest": "random_forest",
        "gbte": "gbte",
        "modelo_gam": "gam",
        "regressao_penalizada": "penalized_regression",
        "prediction_rule_ensemble": "rulefit",
    }

    for modulo, entry_point, kwargs in algorithms:
        module_name = modulo.__name__.split('.')[-1]
        output_name = model_names.get(module_name, module_name)
        
        print("\n" + "#"*80)
        print(f"### RUNNING MODULE: {module_name.upper()} ###")
        print("#"*80 + "\n")
        
        try:
            # Get the function from module using reflection
            func = getattr(modulo, entry_point)
            
            # Call the function passing the file and any extra arguments
            results = func(file_path=CAMINHO_ARQUIVO, **kwargs)
            
            if results and isinstance(results, dict):
                # Save results to txt file
                output_file = save_results_to_file(results, output_name)
                print(f"\n[SUCCESS] Module {module_name} completed.")
                print(f"[SAVED] Results written to {output_file}")
            else:
                print(f"\n[WARNING] Module {module_name} returned unexpected results.")
                
        except AttributeError:
            print(f"\n[CONFIGURATION ERROR] Function '{entry_point}' does not exist in {module_name}.")
        except Exception as e:
            print(f"\n[EXECUTION ERROR] Error running {module_name}: {e}")
            # Continue ensures one error doesn't stop the others
            continue
        
        # Small pause to ensure prints don't mix in buffer
        time.sleep(1) 

    print("\n" + "="*80)
    print("=== ALL TRAININGS HAVE BEEN COMPLETED ===")
    print("="*80)


if __name__ == "__main__":
    run_all_models()

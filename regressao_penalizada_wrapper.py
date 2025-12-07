"""
Wrapper: Regressão Penalizada
Importa do módulo refatorizado em models/
"""
from models.regressao_penalizada import executar_pipeline
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arquivo', type=str, default=None)
    parser.add_argument('--target', type=str, default='target')
    args = parser.parse_args()
    executar_pipeline(caminho_arquivo=args.arquivo, target_col=args.target)

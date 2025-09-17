# main.py
from __future__ import annotations
import argparse
import importlib
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"
APP_DEFAULT = ROOT / "app" / "app.py"   # caminho do seu app Streamlit
ETL_DEFAULT_MODULE = "api"              # usa src/api.py por padrão

# garante que src/ esteja no PYTHONPATH
sys.path.insert(0, str(SRC))

load_dotenv()  # opcional: carrega .env

def run_etl(module_name: str = ETL_DEFAULT_MODULE,
            func_candidates: tuple[str, ...] = ("run_etl", "main", "run")):
    """
    Importa src/<module_name>.py e tenta executar uma das funções:
    run_etl(), main(), run(). Se nenhuma existir, roda `python -m <module_name>`.
    """
    print(f"[runner] Importando módulo: {module_name}")
    mod = importlib.import_module(module_name)

    for fname in func_candidates:
        fn = getattr(mod, fname, None)
        if callable(fn):
            print(f"[runner] Executando {module_name}.{fname}() ...")
            fn()
            print("[runner] ETL finalizado.\n")
            return

    # fallback: roda o módulo como script
    print(f"[runner] Nenhuma {func_candidates} encontrada. Rodando `python -m {module_name}` ...")
    subprocess.run([sys.executable, "-m", module_name], check=True)
    print("[runner] ETL finalizado.\n")

def run_streamlit(app_path: Path, port: int = 8501, host: str = "localhost"):
    """Abre o app Streamlit."""
    if not app_path.exists():
        raise FileNotFoundError(f"App não encontrado: {app_path}")
    print(f"[runner] Abrindo Streamlit em http://{host}:{port} -> {app_path}")
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(port),
        "--server.address", host,
    ]
    subprocess.run(cmd, check=True)

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Runner ETL (src/api.py) + Streamlit (app/app.py)")

    # >>> Opções GLOBAIS (sempre presentes) <<<
    p.add_argument("--etl-module", default=ETL_DEFAULT_MODULE, help="Módulo em src/ (default: api)")
    p.add_argument("--funcs", nargs="*", default=["run_etl","main","run"],
                   help="Funções a tentar no módulo (ordem). Default: run_etl main run")
    p.add_argument("--app",  default=str(APP_DEFAULT), help="Caminho do app Streamlit (default: app/app.py)")
    p.add_argument("--port", type=int, default=8501,   help="Porta do Streamlit (default: 8501)")
    p.add_argument("--host", default="localhost",      help="Host do Streamlit (default: localhost)")

    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("etl", help="Rodar apenas o ETL")
    sub.add_parser("app", help="Rodar apenas o app Streamlit")
    sub.add_parser("all", help="Rodar ETL e depois abrir o app")

    # se nenhum subcomando for passado, assume "all"
    p.set_defaults(cmd="all")
    return p.parse_args(argv)

def main():
    args = parse_args()

    if args.cmd == "etl":
        run_etl(args.etl_module, tuple(args.funcs))
    elif args.cmd == "app":
        run_streamlit(Path(args.app), port=args.port, host=args.host)
    elif args.cmd == "all":
        run_etl(args.etl_module, tuple(args.funcs))
        run_streamlit(Path(args.app), port=args.port, host=args.host)

if __name__ == "__main__":
    main()

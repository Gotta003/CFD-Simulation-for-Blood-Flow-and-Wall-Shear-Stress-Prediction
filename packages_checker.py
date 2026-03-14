import sys

checks={
    "numpy": lambda: __import__("numpy").__version__,
    "scipy": lambda: __import__("scipy").__version__,
    "pandas": lambda: __import__("pandas").__version__,
    "vtk": lambda: __import__("vtk").__version__,
    "pyvista": lambda: __import__("pyvista").__version__,
    "sklearn": lambda: __import__("sklearn").__version__,
    "xgboost": lambda: __import__("xgboost").__version__,
    "lightgbm": lambda: __import__("lightgbm").__version__,
    "torch": lambda: __import__("torch").__version__,
    "torch_geometric": lambda: __import__("torch_geometric").__version__,
    "shap": lambda: __import__("shap").__version__,
    "matplotlib": lambda: __import__("matplotlib").__version__,
    "plotly": lambda: __import__("plotly").__version__,
}

GREEN="\033[0;32m"
RED="\033[0;31m"
RESET="\033[0m"
failed=[]

for pkg, ver in checks.items():
    try:
        print(f"{GREEN}{pkg:<20} {ver}{RESET}")
    except Exception as e:
        print(f"{RED}{pkg:<20} not found{RESET}")
        failed.append(pkg)

if failed:
    print(f"\n{RED}The following packages are missing: {', '.join(failed)}{RESET}")
    sys.exit(1)
else:
    print(f"\n{GREEN}All packages are installed!{RESET}")
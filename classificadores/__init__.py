"""
Descoberta automática dos classificadores.

Qualquer arquivo algoritmo_N.py nesta pasta que exponha NOME (str) e
criar() -> (modelo, grade) é registrado como um algoritmo disponível.
Arquivos ainda vazios (sem NOME/criar) são ignorados silenciosamente, para
que o app funcione mesmo antes de o grupo implementar os 5.
"""

import importlib
import pkgutil
from pathlib import Path

_PASTA = Path(__file__).parent


def descobrir():
    """Retorna lista ordenada de dicts: {chave, nome, modulo}."""
    encontrados = []
    for info in pkgutil.iter_modules([str(_PASTA)]):
        if info.name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"{__name__}.{info.name}")
        except Exception:  # arquivo com import quebrado -> ignora
            continue
        if hasattr(mod, "NOME") and hasattr(mod, "criar") and callable(mod.criar):
            encontrados.append({"chave": info.name, "nome": mod.NOME, "modulo": mod})
    encontrados.sort(key=lambda d: d["chave"])
    return encontrados

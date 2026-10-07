"""Descoberta dos classificadores compatíveis com o fluxo do jogo."""

import importlib
import pkgutil
from pathlib import Path
from types import ModuleType

_PASTA = Path(__file__).parent
FUNCOES_CONTRATO = ("treinar_e_avaliar", "classificar")


def descobrir() -> list[dict[str, str | ModuleType]]:
    """Lista módulos que implementam o contrato usado pela interface."""
    encontrados: list[dict[str, str | ModuleType]] = []
    for info in pkgutil.iter_modules([str(_PASTA)]):
        if info.name.startswith("_"):
            continue
        modulo = importlib.import_module(f"{__name__}.{info.name}")
        nome = getattr(modulo, "NOME", None)
        if isinstance(nome, str) and nome and all(
            callable(getattr(modulo, funcao, None))
            for funcao in FUNCOES_CONTRATO
        ):
            encontrados.append(
                {
                    "chave": info.name,
                    "nome": nome,
                    "modulo": modulo,
                }
            )
    encontrados.sort(key=lambda item: str(item["nome"]).casefold())
    return encontrados

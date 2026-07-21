"""Registro dos agentes disponíveis na linha de comando.

Centraliza a criação por nome para que jogar.py e torneio.py usem a mesma lista.
Agente novo entra aqui uma vez e passa a valer nos dois.
"""

from __future__ import annotations

from .aleatorio import AgenteAleatorio
from .base import Agente
from .genetico import AgenteGenetico
from .heuristico import AgenteHeuristico
from .reforco import AgenteReforco

# Os agentes do estudo, mais o piso de referência.
TIPOS = ("aleatorio", "heuristico", "genetico", "reforco")

BASELINES = ("aleatorio",)

# Ficam fora do torneio padrão. Só entram se pedidas por nome em --tipos.
VARIANTES = ("heuristico_preditivo",)

TODOS = TIPOS + VARIANTES


def criar_agente(tipo: str, lado: str, n_acoes: int, seed: int = 0) -> Agente:
    """Instancia um agente pelo nome. O lado diz qual raquete ele defende."""
    if tipo == "aleatorio":
        return AgenteAleatorio(n_acoes, seed=seed)
    if tipo == "heuristico":
        return AgenteHeuristico(lado=lado)
    if tipo == "heuristico_preditivo":
        return AgenteHeuristico(lado=lado, prever=True)
    if tipo == "genetico":
        return AgenteGenetico(lado=lado)
    if tipo == "reforco":
        return AgenteReforco(lado=lado, n_acoes=n_acoes, seed=seed)
    raise ValueError(f"tipo de agente desconhecido: {tipo}")

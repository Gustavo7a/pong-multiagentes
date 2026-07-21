"""Agente aleatório, o piso de referência do projeto.

Sorteia uma ação qualquer do espaço de ações. Serve como linha de base: quem não
ganhar dele com folga não está jogando de verdade.
"""

from __future__ import annotations

import numpy as np

from .base import Agente


class AgenteAleatorio(Agente):
    nome = "aleatorio"

    def __init__(self, n_acoes: int, seed: int | None = None):
        self.n_acoes = n_acoes
        self._rng = np.random.default_rng(seed)

    def agir(self, observacao: np.ndarray) -> int:
        return int(self._rng.integers(self.n_acoes))

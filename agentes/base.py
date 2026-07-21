"""Interface comum a todos os agentes do projeto."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Agente(ABC):
    """Contrato mínimo de um agente para o ambiente Pong."""

    nome: str = "agente"

    @abstractmethod
    def agir(self, observacao: np.ndarray) -> int:
        """Escolhe uma ação a partir dos 128 bytes da RAM."""
        raise NotImplementedError

    def reiniciar(self) -> None:
        """Chamado no início de cada episódio. Sobrescreva se guardar estado."""

    def observar(self, recompensa: float, terminou: bool) -> None:
        """Retorno do ambiente depois da ação. Serve para quem aprende."""

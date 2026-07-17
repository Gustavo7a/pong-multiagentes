"""Interface comum a todos os agentes do projeto.

Todo agente (aleatório, heurístico, aprendizado por reforço, genético) deve
implementar ``agir`` recebendo a observação (RAM de 128 bytes) e devolvendo uma
ação inteira válida no espaço de ações do ambiente.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Agente(ABC):
    """Contrato mínimo de um agente para o ambiente Pong."""

    nome: str = "agente"

    @abstractmethod
    def agir(self, observacao: np.ndarray) -> int:
        """Escolhe uma ação a partir da observação (vetor de 128 bytes)."""
        raise NotImplementedError

    def reiniciar(self) -> None:
        """Chamado no início de cada episódio. Sobrescreva se guardar estado."""

    def observar(self, recompensa: float, terminou: bool) -> None:
        """Recebe o retorno do ambiente após a ação. Útil para aprendizado."""

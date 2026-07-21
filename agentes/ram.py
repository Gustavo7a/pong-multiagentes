"""Traduz a RAM do Atari para o estado do Pong.

A observação do ambiente são 128 bytes da memória do console. Alguns guardam as
posições da bola e das raquetes; o resto não interessa aqui.

Os endereços e as faixas foram medidos com scripts/inspecionar_ram.py neste
ambiente, não copiados de tabela pronta.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BYTE_BOLA_X = 49
BYTE_BOLA_Y = 54
BYTE_RAQUETE_ESQUERDA = 50  # second_0, coluna x=64
BYTE_RAQUETE_DIREITA = 51  # first_0, coluna x=188
BYTE_RAQUETE_ESQUERDA_X = 45
BYTE_RAQUETE_DIREITA_X = 46
BYTE_PLACAR_ESQUERDA = 13
BYTE_PLACAR_DIREITA = 14

# Limites da raquete, medidos segurando CIMA e BAIXO até travar.
RAQUETE_Y_MINIMO = 38
RAQUETE_Y_MAXIMO = 203

# Paredes onde a bola ricocheteia, na escala vertical dela.
BOLA_Y_MINIMO = 45
BOLA_Y_MAXIMO = 207

# Bola e raquete não usam a mesma origem vertical: nas rebatidas o bola_y fica
# uns 10 pixels acima do byte da raquete.
DESLOCAMENTO_RAQUETE = 10.0


@dataclass(frozen=True)
class EstadoPong:
    """Estado do jogo em coordenadas de tela."""

    bola_x: float
    bola_y: float
    raquete_esquerda_y: float
    raquete_direita_y: float
    raquete_esquerda_x: float
    raquete_direita_x: float
    placar_esquerda: int
    placar_direita: int

    @property
    def bola_em_jogo(self) -> bool:
        """Falso entre um ponto e o saque seguinte, quando a bola some da tela."""
        return self.bola_x > 0 and self.bola_y > 0

    def centro_raquete(self, lado: str) -> float:
        """Centro da raquete já na escala vertical da bola."""
        bruto = (
            self.raquete_esquerda_y if lado == "esquerda" else self.raquete_direita_y
        )
        return bruto + DESLOCAMENTO_RAQUETE

    def x_raquete(self, lado: str) -> float:
        return self.raquete_esquerda_x if lado == "esquerda" else self.raquete_direita_x


def decodificar(observacao: np.ndarray) -> EstadoPong:
    ram = np.asarray(observacao, dtype=np.int64)
    return EstadoPong(
        bola_x=float(ram[BYTE_BOLA_X]),
        bola_y=float(ram[BYTE_BOLA_Y]),
        raquete_esquerda_y=float(ram[BYTE_RAQUETE_ESQUERDA]),
        raquete_direita_y=float(ram[BYTE_RAQUETE_DIREITA]),
        raquete_esquerda_x=float(ram[BYTE_RAQUETE_ESQUERDA_X]),
        raquete_direita_x=float(ram[BYTE_RAQUETE_DIREITA_X]),
        placar_esquerda=int(ram[BYTE_PLACAR_ESQUERDA]),
        placar_direita=int(ram[BYTE_PLACAR_DIREITA]),
    )

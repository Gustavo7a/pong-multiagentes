"""Ambiente Pong multiagente do PettingZoo, com observação pela RAM.

Contexto II do estudo dirigido. A observação é só a RAM do console
(obs_type="ram"), ou seja, 128 bytes com valores de 0 a 255. Os dois agentes
recebem o mesmo vetor e escolhem uma ação discreta. Quem marca ponto ganha +1 e
quem sofre leva -1, então a soma é sempre zero.

Cuidado com os nomes: first_0 é a raquete da direita e second_0 é a da esquerda,
o contrário do que o nome sugere. Verificado com scripts/inspecionar_ram.py,
segurando CIMA só para o first_0 e vendo mexer o byte 51, que é a raquete da
coluna x=188.
"""

from __future__ import annotations

from pettingzoo.atari import pong_v3

AGENTE_ESQUERDA = "second_0"
AGENTE_DIREITA = "first_0"


# Uma partida até 21 pontos leva de 4 mil a 15 mil passos. Só que, se os dois
# lados defendem bem, o peloteio não acaba e por volta do passo 65 mil a ROM
# congela: a bola some, as raquetes param e o ALE começa a descontar ponto de um
# dos lados. Testei as 6 ações dos dois agentes nesse estado e nenhuma destrava.
# O corte abaixo encerra como empate antes disso.
MAX_CICLOS = 20_000


def criar_ambiente(
    render_mode: str | None = None,
    max_cycles: int = MAX_CICLOS,
    full_action_space: bool = False,
):
    """Cria o ambiente no modo parallel.

    Em render_mode, "human" abre a janela e None roda sem desenhar nada, que é
    bem mais rápido. Com full_action_space desligado sobram só as 6 ações que o
    Pong usa de fato.
    """
    return pong_v3.parallel_env(
        obs_type="ram",
        render_mode=render_mode,
        max_cycles=max_cycles,
        full_action_space=full_action_space,
    )


def descrever_ambiente() -> None:
    """Imprime os espaços de observação e ação. Útil para depurar."""
    env = criar_ambiente()
    env.reset(seed=0)
    print("Agentes:", env.agents)
    for agente in env.agents:
        print(f"  {agente}:")
        print(f"    observação: {env.observation_space(agente)}")
        print(f"    ação:       {env.action_space(agente)}")
    env.close()


if __name__ == "__main__":
    descrever_ambiente()

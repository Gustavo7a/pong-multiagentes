"""Roda partidas de Pong entre dois agentes e coleta métricas.

Uso:
    python jogar.py                                  # aleatório x aleatório
    python jogar.py --esquerda heuristico            # heurístico x aleatório
    python jogar.py --episodios 20                   # mais episódios
    python jogar.py --render                         # abre a janela do jogo
    python jogar.py --render --atraso 0.03           # janela em câmera lenta
    python jogar.py --seed 42

É aqui que agente e ambiente se conectam: cada agente recebe os 128 bytes da
RAM, devolve uma ação inteira e leva de volta a recompensa (+1 por ponto feito,
-1 por ponto sofrido) mais o aviso de término.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

from ambiente_pong import (
    AGENTE_DIREITA,
    AGENTE_ESQUERDA,
    MAX_CICLOS,
    criar_ambiente,
)
from agentes import TODOS, criar_agente


def jogar_episodio(
    env, agentes: dict, seed: int | None = None, atraso: float = 0.0
) -> dict:
    """Joga um episódio inteiro e devolve a recompensa acumulada de cada agente."""
    observacoes, _ = env.reset(seed=seed)
    for ag in agentes.values():
        ag.reiniciar()

    total = defaultdict(float)

    while env.agents:
        acoes = {
            nome: agentes[nome].agir(observacoes[nome]) for nome in env.agents
        }
        observacoes, recompensas, terminacoes, truncamentos, _ = env.step(acoes)
        for nome, r in recompensas.items():
            total[nome] += r
            fim = terminacoes.get(nome, False) or truncamentos.get(nome, False)
            agentes[nome].observar(r, fim)
        if atraso:
            time.sleep(atraso)

    return dict(total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pong multiagente (PettingZoo)")
    parser.add_argument("--episodios", type=int, default=5)
    parser.add_argument("--render", action="store_true", help="abre a janela")
    parser.add_argument(
        "--atraso",
        type=float,
        default=0.0,
        help="segundos de pausa entre passos (ex.: 0.03 = ~30fps, 0.1 = bem devagar)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--esquerda", choices=TODOS, default="aleatorio", help="agente do second_0"
    )
    parser.add_argument(
        "--direita", choices=TODOS, default="aleatorio", help="agente do first_0"
    )
    parser.add_argument(
        "--max-ciclos",
        type=int,
        default=MAX_CICLOS,
        help="passos máximos por episódio (empate se estourar)",
    )
    args = parser.parse_args()

    env = criar_ambiente(
        render_mode="human" if args.render else None,
        max_cycles=args.max_ciclos,
    )
    env.reset(seed=args.seed)

    n_acoes = env.action_space(AGENTE_ESQUERDA).n
    agentes = {
        AGENTE_ESQUERDA: criar_agente(
            args.esquerda, "esquerda", n_acoes, args.seed
        ),
        AGENTE_DIREITA: criar_agente(
            args.direita, "direita", n_acoes, args.seed + 1
        ),
    }

    print(f"Ações disponíveis por agente: {n_acoes}")
    print(
        f"{AGENTE_ESQUERDA} = {agentes[AGENTE_ESQUERDA].nome}   "
        f"{AGENTE_DIREITA} = {agentes[AGENTE_DIREITA].nome}"
    )
    print(f"Rodando {args.episodios} episódio(s)...\n")

    vitorias = defaultdict(int)
    empates = 0
    for ep in range(args.episodios):
        total = jogar_episodio(env, agentes, seed=args.seed + ep, atraso=args.atraso)
        esquerda = total.get(AGENTE_ESQUERDA, 0.0)
        direita = total.get(AGENTE_DIREITA, 0.0)

        if esquerda == direita:
            empates += 1
            resultado = "empate (limite de ciclos)"
        else:
            vencedor = AGENTE_ESQUERDA if esquerda > direita else AGENTE_DIREITA
            vitorias[vencedor] += 1
            resultado = f"venceu {agentes[vencedor].nome}"

        print(
            f"Episódio {ep + 1:>2}: "
            f"{AGENTE_ESQUERDA}={esquerda:+.0f}  "
            f"{AGENTE_DIREITA}={direita:+.0f}  -> {resultado}"
        )

    env.close()

    print("\nResumo:")
    for nome in (AGENTE_ESQUERDA, AGENTE_DIREITA):
        print(
            f"  {nome} ({agentes[nome].nome}): "
            f"{vitorias[nome]}/{args.episodios} vitórias"
        )
    if empates:
        print(f"  empates: {empates}/{args.episodios}")


if __name__ == "__main__":
    main()

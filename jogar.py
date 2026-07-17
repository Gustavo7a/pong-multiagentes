"""Roda partidas de Pong entre dois agentes e coleta métricas.

Uso:
    python jogar.py                      # aleatório x aleatório, 5 episódios
    python jogar.py --episodios 20       # mais episódios
    python jogar.py --render             # abre a janela do jogo
    python jogar.py --seed 42

Este é o ponto de conexão agente <-> ambiente (item 3 do vídeo):
    entrada  -> observação de RAM (128 bytes) de cada agente
    saída    -> ação inteira escolhida por cada agente
    feedback -> recompensa (+1 ponto feito, -1 ponto sofrido) e término
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from ambiente_pong import AGENTE_DIREITA, AGENTE_ESQUERDA, criar_ambiente
from agentes import AgenteAleatorio


def jogar_episodio(env, agentes: dict, seed: int | None = None) -> dict:
    """Joga um episódio completo. Retorna a recompensa acumulada por agente."""
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

    return dict(total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pong multiagente (PettingZoo)")
    parser.add_argument("--episodios", type=int, default=5)
    parser.add_argument("--render", action="store_true", help="abre a janela")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = criar_ambiente(render_mode="human" if args.render else None)
    env.reset(seed=args.seed)

    n_acoes = env.action_space(AGENTE_ESQUERDA).n
    agentes = {
        AGENTE_ESQUERDA: AgenteAleatorio(n_acoes, seed=args.seed),
        AGENTE_DIREITA: AgenteAleatorio(n_acoes, seed=args.seed + 1),
    }

    print(f"Ações disponíveis por agente: {n_acoes}")
    print(f"Rodando {args.episodios} episódio(s)...\n")

    vitorias = defaultdict(int)
    for ep in range(args.episodios):
        total = jogar_episodio(env, agentes, seed=args.seed + ep)
        vencedor = max(total, key=total.get)
        vitorias[vencedor] += 1
        print(
            f"Episódio {ep + 1:>2}: "
            f"{AGENTE_ESQUERDA}={total.get(AGENTE_ESQUERDA, 0):+.0f}  "
            f"{AGENTE_DIREITA}={total.get(AGENTE_DIREITA, 0):+.0f}  "
            f"-> venceu {vencedor}"
        )

    env.close()

    print("\nResumo de vitórias:")
    for nome, v in vitorias.items():
        print(f"  {nome}: {v}/{args.episodios}")


if __name__ == "__main__":
    main()

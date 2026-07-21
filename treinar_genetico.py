"""Evolui os pesos do agente genético.

Cada indivíduo é um vetor de pesos que joga algumas partidas e recebe como
aptidão o saldo de pontos. Os melhores se reproduzem, os filhos sofrem mutação e
com o tempo a população chega a pesos que jogam Pong. Sem gradiente nenhum.

Uma geração faz o seguinte: avalia todo mundo em partidas curtas contra o
adversário de treino, passa os melhores intactos por elitismo, escolhe pais por
torneio de 3, cruza com máscara uniforme e aplica ruído gaussiano nos filhos.

Uso:
    python treinar_genetico.py                          # padrões
    python treinar_genetico.py --geracoes 40 --populacao 40
    python treinar_genetico.py --adversario heuristico  # oponente mais duro
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from agentes.fabrica import criar_agente
from agentes.genetico import (
    ARQUIVO_PESOS,
    N_GENES,
    AgenteGenetico,
    carregar_genes,
    genes_aleatorios,
    salvar_genes,
)
from ambiente_pong import AGENTE_DIREITA, AGENTE_ESQUERDA, criar_ambiente

# Partidas curtas de propósito. O que interessa é comparar os indivíduos entre
# si, não jogar até 21 pontos, e assim cada avaliação custa bem menos.
CICLOS_TREINO = 2_500


def avaliar(
    env,
    genes: np.ndarray,
    lados: tuple[str, ...],
    adversario: str,
    n_acoes: int,
    episodios: int,
    seed_base: int,
) -> float:
    """Aptidão do indivíduo, medida pelo saldo médio de pontos por partida.

    Avaliar nos dois lados evita que a evolução se especialize em uma raquete só,
    já que quem saca e quem recebe não jogam a mesma partida.
    """
    saldo = 0.0
    partidas = 0

    for lado in lados:
        nome_proprio = AGENTE_ESQUERDA if lado == "esquerda" else AGENTE_DIREITA
        nome_oponente = AGENTE_DIREITA if lado == "esquerda" else AGENTE_ESQUERDA
        lado_oponente = "direita" if lado == "esquerda" else "esquerda"
        agente = AgenteGenetico(lado=lado, genes=genes)

        for ep in range(episodios):
            semente = seed_base + ep
            oponente = criar_agente(
                adversario, lado_oponente, n_acoes, seed=semente
            )
            agentes = {nome_proprio: agente, nome_oponente: oponente}

            observacoes, _ = env.reset(seed=semente)
            for ag in agentes.values():
                ag.reiniciar()

            while env.agents:
                acoes = {
                    nome: agentes[nome].agir(observacoes[nome])
                    for nome in env.agents
                }
                observacoes, recompensas, _, _, _ = env.step(acoes)
                saldo += recompensas.get(nome_proprio, 0.0)
            partidas += 1

    return saldo / partidas


def selecionar(
    populacao: np.ndarray, aptidoes: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Sorteia três indivíduos e devolve o mais apto."""
    disputantes = rng.integers(len(populacao), size=3)
    vencedor = disputantes[int(np.argmax(aptidoes[disputantes]))]
    return populacao[vencedor]


def cruzar(
    pai: np.ndarray, mae: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Cruzamento uniforme, com cada gene vindo de um dos dois pais."""
    mascara = rng.random(N_GENES) < 0.5
    return np.where(mascara, pai, mae)


def mutar(
    genes: np.ndarray, taxa: float, desvio: float, rng: np.random.Generator
) -> np.ndarray:
    """Ruído gaussiano em uma fração dos genes."""
    alvos = rng.random(N_GENES) < taxa
    ruido = rng.normal(0.0, desvio, size=N_GENES) * alvos
    return genes + ruido


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino do agente genético")
    parser.add_argument("--geracoes", type=int, default=25)
    parser.add_argument("--populacao", type=int, default=30)
    parser.add_argument("--episodios", type=int, default=2, help="partidas por indivíduo")
    parser.add_argument("--elite", type=int, default=3)
    parser.add_argument("--mutacao", type=float, default=0.25, help="desvio do ruído")
    parser.add_argument("--taxa-mutacao", type=float, default=0.3)
    parser.add_argument(
        "--lado", choices=("esquerda", "direita", "ambos"), default="ambos"
    )
    parser.add_argument("--adversario", default="aleatorio")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--saida", default=str(ARQUIVO_PESOS))
    parser.add_argument(
        "--inicial",
        default=None,
        help="continua de pesos já salvos, para treinar primeiro contra o "
        "aleatório e depois contra o heurístico",
    )
    args = parser.parse_args()

    lados = ("esquerda", "direita") if args.lado == "ambos" else (args.lado,)

    rng = np.random.default_rng(args.seed)
    env = criar_ambiente(max_cycles=CICLOS_TREINO)
    env.reset(seed=args.seed)
    n_acoes = env.action_space(AGENTE_ESQUERDA).n

    if args.inicial:
        # Retoma de um campeão anterior, ele mesmo mais cópias mutadas dele.
        campeao = carregar_genes(args.inicial)
        populacao = np.array(
            [campeao]
            + [
                mutar(campeao, args.taxa_mutacao, args.mutacao, rng)
                for _ in range(args.populacao - 1)
            ]
        )
    else:
        populacao = np.array(
            [genes_aleatorios(rng) for _ in range(args.populacao)]
        )

    print(
        f"Genes por indivíduo: {N_GENES}   população: {args.populacao}   "
        f"gerações: {args.geracoes}"
    )
    print(f"Adversário de treino: {args.adversario} (lado {args.lado})")
    if args.inicial:
        print(f"Continuando de: {args.inicial}")
    print()

    melhor_genes = populacao[0].copy()
    melhor_aptidao = -np.inf
    historico = []
    inicio = time.time()

    for geracao in range(args.geracoes):
        # Sementes novas a cada geração, senão o campeão decora uma partida só.
        seed_base = args.seed + 1000 * (geracao + 1)
        aptidoes = np.array(
            [
                avaliar(
                    env,
                    genes,
                    lados,
                    args.adversario,
                    n_acoes,
                    args.episodios,
                    seed_base,
                )
                for genes in populacao
            ]
        )

        ordem = np.argsort(aptidoes)[::-1]
        if aptidoes[ordem[0]] > melhor_aptidao:
            melhor_aptidao = float(aptidoes[ordem[0]])
            melhor_genes = populacao[ordem[0]].copy()

        historico.append(
            (float(aptidoes[ordem[0]]), float(aptidoes.mean()))
        )
        print(
            f"Geração {geracao + 1:>3}/{args.geracoes}: "
            f"melhor {aptidoes[ordem[0]]:+6.2f}   "
            f"média {aptidoes.mean():+6.2f}   "
            f"({time.time() - inicio:5.0f}s)"
        )

        nova = [populacao[i].copy() for i in ordem[: args.elite]]
        while len(nova) < args.populacao:
            filho = cruzar(
                selecionar(populacao, aptidoes, rng),
                selecionar(populacao, aptidoes, rng),
                rng,
            )
            nova.append(mutar(filho, args.taxa_mutacao, args.mutacao, rng))
        populacao = np.array(nova)

    env.close()

    caminho = salvar_genes(
        melhor_genes,
        args.saida,
        aptidao=melhor_aptidao,
        adversario=args.adversario,
        lado=args.lado,
        historico=np.array(historico),
    )
    print(f"\nMelhor aptidão: {melhor_aptidao:+.2f} (saldo médio por partida)")
    print(f"Pesos salvos em {caminho}")


if __name__ == "__main__":
    main()

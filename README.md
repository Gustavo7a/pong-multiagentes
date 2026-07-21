# Pong multiagente: estudo dirigido de IA (2026.1)

Ambiente Pong da biblioteca [PettingZoo](https://pettingzoo.farama.org/), na
categoria Atari, configurado para dar como observação só a RAM do console
(`obs_type="ram"`). A cada passo os dois agentes recebem os mesmos 128 bytes da
memória do Atari, com valores de 0 a 255.

Contexto II do estudo dirigido: dois tipos de agentes competindo no mesmo
ambiente. O repositório traz o ambiente, o baseline aleatório, o loop de partida
e os agentes heurístico e genético.

## Estrutura

```
ambiente_pong.py       # ambiente PettingZoo do Pong com observação de RAM
jogar.py               # roda partidas entre 2 agentes e mostra o placar
torneio.py             # todo-contra-todo com tabela de classificação
treinar_genetico.py    # algoritmo genético que evolui os pesos
agentes/
  base.py              # interface Agente (agir / reiniciar / observar)
  fabrica.py           # registro de agentes usado pela linha de comando
  aleatorio.py         # AgenteAleatorio, o piso de referência
  ram.py               # traduz os 128 bytes em bola, raquetes e placar
  heuristico.py        # AgenteHeuristico, regras sobre o estado da RAM
  genetico.py          # AgenteGenetico, rede evoluída sem regra escrita
  pesos_genetico.npz   # pesos campeões, gerados pelo treino
scripts/
  inspecionar_ram.py   # mede o mapa da RAM na marra
requirements.txt
```

## Ambiente

- Jogo: Pong (`pettingzoo.atari.pong_v3`), dois jogadores competindo.
- Agentes: `first_0` é a raquete da direita e `second_0` é a da esquerda, o
  contrário do que os nomes sugerem. Confirmado com
  `scripts/inspecionar_ram.py`: só o `first_0` mexe o byte 51 da RAM, que é a
  raquete da coluna x=188.
- Observação: vetor de 128 bytes da RAM.
- Ação: 6 opções, sendo `0` parado, `1` saque (FIRE), `2` sobe, `3` desce, e
  `4` e `5` o saque junto com movimento.
- Recompensa: `+1` ao marcar ponto e `-1` ao sofrer, então a soma é zero.
- Término: fim da partida em 21 pontos ou `max_cycles`.

### Limite de ciclos

`max_cycles` vale 20 mil por padrão. Uma partida até 21 pontos leva de 4 mil a
15 mil passos, mas quando os dois lados defendem bem o peloteio não termina e por
volta do passo 65 mil a ROM congela: a bola some (o byte 54 vai a zero), as
raquetes param e o ALE começa a descontar pontos de um dos lados. Testei as 6
ações dos dois agentes nesse estado e nenhuma destrava. O corte em 20 mil encerra
o episódio como empate antes de chegar lá.

## Mapa da RAM

`agentes/ram.py` traduz o vetor de 128 bytes em posições. Os endereços foram
medidos neste ambiente, não copiados de tabela pronta:

| byte | conteúdo | faixa medida |
|------|----------|--------------|
| 49 | x da bola | 52 a 205 |
| 54 | y da bola | 45 a 207, com 0 quando está fora de jogo |
| 50 | y da raquete esquerda (`second_0`) | 38 a 203 |
| 51 | y da raquete direita (`first_0`) | 38 a 203 |
| 45 e 46 | coluna de cada raquete | 64 e 188 |

Bola e raquete usam origens verticais diferentes. Nas rebatidas o `bola_y` fica
uns 10 pixels acima do byte da raquete, e é isso que o `DESLOCAMENTO_RAQUETE`
corrige.

Para refazer as medições:

```bash
python scripts/inspecionar_ram.py --passos 25000
```

O script move cada raquete e vê qual byte responde, mede os extremos do eixo
vertical e usa os instantes de rebatida para achar a coluna de cada raquete e o
deslocamento entre os referenciais.

## Agentes

### aleatorio

Sorteia uma ação qualquer. É a referência a ser batida.

### heuristico

Decodifica a RAM, alinha a raquete com a altura atual da bola e para de corrigir
quando a diferença cabe numa zona morta de 3 pixels.

Dois detalhes que o ambiente impõe:

O saque, primeiro. Entre um ponto e o próximo a bola some e o jogo espera FIRE.
Como o botão é sensível à borda, segurar FIRE não dispara de novo, então o agente
alterna FIRE e parado para simular apertar e soltar.

O sentido do eixo, depois. O agente testa nos primeiros passos se a ação `2`
aumenta ou diminui o byte da sua raquete. Assim o mesmo código funciona nos dois
lados, sem constante chumbada.

### genetico

Aqui ninguém programa a decisão. O agente é uma rede pequena, com 6 entradas, 4
neurônios `tanh` e 3 saídas, num total de 43 pesos, e o algoritmo genético só
olha o placar do episódio. Sem gradiente e sem recompensa passo a passo.

As entradas normalizadas são altura da bola, altura da própria raquete,
distância horizontal até ela, velocidade vertical da bola, velocidade horizontal
já orientada (positiva quando a bola vem) e um viés. As saídas pontuam parado,
sobe e desce, e ganha a maior. O saque com FIRE alternado fica fora do que
evolui, senão nenhum indivíduo conseguiria pontuar.

Em cada geração, todo mundo joga partidas curtas de 2500 ciclos nos dois lados e
recebe o saldo médio como aptidão. Os 3 melhores passam intactos por elitismo, os
pais saem de torneios de 3, o cruzamento usa máscara uniforme e os filhos levam
ruído gaussiano em cerca de 30% dos genes.

```bash
python treinar_genetico.py                              # cerca de 10 min
python treinar_genetico.py --geracoes 40 --populacao 40
python treinar_genetico.py --adversario heuristico --inicial agentes/pesos_genetico.npz
```

As sementes mudam a cada geração, senão o campeão decora uma partida específica.

## Resultados

Comparar cada agente só contra o aleatório não diz nada, porque qualquer coisa
que rastreie a bola ganha de 21x0. Por isso o torneio é todo-contra-todo, com
cada confronto disputado nos dois lados:

```bash
python torneio.py --episodios 2
python torneio.py --sem-baseline          # só os agentes de verdade
```

Classificação com 2 partidas por lado (V=3, E=1):

| agente | P | V | E | D | saldo |
|---|---|---|---|---|---|
| `heuristico` | 24 | 8 | 0 | 0 | +168 |
| `genetico` | 12 | 4 | 0 | 4 | -6 |
| `aleatorio` | 0 | 0 | 0 | 8 | -162 |

Saldo por confronto, linha contra coluna:

| | aleatorio | heuristico | genetico |
|---|---|---|---|
| aleatorio | . | -84 | -78 |
| heuristico | +84 | . | +84 |
| genetico | +78 | -84 | . |

Três leituras disso.

A previsão do impacto atrapalha. Antes de fechar o heurístico testei uma versão
que antecipava onde a bola ia cruzar a linha da raquete, extrapolando a
velocidade e simulando as ricochetes. Ela perdeu por +44 de saldo para a versão
que só persegue a altura atual da bola: extrapolar de dois frames erra, e a
raquete sai cedo para o lugar errado, enquanto a versão simples chega a tempo do
mesmo jeito. Ficou guardada como `heuristico_preditivo`, fora do torneio padrão,
e a comparação se repete com `python torneio.py --tipos heuristico heuristico_preditivo`.

O genético aprende a jogar, mas não a defender. Ganha do aleatório com folga,
+78 de saldo, quase 20 pontos por partida, e ainda assim perde de 21x0 para o
heurístico. Rebater bola rápida exige uma precisão que 43 pesos evoluídos só pelo
placar não alcançaram no tempo de treino usado.

O treino trava contra oponente determinístico. Contra o aleatório a aptidão sobe
até +17,5. Contra o heurístico, ela melhora de -12 para -4 e congela por 15
gerações seguidas no mesmo valor exato. Sem ruído no adversário a paisagem de
aptidão vira degrau, e a mutação gaussiana não acha a próxima subida. Treinar
primeiro contra o aleatório e depois contra o heurístico foi o que destravou;
partir direto para o heurístico não sai do lugar.

## Instalação

Requer Python 3.9 até 3.13.

> No Windows: o Atari do PettingZoo depende do `multi-agent-ale-py`, que não
> publica wheel para Windows, só para Linux e macOS ARM. Use WSL2 com Ubuntu,
> onde a wheel instala pronta e o render sai pelo WSLg. Antes de tudo rode
> `sudo apt install -y python3-venv python3-pip`.

```bash
python3 -m venv ~/pong-venv
source ~/pong-venv/bin/activate

pip install -r requirements.txt
```

Os ambientes Atari precisam das ROMs. O pacote `autorom[accept-rom-license]`
instala elas sozinho, mas se faltar alguma coisa force com:

```bash
AutoROM --accept-license
```

## Execução

Ver os espaços de observação e ação:

```bash
python ambiente_pong.py
```

Rodar partidas:

```bash
python jogar.py --episodios 5                       # aleatório x aleatório
python jogar.py --esquerda heuristico               # heurístico x aleatório
python jogar.py --esquerda genetico --direita heuristico
python jogar.py --direita heuristico --render --atraso 0.03   # câmera lenta
python jogar.py --max-ciclos 40000                  # episódios mais longos
python torneio.py                                   # todos contra todos
```

O `--atraso` é a pausa em segundos entre passos. Com `0.03` dá para acompanhar a
jogada e com `0.1` fica quase quadro a quadro.

## Próximos passos

Adicionar em `agentes/`, todos herdando de `Agente` (`base.py`):

- [x] agente heurístico
- [x] agente genético (rede de 43 pesos evoluída pelo placar)
- [ ] agente de aprendizado por reforço

Cada agente novo só precisa implementar `agir(observacao) -> int` e registrar o
nome em `agentes/fabrica.py`.

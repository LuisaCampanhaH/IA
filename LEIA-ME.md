# Trabalho em Competição: Detecção de Intrusão em Redes

Inteligência Artificial, Profa. Cristiane Neri Nobre

## Como a pasta está organizada

```
IA/
├── BANCO_DE_DADOS/                     dados da professora (treino, teste, modelo de submissão) e os arquivos limpos
├── codigo/                             funções em .py que os notebooks importam
│   ├── limpeza.py                      Etapa 1: duplicatas, pares contraditórios, valores impossíveis, vazamento
│   ├── ausentes_codificacao.py         Etapa 2: imputação e codificação
│   ├── transformacao.py                Etapa 2: log1p e normalização
│   ├── avaliacao.py                    avaliação oficial do grupo (folds 5x2, semente 42)
│   └── experimentos.py                 Etapa 3: roda e registra cada experimento
├── resultados_experimentos/            resultados guardados da Etapa 3 (para o notebook rodar em 1 min)
├── Etapa 1 - Analise exploratoria/     notebook da Etapa 1
├── Etapa 2 - Preparacao dos dados/     notebooks da Etapa 2 (um por pessoa + o integrado)
├── Etapa 3 - Classificadores/          notebook + relatório da Etapa 3
└── Etapa 4 - Entrega final/            notebook final (sem e com score), relatório final e submissões
```

## O que entregar em cada etapa

| Etapa | Data | Entregar |
|---|---|---|
| 1. Análise exploratória | 01/11 | `etapa1_analise_exploratoria.ipynb` |
| 2. Pré-processamento | 07/11 | `etapa2_preparacao_dados.ipynb` |
| 3. Classificadores | 21/11 | `etapa3_classificadores.ipynb` (o `Relatorio_Etapa3.pdf` é um extra: o enunciado pede só o notebook) |
| 4. Competição | 28/11 | um dos notebooks `etapa4_final*.ipynb`, `Relatorio_Final_Etapa4.pdf` (3 a 5 páginas), `submissao_<equipe>.csv` e a apresentação |

## Como rodar um notebook

Abra o notebook **dentro da pasta da etapa** e use *Restart & Run All*. A primeira célula de cada notebook muda para a pasta principal, para achar `BANCO_DE_DADOS/` e `codigo/`.

Na Etapa 4, antes de rodar:
- troque `EQUIPE = "XX"` pelo número da equipe;
- escolha `USAR_SCORE = False` ou `True`.

O arquivo `submissao_<equipe>.csv` é gravado na própria pasta da Etapa 4.

# Annotation Tool

Ferramenta interativa para validar detecoes da classe `car` usando um modelo YOLO treinado (`../best.pt`) e registrar resultados em um dataset COCO localizado em `output_dataset/` (imagens aprovadas + `annotations.coco.json`).

## Requisitos
- Python 3.9+
- Dependencias instaladas no ambiente ativo:
  - `ultralytics`
  - `opencv-python`
  - `numpy`
  - `pillow`
  - `tqdm` (opcional, mas recomendada se usar o pipeline de extracao de frames)

> Observacao: `tkinter` ja acompanha a instalacao oficial do Python em Windows e macOS. No Linux pode ser necessario instalar `python3-tk` via gerenciador de pacotes.

## Estrutura esperada
```
tracker/
├── best.pt
├── videos/
│   └── ... video.mp4
└── testa_tracking/
    ├── main.py
    ├── README.md
    └── output_dataset/
        ├── images/
        │   └── ... frames_validados.jpg
        └── annotations.coco.json
```

## Como usar
1. Ajuste `VIDEO_PATH` em `main.py` para apontar para o video a ser validado (caminho absoluto ou relativo).
2. Certifique-se de que `best.pt` esteja no diretorio pai (`../best.pt` a partir de `testa_tracking`).
3. Ative o ambiente virtual e execute:
   ```bash
   python main.py
   ```
4. Para cada frame processado com deteccoes (confidencia >= 90% para `car`), uma janela mostrara as caixas e a confianca. Os frames validados sao gravados em `output_dataset/images/`.

## Controles
- `Validar (Enter)`: mantem as deteccoes exibidas, grava o frame em `output_dataset/images/` e atualiza `annotations.coco.json`.
- `Rejeitar (Espaco)`: ignora o frame atual e segue para o proximo.
- `Sair (Esc)`: encerra o processo; se houver anotacoes em memoria, sao persistidas antes de fechar.
- Botao `Modo anotacao (K)` ou tecla `k`: alterna o modo de anotacao manual (inicia ativado). Quando ativo, clique e arraste dentro da janela para criar novas bounding boxes; elas sao adicionadas ao JSON ao validar o frame atual.
- Botao `Remover anotacao`: ativa o modo de exclusao. Clique sobre uma bounding box (manual ou da rede) para remove-la do frame atual; ela deixara de aparecer e nao sera salva no JSON.

## Saida COCO
O arquivo `annotations.coco.json` (salvo em `output_dataset/`) e atualizado continuamente com a estrutura:
- `categories`: lista contendo a classe `car` (id 1).
- `images`: um item por frame validado (`file_name`, `width`, `height`, `id`).
- `annotations`: cada deteccao aprovada com `bbox` no formato `[x, y, largura, altura]`, `score`, `image_id` e `category_id`.

Os campos `image_id` e `annotation_id` sao gerados sequencialmente e começam em 1. Cada frame aprovado gera um arquivo `VIDEO_FRAME_xxxxx.jpg` em `output_dataset/images/`, alinhado com o registro correspondente no JSON.

## Dicas
- Se quiser trabalhar com outro limiar de confianca ou classe, ajuste `CONF_THRESHOLD` e `TARGET_CLASS` no topo do `main.py`.
- Caso o video seja longo, considere interromper com `Esc`; o progresso ate o momento sera mantido no JSON.
- Para refinar anotacoes, utilize o modo manual (`k`) para complementar deteccoes que a rede nao encontrou.
- Se precisar descartar uma caixa incorreta antes de validar, ative o modo `Remover anotacao` e clique diretamente sobre ela.

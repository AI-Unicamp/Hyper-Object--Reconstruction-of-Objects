# A Clear Starting Point: A Reproducible Experimental Framework for the ICASSP 2026 Hyper-Object Challenge

We present a reproducible experimental framework for the ICASSP 2026 Hyperobject Challenge (Track 2), focusing on baseline engineering rather than architectural changes. Using the official organizer-provided baseline as a fixed reference, we standardize data handling, training and evaluation control, and experiment configuration to enable reliable comparison and rapid ablation. Our framework includes reproducible workflows, category-aware sampling, and practical I/O optimizations. Without modifying the baseline architecture, we achieve competitive performance (0.57146/SSC on the test-private split).

> Hyperobject Challenge official repository: https://github.com/hyper-object/2026-ICASSP-SPGC

This repository contains:
- Complete reproducible tutorial (README)
- Config .yaml files
- CLI flags implemented
- (conferir) Conversion script .h5 -> .zarr
- (conferir) Code for SMOTE generation

## The Challenge - Track 2 summary (from the official challenge README)

This repository provides a **reproducible training/evaluation framework** for the **ICASSP 2026 Hyper-Object Challenge — Track 2 (SPGC GC7)**.

- **Input:** low-resolution RGB image captured with a commodity camera  
- **Output:** high-resolution hyperspectral cube with **C = 61** spectral bands, with **spatial upscaling**  
- **Ranking metric:** **Spectral–Spatial–Color (SSC)** score in **[0, 1]**
  - **Spectral:** SAM, SID, ERGAS
  - **Spatial:** PSNR, SSIM computed on a **standardized sRGB render** (D65 illuminant, CIE 1931 2°)
  - **Color:** ΔE00 computed on the same standardized sRGB render

Official page: https://hyper-object.github.io/

---

### 1) Setup

ICASSP 2026 Hyperobject Challenge (Track 2)
SPGC GC7 Track 2
Track 2 — Joint Spatial & Spectral Super-Resolution
  Input: low-resolution RGB image captured with a commodity camera.
  Output: high-resolution hyperspectral cube with C = 61 and spatial upscaling.

Submissions are ranked by Spectral-Spatial-Color (SSC) score, range in [0,1].
  Spectral: SAM, SID, ERGAS
  Spatial: PSNR, SSIM on a standardized sRGB render (D65, CIE 1931 2°)
  Color: ΔE00 on the same sRGB render


## README DO REPOSITÓRIO ORIGINAL:

## Para rodar o baseline do challenge:
### 1. Crie um virtual environment. Obs.: confira as regras do seu lab (no MIC Lab, por ex., deve-se criar um conda environment, e não um python environment).

#### 1.1. Criando um virtual environment (python)
- Entre na pasta de interesse
- Crie um venv (obs.: criar um venv para cada modelo)
- Ative o venv
- Install requirements

```
cd 2026-ICASSP-SPGC
mkdir venvs
cd venvs
python3 -m venv venv-baseline
source venv-baseline/bin/activate
pip install -r requirements.txt
```

#### 1.2. Criando um conda environment
```
# caso tenha instalado conda e apareca command not found: source miniconda3/bin/activate
conda create -n baseline-venv python=3.11 -y
conda activate baseline-venv
# verifique com:  conda info --envs
```

### 2. Faça upload dos dados
- Faca upload do dataset dentro da pasta `data/` 
- Coloque os dados da track 1/2 dentro de pastas dentro dessa, chamadas `track1` e `track2`
- Mude o nome da pasta publica de teste do track 2 de `public-test` para `test-public`

Portanto, a estrutura do diretório será (representadas somente as pastas acima):

```
2026-ICASSP-SPGC
    |-- data
        |-- track1/
            |-- train/
                |-- mosaic/
                |-- hsi_61/
            |-- test-public/
                |-- mosaic/
                |-- hsi_61/
            |-- test-private/
                |-- mosaic/
        |-- track2/
            |-- train/
                |-- rgb_2/
                |-- hsi_61/
            |-- test-public/
                |-- rgb_2/
                |-- hsi_61/
            |-- test-private/
                |-- rgb_2/
    |-- venvs   # somente se criado o venv do python
        |-- venv-baseline
```

**OBS:** Pode utilizar uma pasta diferente de `data` para armazenar os dados, basta passar `-d DATA_DIR` nos scripts,
onde `DATA_DIR` é a pasta onde estão os dados. A pasta ainda deve seguir a estrutura acima, contudo.

#### Dados extras
Gerei algums "sub-datasets" a mais que podem ser necessários:

* `hsi_61_zarr`: São os dados das imagens hiperspectrais comprimidos em um formato que acelera a leitura em HDs.
O `train` está disponível [aqui](https://drive.google.com/file/d/1VUhh06-X8rkoGVT9kgp1SkCEzK5VDNue/view?usp=sharing)
e o `test-public` está disponível [aqui](https://drive.google.com/file/d/1mYDPvYhqCs1fbDjPfItGDOuyOWf1_nvF/view?usp=sharing).
* `rgb_full`: Renderizações em RGB das imagens a partir das imagens de 61 bandas. São usadas no pré-treino do TRevSCI. O `train`
está disponível [aqui](https://drive.google.com/file/d/15G-PIgDrjGClxsio5a-r68x0KlFBDsoL/view?usp=sharing) e o `test-public`
está disponível [aqui](https://drive.google.com/file/d/1A2xJ9eBADNx5hFru3gjNkqdws34Cp-Bp/view?usp=sharing).

Os "sub-datasets" são os mesmos para ambas as tracks. Se estiver usando Linux/Mac, não precisa baixar duas vezes, basta fazer symlinks.
Por exemplo, se você estiver todos os datasets no track2, basta navegar para a pasta `data/track1/train/` e rodar:
```
ln -s ../../track2/train/hsi_61 .
ln -s ../../track2/train/hsi_61_zarr .
ln -s ../../track2/train/rgb_full .
```
Igualmente, para o `test-public`, navegue para `data/track1/test-public` e rode:
```
ln -s ../../track2/test-public/hsi_61 .
ln -s ../../track2/test-public/hsi_61_zarr .
ln -s ../../track2/test-public/rgb_full .
```
Assim, a estrutura para cada track será:
```
|-- track{1 ou 2}/
    |-- train/
        |-- {mosaic ou rgb_2}/
        |-- rgb_full/
        |-- hsi_61/
        |-- hsi_61_zarr/
    |-- test-public/
        |-- {mosaic ou rgb_2}/
        |-- hsi_61/
        |-- rgb_full/
        |-- hsi_61/
        |-- hsi_61_zarr/
    |-- test-private/
        |-- {mosaic ou rgb_2}/
```

### 3. Instalando requirements e rodando
#### Instalando requirements
Entre na pasta onde está o arquivo requirements.txt e execute:
```
pip install requirements -r requirements.txt
```

### 4. Treinando os modelos
```
python train.py --track TRACK --config PATH_TO_CONFIG
```
Onde `TRACK` deve ser substituído pela track a utilizar (1 ou 2) e `PATH_TO_CONFIG` é um arquivo de configuração.
Caso a configuração não seja especificada, o script usará o baseline. Ou seja, para treinar o baseline do track 2, basta:
```
python train.py --track 2
```

- Case dê OOM, pode ser que seja melhor usar a variável de ambiente: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

#### Treino acelerado
O script `train_fast.py` possibilita um treino mais rápido ao carregar as entradas (leves) e saídas (pesadas) separadamente,
além de possibilitar outros esquemas de treino (por ex. treino da TRevSCI requer esse script). Rodando
```
python train_fast.py --track TRACK --config PATH_TO_CONFIG
```
ele irá automaticamente usar a entrada `mosaic` ou `rgb_2` para a track 1 ou 2, respectivamente, além de usar `hsi_61_zarr`
para a saída. Se você não tiver esse subset instalado, deve usar a flag `-o hsi_61`. Pode-se mudar o dataset de entrada usando
a flag `-i`. Para treinar o MST++ no `rgb_full` para tentar gerar o `hsi_61`, por exemplo, pode-se fazer:
```
python train_fast.py --config PATH_TO_CONFIG -i rgb_full -o hsi_61
```
A flag `-s SEED` também pode ser utilizada para gerar treinos replicáveis (i.e. shuffles e transforms pré-determinados).

#### Indexação alternativa para dados de teste
Com o script `train_fast.py`, pode-se indicar um arquivo com a flag `--index` listando os IDs das samples a serem usadas
como dataset de test. `--index config/indexing/alt.txt` aponta para uma índice de 12 imagens, 4 de cada categoria.

### 6. Avaliando o modelo em test-public
```python evaluate.py --track TRACK --config PATH_TO_CONFIG --model path/to/model.tar```
Para validar um modelo treinado, basta passar a configuração utilizada assim como o checkpoint salvo.

#### Avaliação acelerada
Igualmente ao `train_fast.py`, agora há o `evaluate_fast.py`, que possibilita validação de treinos alternativos como o da TRevSCI.
```python evaluate_fast.py --track TRACK --config PATH_TO_CONFIG --model path/to/model.tar```
Modelos treinados com o `train_fast.py` também salvam a sua configuração com o modelo, então `--config` pode ser omitido. Por ex:
```
python evaluate_fast.py --track TRACK --model runs/track1/blahblah/
```

### 7. Gerando predições para test-private
```
python submission.py --track TRACK --config PATH_TO_CONFIG --model path/to/model.tar
```
Pode-se alterar a pasta de destino da submissão com a flag `-o OUT_DIR`. Por padrão ela vai na pasta `submission_files`.

**OBS:** Modelos treinados com os scripts antigos não serão compatíveis com os scripts acima.
Se quiser rodar um modelo treinado com o script antigo, precisa fazer a seguinte alteração no script de submissão:
```diff
-  model.load_state_dict(checkpoint["model"])
+  model.load_state_dict(checkpoint)
```

#### Predição acelerada
Para usar modelos com a pipeline TRevSCI->MST++, é preciso usar o script `submission_fast.py`. A lógica é a mesma do
`evaluate_fast.py`.

### 8. Adicionar um modelo
Para adicionar um novo modelo:
1. Implemente o modelo na pasta `models/`, como em `example.py`.
2. Adicione uma configuração para o modelo em `configs/`, a partir do arquivo `CONFIG_TEMPLATE.yaml`.
3. Em `models/__init__.py`, altere a função `setup_model` para incluir o setup do modelo, a partir do exemplo no arquivo.

### 9. wandb
Para salvar os dados de treino para o wandb, primeiro é necessário fazer o login com:
```
wandb login
```
e usar a sua chave de API. Em seguida, adicione a flag `--use_wandb` quando for treinar, por exemplo:
```
python train.py --track TRACK --config PATH_TO_CONFIG --use_wandb
```

### Extras
#### A. Para testar se a GPU está sendo corretamente acessada:
    - digite python no terminal
    - import torch
    - veja por algum desses comandos:
        - torch.cuda.is_available(): Returns True if a CUDA-enabled GPU is available and PyTorch can utilize it, otherwise False.
        - torch.cuda.device_count(): Returns the number of available GPUs.
        - torch.cuda.current_device(): Returns the index of the currently selected GPU.
        - torch.cuda.get_device_name(device_id): Returns the name of the GPU specified by device_id.
        - torch.cuda.get_device_properties(device_id): Returns properties of the specified GPU, such as memory and compute capability.
    - dê exit() para voltar ao shell.
#### B. Lembre-se de atualizar o arquivo requirements.txt caso você tenha que instalar mais alguma coisa (via pip)
#### C. Configurando git
        git config --global user.name "Your Name"
        git config --global user.email "your.email@example.com"

### Referência (grosseira) para as métricas:
- SSC_arith (0–1, ↑): ≥ 0.80 bom, ≥ 0.90 ótimo
- SSC_geom (0–1, ↑): ≥ 0.78 bom, ≥ 0.88 ótimo
- SAM_deg (°, ↓): ângulo espectral (<5° bom, <2° ótimo)
- SID (≥0, ↓): divergência espectral (<0.01 bom)
- ERGAS (↓): erro relativo global (<4 bom, <2 ótimo)
- PSNR_dB (↑): qualidade de imagem (>35 dB bom, >40 ótimo)
- SSIM (0–1, ↑): estrutura da imagem (>0.9 bom, >0.95 ótimo)
- DeltaE00 (↓): diferença de cor perceptual (<3 bom, <1 ótimo)


ADICIONAR COLOCAÇÃO (RANKING) - IMAGEM

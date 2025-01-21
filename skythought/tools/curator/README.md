# Using [Bespoke Curator](https://github.com/bespokelabsai/curator) to generate datasets for the Stratos-R1 model.

## Setup

Assuming we are in the top-level directory, install the dependencies:

```bash
pip install -r skythought/tools/requirements.txt
```

## Generate reasoning traces from DeepSeek-R1 using Curator

Our final dataset contains the following subsets:

* Numina: 10.5k samples from the `math`, `olympiads`, and `amc_aime` subset of the [difficulty-labeled Numina dataset](https://huggingface.co/datasets/NovaSky-AI/labeled_numina_difficulty_162K).
* APPS: ~2.5k samples from the [APPS dataset](https://huggingface.co/datasets/codeparrot/apps).
* TACO: ~3k samples from the [TACO dataset](https://huggingface.co/datasets/BAAI/TACO).
* STILL-2: ~1k samples from the [STILL-2 dataset](https://huggingface.co/datasets/RUC-AIBOX/long_form_thought_data_5k).

Run the following scripts to generate reasoning traces and filter out incorrect reasoning traces:

### Numina

```bash
python skythought/tools/curator/generate_numina_data.py
```

### APPS

```bash
python skythought/tools/curator/generate_apps_data.py
```

### TACO

```bash
python skythought/tools/curator/generate_taco_data.py
```

### Combine the datasets

```bash
python skythought/tools/curator/combine_data.py
```

# Action Imitation in Common Action Space for
Customized Action Image Synthesis




![main_1_fix](./assets/intro_v1.png)

## Dependencies and Installation

```
conda create -n clif python=3.9
pip install diffusers==0.23.1
conda activate clif
```

## Training

### Step 1:

We first perform text inversion on customized action token embedding  with action space to encode  the action features into token embeddings

```
bash run_train_ti.sh
```

### Step 2:

Then we train lora and token embeddings together

```
bash run_train_lora.sh
```

## Evaluation

The evaluation of our method are based on two metrics: *S_{action}* and *S_{actor}*.

The prompts used in our quantitative evaluations can be found in dataset.

## Acknowledgements

This code is builds on the code from the [diffusers](https://github.com/huggingface/diffusers) library

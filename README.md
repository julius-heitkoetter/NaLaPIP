# NaLaPIP

![NaLaPIP Framework Image](https://julius-heitkoetter.github.io/NaLaPIP/paper/images/966_diagram1.pdf)

Natural Language Probing with Intuitive Physics models and framework: a class project for MIT's Computation Congitive Science class taught by Josh Tennebaum. 

Read the paper [here](https://julius-heitkoetter.github.io/NaLaPIP/paper/output/Probing-Intuitive-Physics-Understanding.png)

## Setup

### Create python environment

Create a conda python environemtn with the correct packages, as specified in `environment.yml`. This will ensure you have all the requirements needed for both python and WebPPL, a javascript based probalistic programming language. 

To create the conda enviornment, use
```
conda env create -f environment.yml
```

Once you have installed the conda environment (you will only need to do this once), you must activate it (do this every time you open a new script):
```
conda activate NaLaPIP
```

### Upload correct API keys

This repo require API keys: [OpenAI](https://platform.openai.com/docs/quickstart?context=python). Make sure you have them.

One you have it, save the openAI key in your home directory under `~/.openai/api.key`. It is also recommended that if you are using a shared computing system, you restrict the permissions on the folders using commands like `chmod 700`.

### Run the install script

In your python environment and with your API key in the correct location, run the install script:

```
source install.sh
```

this will then create a setup script. You will have to rerun this setup script everytime you start in a new environment using

```
source setup.sh
```

## Runing Experiments

Below is a brief overview of the different modules needed to run this experiment. 

### NaLaPIP

The NaLaPIP model is composed of 2 main modules: `codex` and `webppl`. 

The code modules is run using a command such as
```
python codex_prompting.py input_data.csv run-2023-12-05-1823
```
where `input_data.csv` can be replaced by any file in the `data/` directory and `run-2023-12-05-1823` can be replaced by any experiment name.

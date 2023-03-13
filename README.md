# ExeDS
Welcome! This repo contains the data and source code for paper **Execution-based Evaluation for Data Science Code Generation Models**. 

Automatically generating code is beneficial to the productivity of data science practitioners. Future progress towards this goal requires systems to generate executable code and measure execution correctness. In this repo we introduce ExeDS, a data science code generation dataset for execution evaluation, which contains 534 problems with execution outputs from Jupyter Notebooks, together with 123K examples for training and validation. ExeDS leverages three novel execution metrics to evaluate the executability and output correctness of code, including no error rate, execution exact match, and execution F1 score. We hope our dataset and execution metrics could draw more attention to the execution correctness of code and result in significant advances in code generation! 

## 1. ExeDS Data and Evaluation Metrics

Here, we describe how to use our benchmark and script to evaluate the execution of generated code.

#### 1.0 Evaluation Environment

Before running scripts to evaluate, first set up the environment with the following commands:

```
cd ./evaluation
conda create -n EvalExeDS python==3.7
conda activate EvalExeDS
pip install -r requirements_execution.txt
pip install tree_sitter==0.19.0
pip install rouge
```

#### 1.1 Download ExeDS data 

First, you can download the whole dataset, including ExeDS testset and training/validation set from this [link](https://drive.google.com/drive/folders/1gX07Uxawj5ch7mvlFKxlDTzAsopZPIxn?usp=sharing). We do not directly put them in this repo due to the file limit of GitHub. Download all the files and move them to `./dataset`. 

The raw notebooks should be found in `./dataset/ExeDS_notebooks/`, each with its data dependencies used when executing. 

The csv file `answers.csv` contains the answers for ExeDS testset. Each row contains information of notebook index, row index, ground truth execution output, ground truth code snippet.

You don't need to download other raw notebooks if you just want to use ExeDS. All the 13525 notebooks we use in our work are from the JuiCe dataset ([paper link](https://arxiv.org/pdf/1910.02216.pdf)). You can find the original notebooks from [here](https://github.com/rajasagashe/JuICe).

#### 1.2 Generate code snippet

In this step, you can generate the code snippets for each example of ExeDS with your model. 

The generations for next step evaluation should be written to a `json` file with a list of dictionary, where each should contain at least two keys `target` and `generation`. The template format of the generations should be:

```
[
  {"target": "ground truth code snippet 1",
   "generation": "generated code snippet 1",
  },
  
  {"target": "ground truth code snippet 2",
   "generation": "generated code snippet 2",
  },
  ...
]
```

You can also use our scripts in Section 2, 3, 4 to generate code with the baseline models.

#### 1.3 Evaluate

The evaluation process contains three steps: 

(1) Create environment for testing, controlled by  `--do_create_notebook`.

(2) Rerun the notebooks to obtain execution output, controlled by  `--do_run`.

(3) Evaluate the execution output, controlled by  `--do_evaluate`.

You can also separately run the three steps. Here we give the whole scripts for evaluation as follows. It approximately takes 4 hours to run.

```
export SAVE_RESULT="dir-to-save-generation-file"
python evaluate_execution.py \
  --do_create_notebook \
  --do_run \
  --do_evaluate \
  --split test \
  --path_generation ${SAVE_RESULT}/split_generation_results.json \
  --path_dataset ../dataset/exeds_test.json \
  --data_dir ../dataset/ExeDS_notebooks \
  --path_save_notebooks ${SAVE_RESULT}/testbed_notebooks \
  2>&1 |tee ../logs/evaluate_execution.log
```

## 2. Rerun baseline mothods

#### 1. Data Preparation

Download the data from xxx as described in Section xxx

#### 2. Preprocessing

Use the following scripts to preprocess data for training/validation/test sets.

```
cd ./preprocess
python preprocess.py \
            --split train \
            --file_name exeds_train.json \
            --do_fairseq_tokenization \
            --do_gptneo \
            --token_type token \
            --context_range 3 \
            --max_code_cell_tokens 200 \
            --max_md_cell_tokens 200 \
            --max_ctx_cell_tokens 900
python preprocess.py \
            --split dev \
            --file_name exeds_dev.json \
            --do_fairseq_tokenization \
            --do_gptneo \
            --token_type token \
            --context_range 3 \
            --max_code_cell_tokens 200 \
            --max_md_cell_tokens 200 \
            --max_ctx_cell_tokens 900
python preprocess.py \
            --split test \
            --file_name exeds_test.json \
            --do_fairseq_tokenization \
            --do_gptneo \
            --token_type token \
            --context_range 3 \
            --max_code_cell_tokens 200 \
            --max_md_cell_tokens 200 \
            --max_ctx_cell_tokens 900
```

The parameter `--do_fairseq_tokenization`  controls whether to prepare data for PyMT5 and JuPyT5 

The parameter `--do_gptneo` controls whether to prepare data for GPT-neo series.

#### 3. Training and Evaluation 

Please refer to each Section 3 and 4  for details.

## 3. Baseline CodeGPT and CodeGPT-adapted

#### 1 Environment:

```
pip install torch==1.6
pip install tensorboard
pip install attrs==19.1.0
pip install transformers==3.3
pip install tree_sitter==0.19.0
pip install tokenizers
pip install sentencepiece
pip install scikit-learn
pip install altair
pip install tqdm
pip install rouge
pip install fuzzywuzzy
```

#### 2 Preprocessing for CodeGPT series

Following the Section 2 .2 for preprocessing details.

#### 3 Training

Use the following command:

```
bash traineval_gpt.sh ../ microsoft/CodeGPT-small-py prepro_addTab-df_madeup_token_range3_lineLen1-25_c200m200a900 30 16
```

The parameters in the above command denote:

```
$1: path to the root dir
$2: checkpoint used to intialize the weights, you can use "microsoft/CodeGPT-small-py-adaptedGPT2" or "microsoft/CodeGPT-small-py"
$3: path to the preprocessed data
$4: epochs (default 30)
$5: Number of GPUs (8 or 16)
```

#### 4 Test execution

```
cd codegpt/
cp ../evaluation/* ./ -r
bash evaluate_execution.sh ../ microsoft/CodeGPT-small-py prepro_addTab-df_madeup_token_range3_lineLen1-25_c200m200a900 30 16
```

The parameters are the same with the training step

## 4. Baseline GPT-neo

#### 4.1 Environment:

Pull docker from Docker Hub: `ranpox/pytorch:1.10.0-cuda10.2-apex`. Then

```
cd gptneo/ 
pip install -r requirements.txt
pip install tree_sitter==0.19.0
pip install rouge
```

#### 4.2 Preprocessing for GPT-neo series

Following the Section 2.2 for preprocessing details.

#### 4.3 Training

Use the following command:

```
bash traineval_neo.sh ../ EleutherAI/gpt-neo-125M prepro_addTab-df_madeup_token_range3_lineLen1-25_c200m200a900 10 16
```

The parameters in the above command denote:

```
$1: path to the root dir
$2: checkpoint used to intialize the weights, you can use "EleutherAI/gpt-neo-125M",  "EleutherAI/gpt-neo-1.3B", or "EleutherAI/gpt-neo-2.7B"
$3: path to the preprocessed data
$4: epochs (default 10)
$5: Number of GPUs (8 or 16)
```

#### 4.4 Generate code for testset

```
bash only_predict_neo.sh ../ EleutherAI/gpt-neo-125M prepro_addTab-df_madeup_token_range3_lineLen1-25_c200m200a900 10 16
```

The parameters are the same with the training step

#### 4.5 Test execution

```
cd gpt_neo/
cp ../evaluation/* ./ -r
bash evaluate_execution.sh ../ EleutherAI/gpt-neo-125M prepro_addTab-df_madeup_token_range3_lineLen1-25_c200m200a900 10 16
```

The parameters are the same with the training step

# Cite 

If you find this repo and paper helpful for you, please cite: 

```
@article{huang2022execution,
  title={Execution-based Evaluation for Data Science Code Generation Models},
  author={Huang, Junjie and Wang, Chenglong and Zhang, Jipeng and Yan, Cong and Cui, Haotian and Inala, Jeevana Priya and Clement, Colin and Duan, Nan and Gao, Jianfeng},
  journal={arXiv preprint arXiv:2211.09374},
  year={2022}
}
```

If you use our dataset, please also consider to cite the original JuiCe dataset since our dataset is built upon JuiCe:

```
@article{Agashe2019JuICe,
  title={JuICe: A Large Scale Distantly Supervised Dataset for Open Domain Context-based Code Generation},
  author={Rajas Agashe and Srini Iyer and Luke Zettlemoyer},
  journal={EMNLP-IJCNLP},
  year={2019},
  pages = {5436--5446},
  url = "https://aclanthology.org/D19-1546",
}
```

# Contact 

Feel free to contact Junjie Huang ([JunjayHuang@outlook.com](mailto:guody5@mail2.sysu.edu.cn)), Chenglong Wang ([chengwang@microsoft.com](mailto:chenwang@microsoft.com)), and Nan Duan ([nanduan@microsoft.com](mailto:nanduan@microsoft.com)) if you have any further questions.


# AIxcellent Vibes at GermEval 2025 Shared Task on Candy Speech Detection

## Results
| Subtask | Submission | Model              | (strict) F1 Score | |
|---------|------------|--------------------|------------------:|-|
|       1 |          1 | Qwen3-Embedding-8B |             0.875 | [Notebook](https://github.com/dslaborg/germeval2025/blob/main/subtask_1/submission_subtask1.ipynb) |
|       1 |          2 | XLM-RoBERTa-Large  |             0.891 | [Notebook](https://github.com/dslaborg/germeval2025/blob/main/subtask_1/submission_subtask1-2.ipynb) |
|       2 |          1 | GBERT-Large        |             0.623 | [Notebook](https://github.com/dslaborg/germeval2025/blob/main/subtask_2/submission_subtask2.ipynb) |
|       2 |          2 | XLM-RoBERTa-Large  |             0.631 | [Notebook](https://github.com/dslaborg/germeval2025/blob/main/subtask_2/submission_subtask2-2.ipynb) |


## Setup 

```bash
python_version="$(cat .python-version)"

# install the interpreter if it’s missing
pyenv install -s "${python_version}"

# select python version for current shell
pyenv shell "${python_version}"

# create venv if missing
if [[ ! -d venv ]]; then
  python -m venv venv
fi

# activate venv & install packages
source venv/bin/activate

pip install -U pip setuptools wheel
pip install -r requirements.txt
``` 



Diese Repository enthält den Code, mit dem die Untersuchungen der Bachelorarbeit **Flauschdetektion (GermEval 2025)** 
im Studiengang Angewandte Mathematik und Informatik (dual) B. Sc. an der Fachhochschule Aachen durchgeführt wurden. 


---


**Studiengang**

Angewandte Mathematik und Informatik B.Sc. ([AMI](https://www.fh-aachen.de/studium/angewandte-mathematik-und-informatik-bsc)) an der [FH Aachen](https://www.fh-aachen.de/), University of Applied Sciences.

**Ausbildung mit IHK Abschluss**

Mathematisch technische/-r Softwareentwickler/-in ([MaTSE](https://www.matse-ausbildung.de/startseite.html)) am Lehr- und Forschungsgebiet Igenieurhydrologie ([LFI](https://lfi.rwth-aachen.de/)) der [RWTH Aachen](https://www.rwth-aachen.de/) University.

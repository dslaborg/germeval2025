# AIxcellent Vibes at GermEval 2025 Shared Task on Candy Speech Detection 🍭

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



---


# :trophy: Model

Model on [Huggingface](https://huggingface.co/cortex359/germeval2025)

## Model Details

- **Model Type:** Transformer-based encoder (XLM-RoBERTa-Large)
- **Developed by:** Christian Rene Thelen, Patrick Gustav Blaneck, Tobias Bornheim, Niklas Grieger, Stephan Bialonski (FH Aachen, RWTH Aachen, ORDIX AG, Utrecht University)
- **Paper:** [AIxcellent Vibes at GermEval 2025 Shared Task on Candy Speech Detection: Improving Model Performance by Span-Level Training](https://arxiv.org/abs/2509.07459v2)
- **Base Model:** [XLM-RoBERTa-Large](https://huggingface.co/FacebookAI/xlm-roberta-large) (Conneau et al., 2020)
- **Fine-tuning Objective:** Detection of *candy speech* (positive/supportive language) in German YouTube comments.

## Model Description

This model is a fine-tuned **XLM-RoBERTa-Large** adapted for the **GermEval 2025 Shared Task on Candy Speech Detection**.
It was trained to identify *candy speech* at both:

- **Binary level:** Classify whether a comment contains candy speech.
- **Span level:** Detect the exact spans and categories of candy speech within comments, using a BIO tagging scheme across **10 categories** (positive feedback, compliment, affection declaration, encouragement, gratitude, agreement, ambiguous, implicit, group membership, sympathy).

The span-level model also proved effective for binary detection by classifying a comment as candy speech if at least one positive span was detected.

## Intended Uses

- **Research:** Analysis of positive/supportive communication in German social media.
- **Applications:** Social media analytics, conversational AI safety (mitigating sycophancy), computational social science.
- **Not for:** Deployments without fairness/robustness testing on out-of-domain data.

## Performance

- **Dataset:** 46k German YouTube comments, annotated with candy speech spans.
- **Training Data Split:** 37,057 comments (train), 9,229 (test).
- **Shared Task Results:**

  - **Subtask 1 (binary detection):** Positive F1 = **0.891** (ranked 1st)
  - **Subtask 2 (span detection):** Strict F1 = **0.631** (ranked 1st)

## Training Procedure

- **Architecture:** XLM-RoBERTa-Large + linear classification layer (BIO tagging, 21 labels including “O”).
- **Optimizer:** AdamW
- **Learning Rate:** Peak 2e-5 with linear decay and warmup (500 steps).
- **Epochs:** 20 (with early stopping).
- **Batch Size:** 32
- **Regularization:** Dropout (0.1), weight decay (0.01), gradient clipping (L2 norm 1.0).
- **Postprocessing:** BIO tag correction and subword alignment.

## Limitations

- **Domain Specificity:** Trained only on German YouTube comments; performance may degrade on other platforms, genres, or languages.
- **Overlapping Spans:** Cannot handle overlapping spans, as they were rare (<2%) in the training data.
- **Biases:** May reflect biases present in the dataset (e.g., demographic skews in YouTube communities).
- **Generalization:** Needs evaluation before deployment in real-world moderation systems.

## Ethical Considerations

- **Positive speech detection** is less studied than toxic speech, but automatic labeling of “supportiveness” may reinforce cultural biases about what counts as “positive.”
- Must be complemented with **human-in-the-loop moderation** to avoid misuse.

## Citation

If you use this model, please cite:

```
@inproceedings{thelen-etal-2025-aixcellent,
    title = "{AI}xcellent Vibes at {G}erm{E}val 2025 Shared Task on Candy Speech Detection: Improving Model Performance by Span-Level Training",
    author = "Thelen, Christian Rene  and
      Blaneck, Patrick Gustav  and
      Bornheim, Tobias  and
      Grieger, Niklas  and
      Bialonski, Stephan",
    editor = "Wartena, Christian  and
      Heid, Ulrich",
    booktitle = "Proceedings of the 21st Conference on Natural Language Processing (KONVENS 2025): Workshops",
    month = sep,
    year = "2025",
    address = "Hannover, Germany",
    publisher = "HsH Applied Academics",
    url = "https://aclanthology.org/2025.konvens-2.33/",
    pages = "398--403"
}
```
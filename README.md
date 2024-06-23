<div align="center">

<h1>In-Context Curriculum Learning</h1>


**A simple, effective demonstration ordering method**
</div>

In-Context Curriculum Learning (ICCL)  encompasses two pivotal segments: curriculum schedule design and curriculum learning implementation.


Here is an example of curriculum-based demonstrations ordering. 

1. ["Literature on the topic of the current study spans across many areas, including verb classification, semiotics, sign language and learning.", "abstract words can be more challenging to learn and memorise.", "neutral"],
2. ["The Lang-8 corpus has often only one corrected sentence per learner sentence, which is not enough for evaluation.", "we ensured that our evaluation corpus has multiple references.", "reasoning"],
3. ["Essentially, that work examines how a word gains new senses, and how some senses of a word may become deprecated.", "here we examine how different words compete to represent the same meaning, and how the degree of success of words in that competition changes over time.", "contrasting"],
4. ["As a complementary area of investigation, a plausible direction would be to shift the focus from the decomposition of words into morphemes, to the organization of words as complete paradigms.", "instead of relying on sub-word units, identify sets of words organized into morphological paradigms (Blevins, 2016).”", "entailment"]

## Benchmark Result

Experiments on five LLMs, including both publicly available and proprietary models, across three NLP tasks. For each model, demonstrations of prompt are organized according to either random or curriculum-based order.


The ICCL effective model is marked by "†".

<!-- | Task   | Model   | Random | Human-Curriculum | Self-Curriculum |
|--------|-------------|-----------------|----------|----------------------|
|SciCite| Llama 2-13B†      | 43.66           | 46.29     |  -          |
| SciCite  | Llama 2-70B | 67.98           | 64.03     |   -         |
|SciCite| Mixtral-8x7B†   | 67.07           | 67.80     |   66.04         |
|SciCite| Qwen-72B-Chat†       | 67.07           | 75.69     |   -         |
|SciCite| GPT-4       | 77.05           | 71.88     |   -         |
|SciNLI| Llama 2-13B      | 21.58           | 19.12     |  -          |
| SciNLI | Llama 2-70B† | 30.05           | 35.67     |   35.67         |
|SciNLI| Mixtral-8x7B†   | 43.18           | 51.56     |  46.04          |
|SciNLI| Qwen-72B-Chat        | 51.54           | 51.17     |   -         |
|SciNLI| GPT-4       | 55.92           | 55.17     |   -         |
|SciERC| Llama 2-13B†      | 31.58           | 33.26     |  -          |
| SciERC  | Llama 2-70B† | 31.39           | 32.94    |   29.12         |
|SciERC| Mixtral-8x7B†   | 32.95           | 33.41     |   32.22         |
|SciERC| Qwen-72B-Chat†        | 29.84           | 30.63     |   -         |
|SciERC| GPT-4       | 38.34           | 36.58     |   -         | -->


## Install

### Install from pip

```shell
pip install -r requirements.txt
```




## Quick Start


### How to run infernece

`python inference/{model_name}.py`

### How to run evaluation

`python evaluation/{dataset_name}.py`





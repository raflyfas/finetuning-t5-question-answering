# Fine-Tuning T5 for Generative Question Answering

This repository contains an end-to-end implementation of **generative question answering** using the **T5-base** encoder–decoder Transformer model. The model is fine-tuned on the **SQuAD (Stanford Question Answering Dataset)** and evaluated using standard QA metrics.

The project demonstrates the complete pipeline, including data preprocessing, model training, quantitative evaluation, and qualitative inference examples.

---

## Group 6 – Deep Learning

| Nama | NIM |
|---------|------|
| Rahmanda Afebrio Yuris Soesatyo | 1103223024 |
| Rafly Fasha Purnomo Putra | 1103223050 |

---
## Task Description

**Task:** Encoder–Decoder (Seq2Seq)
**Model:** T5-base
**Dataset:** SQuAD
**Task Type:** Generative Question Answering

Unlike classification-based QA, this task requires the model to **generate answer text** given a question and its corresponding context.

---

## Repository Structure

```
finetuning-t5-question-answering/
├── finetuning_t5_question_answering.ipynb  # Main notebook
├── requirements.txt                        # Required dependencies
└── README.md                               # Project documentation
```

---

## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** A GPU is recommended for faster training and inference.

---

## Methodology

The workflow implemented in this project consists of the following steps:

1. **Dataset Loading**

   * Load the SQuAD dataset using Hugging Face Datasets.

2. **Data Preprocessing**

   * Format inputs as:

     ```
     question: <question> context: <context>
     ```
   * Tokenize inputs and target answers using the T5 tokenizer.

3. **Model Fine-Tuning**

   * Fine-tune the pre-trained T5-base model using `Seq2SeqTrainer`.
   * Training is performed on a subset of the dataset to ensure computational efficiency.

4. **Evaluation**

   * Quantitative evaluation using:

     * **Exact Match (EM)**
     * **F1 Score**
   * Qualitative evaluation through sample inference results.

5. **Manual Question Answering**

   * Demonstration of model inference using custom user-provided context and questions.

---

## Evaluation Results

| Metric      | Score |
| ----------- | ----- |
| Exact Match | 80.00 |
| F1 Score    | 83.05 |

These results indicate that the fine-tuned model is able to generate accurate and contextually relevant answers on the validation dataset.

---

## Qualitative Examples

Several inference examples are provided in the notebook to illustrate how the model answers different questions from distinct contexts. The predictions are compared directly with ground truth answers to assess correctness qualitatively.

---

## Dependencies

Main libraries used in this project:

* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Evaluate
* NumPy
* Pandas
* Matplotlib
* Seaborn
* TQDM

All dependencies are listed in `requirements.txt`.


---

## Notes

This project is intended for academic and educational purposes, demonstrating practical usage of encoder–decoder Transformer models for generative question answering tasks.

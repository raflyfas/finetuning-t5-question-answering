# Fine-Tuning T5-base for Generative Question Answering

## Group 6 – Deep Learning

| Name                 | Student ID  |
| -------------------- | ----------- |
| Rahmanda Afebrio Yuris Soesatyo	|1103223024|
|Rafly Fasha Purnomo Putra	|1103223050|

---

## Overview

This project focuses on fine-tuning **T5-base**, an encoder–decoder Transformer model, for **generative question answering** using the **SQuAD (Stanford Question Answering Dataset)**.
Unlike extractive QA approaches, the model is trained to **generate answer text** directly given a question and its corresponding context.

All stages of the workflow—including preprocessing, model training, quantitative evaluation, and qualitative analysis—are implemented within a single Jupyter Notebook.

---

## Pre-trained Model

The fine-tuned model artifacts are available at the following link:

**Final Model:**
[https://drive.google.com/drive/folders/1U6mfZZqYC82-enh_OpDFlC8jBpXF2htB](https://drive.google.com/drive/folders/1U6mfZZqYC82-enh_OpDFlC8jBpXF2htB)

The directory contains the fine-tuned T5 model and tokenizer required for inference.

---

## Dataset

The dataset used in this project is **SQuAD (Stanford Question Answering Dataset)**.

* **Task**: Generative Question Answering
* **Source**: Hugging Face
* **Dataset Link**: [https://huggingface.co/datasets/rajpurkar/squad](https://huggingface.co/datasets/rajpurkar/squad)
* **Input**: Question + context paragraph
* **Output**: Answer text span

---

## Objectives

* Fine-tune an encoder–decoder Transformer model for generative QA
* Apply sequence-to-sequence learning using T5-base
* Generate textual answers from question–context pairs
* Evaluate model performance using standard QA metrics (Exact Match and F1)
* Analyze model behavior through qualitative inference examples

---

## Model and Dataset

### Model

* **Base Model**: t5-base
* **Architecture**: Encoder–Decoder Transformer (Seq2Seq)
* **Pre-training Objective**: Text-to-text transfer learning
* **Fine-Tuning Approach**: Full fine-tuning on QA task

### Dataset

* **Dataset**: SQuAD
* **Task Type**: Generative Question Answering
* **Answer Type**: Short factual text answers

---

## Methodology

### Input Formatting

Each data sample is reformatted following the T5 convention:

```
question: <question> context: <context>
```

The target output corresponds to the ground truth answer text provided in the dataset.

### Training Strategy

The model is fine-tuned using a sequence-to-sequence objective, where the encoder processes the input prompt and the decoder generates the answer token by token.
Training is performed on a subset of the dataset to ensure computational efficiency.

---

## Training Configuration

| Parameter          | Value      |
| ------------------ | ---------- |
| Training Samples   | 1000       |
| Validation Samples | 100        |
| Epochs             | 2          |
| Batch Size         | 8          |
| Learning Rate      | 2e-5       |
| Optimizer          | AdamW      |
| Scheduler          | Linear     |
| Max Input Length   | 512        |
| Max Output Length  | 64         |
| Precision          | FP16 (GPU) |

---

## Evaluation Results

Evaluation was conducted on the validation subset using standard QA metrics

| Metric               | Score      |
| -------------------- | ---------- |
| **Exact Match (EM)** | **80.00%** |
| **F1 Score**         | **83.05%** |

### Interpretation

* **Exact Match** reflects the percentage of predictions that exactly match the reference answer
* **F1 Score** captures partial overlap between generated and reference answers

These results indicate that the model is able to generate accurate and contextually relevant answers despite being trained on a limited subset of data.

---

## Qualitative Analysis

Qualitative evaluation demonstrates that the model performs well on factual questions with clearly stated answers in the context.
The model is able to distinguish between similar entities (e.g., team names, locations, and awards) and generate precise answers.

Errors typically occur when multiple plausible answer spans exist or when the context contains highly similar named entities.

---

## Repository Structure

```
finetuning-t5-question-answering/
├── finetuning_t5_question_answering.ipynb  # Main notebook
├── requirements.txt                        # Required dependencies
└── README.md                               # Project documentation
```

---

## Notes

* Evaluation metrics are computed using the official SQuAD evaluation protocol
* Training is performed on a subset of the dataset for efficiency
* Results may vary depending on decoding strategy and random seed

---

## Conclusion

This project demonstrates that fine-tuning **T5-base** for generative question answering can achieve strong performance on the SQuAD dataset, even with limited training data.
The encoder–decoder architecture effectively models the relationship between questions and context, producing accurate and meaningful answer text.

This implementation highlights the effectiveness of sequence-to-sequence Transformers for QA tasks in practical, resource-constrained settings.

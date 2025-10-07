# 🗣️ Post-ASR Correction in Hindi  
**EACL 2026 Submission**  
**Comparing LMs and LLMs in Low-Resource Scenarios**

This repository accompanies our EMNLP 2025 paper:  
**"Post ASR Correction in Hindi: Comparing Language Models and Large Language Models in Low-Resource Scenarios."**  
We benchmark the effectiveness of fine-tuned models (`ByT5`, `mT5`) and large-scale LLMs (`LLaMA`, `ChatGPT-4o-mini`) for improving the quality of Hindi ASR transcriptions.

📄 [Read the Paper (PDF)](./EACL2026___Post_ASR_Correction_in_Hindi__Comparing_Language_Models_and_Large_Language_Models_in_Low_Resource_Scenarios.pdf)

---

## 🔍 Overview

Automatic Speech Recognition (ASR) systems for low-resource languages often struggle due to limited training data and linguistic complexity. Our work frames **post-ASR correction** as a high-overlap text editing task and evaluates:

- Fine-tuned LMs: `mT5`, `ByT5`
- Large LLMs: `LLaMA`, `ChatGPT-4o-mini`
- Correction across **in-domain** and **out-of-domain** speech
- Effectiveness of **ICL vs. fine-tuning**

---

## 🧪 Key Findings

- 🔁 **Inverse Scaling Trend**: Mid-sized LLMs underperform compared to small fine-tuned models.
- ✨ **ByT5**: Best for character-level corrections (transliteration, segmentation).
- 🌐 **mT5**: Best for semantic consistency and domain adaptation.
- 📉 **LLMs** like LLaMA and ChatGPT-4o-mini degrade in low-resource Hindi settings.

---

## 📁 Project Structure

- ByT5_finetuning.py # Fine-tune ByT5 on ASR hypotheses
- ByT5_inference.py # Inference with ByT5
- mT5_finetunig.py # Fine-tune mT5 on Hindi ASR data
- mT5_inference.py # Inference with mT5
- LLama_finetune.py # Fine-tune LLaMA on ASR hypotheses
- LLama_test.py # Evaluate LLaMA on test set
- LLama_evaluate.py # Compute WER/CER
- LLama_compute_metrics.py # Error-type specific metrics
- LLama_filter_dataset.py # Data filtering & preparation
- LLama_prompt.txt # Prompt template for LLaMA/ChatGPT ICL
- ChatGPT_inference.py # In-context learning using ChatGPT
- EMNLP2025_*.pdf # Paper PDF

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Fine-Tune ByT5 / mT5
```bash
python ByT5_finetuning.py --train_file data/train.tsv --output_dir checkpoints/byt5
python mT5_finetunig.py --train_file data/train.tsv --output_dir checkpoints/mt5
```

### 3. Run Inference
```bash
python ByT5_inference.py --model_dir checkpoints/byt5 --input_file data/test.tsv
python mT5_inference.py --model_dir checkpoints/mt5 --input_file data/test.tsv
```

### 4. ChatGPT ICL Inference
```bash
python ChatGPT_inference.py --input_file data/test.tsv --prompt LLama_prompt.txt
```

### 5. Evaluate WER and Error Categories
```bash
python LLama_evaluate.py --ref_file data/test_ref.txt --pred_file outputs/predictions.txt
python LLama_compute_metrics.py
```

### Example Prompt (`LLama_prompt.txt`)
```bash
You are given an ASR hypothesis of a spoken utterance.
Your job is to correct:
1. English word transliteration errors
2. Number recognition errors (Hindi/English)
3. Compound word splits
4. Word segmentation mistakes
5. Underrepresented graphemes

Output only the final corrected version.

Hypothesis: ratha yātrā ke lie jānabūjhakara vāna t.yūrest.a dvārā taitālı̄sa minat.a kı̄ derı̄ kı̄ gaı̄ hai
```

```bash
The following Hindi ASR transcription may or may not contain errors. 
If you find any spelling, grammatical, or meaning-related errors, correct them and return the corrected version without any explanation. 
If the text is already correct, return it unchanged without any explanation.

Hypothesis: ratha yātrā ke lie jānabūjhakara vāna t.yūrest.a dvārā taitālı̄sa minat.a kı̄ derı̄ kı̄ gaı̄ hai
```


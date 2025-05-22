# 🧠 Robustness Meets Knowledge: Probing LLM Weaknesses via Perturbation Sensitivity

## Overview

Large Language Models (LLMs) often exhibit inconsistent robustness to input perturbations and demonstrate uneven proficiency across knowledge domains. While prior research has independently explored robustness and model understanding, this project investigates their intersection.

**Core Research Question:**
_Is a model’s lack of robustness to input variations indicative of a deeper lack of knowledge in the associated skill domain?_

---

## 🔍 Motivation

LLMs, while powerful, can be brittle—struggling with slightly reworded questions or specific domains. Understanding whether this brittleness is a symptom of shallow understanding can guide both evaluation practices and model improvement. By exploring the correlation between perturbation sensitivity and conceptual weakness, this research aims to:

- Illuminate blind spots in LLM capabilities.
- Improve evaluation strategies by combining robustness and knowledge profiling.

---

## 📊 Data & Tools

### Datasets

- **DOVE**
  A dataset of prompt perturbations derived from multiple evaluation benchmarks. For each original question, DOVE provides numerous rephrased variants and corresponding model outputs.

- **EvalTree (on MMLU)**
  A hierarchical evaluation framework that categorizes benchmark questions by skill type. EvalTree provides fine-grained performance metrics per skill and can generate new questions in a structured way.

### Model(s)

- **Primary:** `LLaMA3`
  Chosen due to the availability of an existing EvalTree built on MMLU.

- **Alternative:** `OLMoE`
  May be considered for deeper trace-based analysis if needed, using tools such as **OLMoE Trace**.

---

## 🧪 Evaluation Strategy

### Core Components

- **Robustness Score:**
  Measured as the model's success rate over DOVE-generated question perturbations.

- **Skill Accuracy:**
  Taken from EvalTree’s category-wise evaluation of LLaMA3 on MMLU.

### Evaluation Pipeline

1. **Robustness Assessment:**
   Compare performance on original and perturbed questions from DOVE.

2. **Category Proficiency Analysis:**
   Correlate robustness scores with EvalTree’s domain-wise accuracy ratings.

3. **Generalization Testing (optional):**
   Use EvalTree to generate new questions and measure the model’s ability to generalize within high- and low-robustness domains.

---

## 🗓️ Milestones (Target: 15/06)

### ✅ Phase 1: Correlation Analysis

- Select questions from MMLU that overlap with DOVE.
- Calculate robustness scores for each.
- Compare to category accuracy in EvalTree.
- Analyze trends or correlations between robustness and domain proficiency.

### ⏳ Phase 2: Robustness & Generalization

- Use EvalTree’s generation methods to create new MCQs:

  - From _high-robustness_ categories.
  - From _low-robustness_ categories.

- Evaluate LLaMA3's performance on these to test within-domain generalization.

### 🧪 Optional: Robustness-based EvalTree

- Construct a modified EvalTree where each node represents **average robustness** rather than accuracy.
- Analyze this “robustness tree” to uncover latent structural weaknesses or inconsistencies across domains.

---

## 📁 Structure (Updated)

```bash
.
├── Data/
│   ├── dove/                 # Raw or processed Dove-related data
│   └── evaltree/             # Data related to EvalTree
├── ExtractDove/
│   ├── extract_dove.py       # Scripts to extract or preprocess Dove data
│   └── ...                   # Additional scripts or configs
├── ExtractQuestion/
│   ├── extract_questions.py  # Scripts to extract/generate questions
│   └── ...                   # Additional scripts or configs
├── EvalTree/                 # Forked EvalTree project (converted to normal folder)
│   ├── src/                  # (Example) source code from EvalTree
│   └── ...                   # Other contents from the EvalTree repo
├── README.md
└── requirements.txt
```

---

## 🚀 Getting Started

1. Clone this repo.
2. Set up the environment:

   ```bash
   pip install -r requirements.txt
   ```

3. Run analysis scripts in `/analysis` to reproduce results from Phase 1 and Phase 2.

---

## 📌 Notes

- All code assumes access to the LLaMA3 model with EvalTree support. You may substitute with OLMoE where traceability or tooling support is stronger.
- DOVE dataset must be preprocessed to align questions with EvalTree skill mappings.

---

## ✍️ Authors & Acknowledgements

This research builds on the foundation laid by the creators of DOVE, EvalTree, and MMLU. Special thanks to the teams behind LLaMA3 and OLMoE for making their models and tools available to the community.

---

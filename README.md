

# Sentiment Analysis using Deep Learning on Sentiment140 Dataset

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/TensorFlow-DeepLearning-orange?style=for-the-badge&logo=tensorflow">
  <img src="https://img.shields.io/badge/Kaggle-Dataset-20BEFF?style=for-the-badge&logo=kaggle">
  <img src="https://img.shields.io/badge/NLP-Sentiment%20Analysis-success?style=for-the-badge">
</p>

---

# Overview

This project implements an end-to-end **Sentiment Analysis Pipeline** using the **Sentiment140 Twitter Dataset** and multiple deep learning architectures.

The system performs:

- Data acquisition from Kaggle
- Text preprocessing
- Emoji and emoticon handling
- Tokenization and sequence padding
- Deep learning model training
- Performance comparison
- Real-time sentiment prediction

---

# Pipeline Architecture

```text
Kaggle Dataset
      ↓
Data Cleaning & Preprocessing
      ↓
Tokenization & Padding
      ↓
Train / Validation / Test Split
      ↓
Deep Learning Models
      ↓
Evaluation & Comparison
      ↓
Best Model Selection
      ↓
Real-Time Prediction System
```

---

# Dataset Information

## Sentiment140 Dataset

| Feature | Details |
|---|---|
| Dataset | Sentiment140 |
| Source | Kaggle |
| Size | 1.6 Million Tweets |
| Task | Binary Sentiment Classification |

### Sentiment Labels

| Label | Meaning |
|---|---|
| 0 | Negative |
| 4 | Positive |

---

# Environment Setup

---

## 📌 Mount Google Drive

### Purpose

This step connects Google Colab to Google Drive.

Since Colab sessions are temporary, mounting Google Drive ensures permanent storage of:

- datasets
- trained models
- logs
- outputs

### Output

```python
Mounted at /content/drive
```

---

## 📌 Install Dependencies

### Libraries Used

| Library | Purpose |
|---|---|
| nltk | NLP preprocessing |
| scikit-learn | ML utilities |
| seaborn | Visualization |
| pandas | Data handling |
| numpy | Numerical operations |
| tensorflow | Deep learning |

### Output

```python
Requirement already satisfied
```

---

## 📌 Install Emoji Library

### Purpose

Converts emojis into meaningful text descriptions.

### Example

```python
😂 → "face with tears of joy"
```

### Importance

Helps the model understand emotional context better.

---

# Kaggle API Configuration

---

## 📌 Upload Kaggle API Key

### Purpose

Uploads the `kaggle.json` credentials file.

### Why Needed

Required for authenticated dataset downloads from Kaggle.

---

## 📌 Configure Kaggle API

### Steps

1. Create `.kaggle` directory
2. Move `kaggle.json`
3. Set secure permissions

### Result

Kaggle API becomes accessible programmatically.

---

# Dataset Download & Organization

---

## 📌 Download Sentiment140 Dataset

### Dataset Details

- ~1.6 million tweets
- Around 80MB compressed
- Binary sentiment classification

### Output

```python
Download completed
```

---

## 📌 Create Project Structure

### Generated Folders

```text
project/
│
├── checkpoints/
├── models/
├── logs/
├── results/
└── visualizations/
```

### Additional Feature

- Unique `RUN_ID` generated using timestamps
- Prevents overwrite of experiments

---

## 📌 Unzip and Move Dataset

### Steps

- Extract ZIP file
- Move CSV into Google Drive

### Importance

Ensures permanent dataset storage.

---

# Data Loading & Structuring

---

## 📌 Load Dataset

### Operations

- Load CSV using `latin-1` encoding
- Import into pandas DataFrame

### Initial Observation

- No headers
- Raw unstructured format

---

## 📌 Data Structuring (Phase 1)

### Column Renaming

```python
['target','id','date','flag','user','text']
```

### Label Conversion

| Original | Converted |
|---|---|
| 0 | 0 (Negative) |
| 4 | 1 (Positive) |

### Final Columns Retained

```python
text
sentiment
```

---

## 📌 Dataset Integrity Check

### Verifications

- Dataset shape
- Memory usage
- Column validation
- Class balance

### Result

```text
1.6M rows
Balanced dataset
~200MB memory usage
```

---

# Text Preprocessing

---

## 📌 Conditional Column Drop

### Purpose

Avoid duplicate preprocessing columns.

### Result

```python
clean_text column not found → safe to proceed
```

---

## 📌 Text Cleaning Pipeline

### Preprocessing Steps

#### 1. Normalize Repeated Characters

```text
"soooo" → "soo"
```

#### 2. Remove Noise

- URLs
- Mentions
- Special symbols

#### 3. Handle Contractions

```text
"can't" → "can not"
```

#### 4. Emoji Conversion

```text
😂 → face with tears of joy
```

#### 5. Emoticon Handling

```text
:) → happy
:( → sad
```

#### 6. Space Normalization

Extra spaces removed.

### Output

Creates a new column:

```python
clean_text
```

---

# Preprocessing Verification

---

## 📌 Validation Checks

### Performed Checks

- Original vs cleaned text comparison
- Empty row analysis
- Word length distribution

### Observation

Approximately:

```text
3493 empty rows
```

after cleaning.

---

# Feature Engineering

---

## 📌 Add Text Length Feature

### Added Feature

```python
text_len
```

### Why Useful

- Statistical analysis
- Additional ML feature

---

## 📌 Tokenization & Padding

### Tokenization

Words converted into integer IDs.

### Configuration

| Parameter | Value |
|---|---|
| Vocabulary Size | 20,000 |
| Sequence Length | 120 |

### Example

```text
"I love AI"
↓
[15, 83, 201]
```

### Padding

Ensures fixed-size input for neural networks.

### Output Shape

```python
(100000, 120)
```

---

# Dataset Splitting

---

## 📌 Train / Validation / Test Split

### Distribution

| Dataset | Percentage |
|---|---|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

### Importance

- Prevents overfitting
- Enables fair evaluation

---

# Training Optimization

---

## 📌 Callbacks

### EarlyStopping

Stops training when validation performance stops improving.

### ReduceLROnPlateau

Reduces learning rate automatically when progress stalls.

---

# Deep Learning Architectures

---

## Models Implemented

### 1. BiLSTM

Captures sequential context in both directions.

---

### 2. BiGRU

Faster and lighter alternative to LSTM.

---

### 3. CNN

Detects local phrase-level patterns.

---

### 4. CNN + BiGRU

Combines:

- CNN → local feature extraction
- BiGRU → sequential understanding

---

### 5. BiGRU + Attention

Focuses on the most important words in a sentence.

---

# Model Training Results

| Model | Accuracy |
|---|---|
| BiLSTM | 0.7899 |
| BiGRU | 0.7883 |
| CNN | 0.7934 |
| CNN + BiGRU | **0.7936** ⭐ |
| BiGRU + Attention | 0.7907 |

---

# Best Performing Model

## CNN + BiGRU

### Why It Performs Best

The hybrid architecture combines:

| Component | Strength |
|---|---|
| CNN | Local phrase detection |
| BiGRU | Sequential context understanding |

This enables the model to capture both:

- important local patterns
- long-term contextual dependencies

---

# Model Saving

---

## 📌 Save Best Model

### Saved Components

- Trained model
- Tokenizer
- Configuration files

### Result

```text
CNN+BiGRU saved successfully
```

---

# Inference System

---

## 📌 Load Model & Predict

### Includes

- preprocessing pipeline
- tokenizer loading
- prediction function

### Result

Model becomes ready for real-time inference.

---

# Prediction Testing

---

## 📌 Test Cases Included

- Emojis
- Sarcasm
- Mixed sentiment
- Custom text

### Output

Predicted sentiment with confidence scores.

---

# Interactive Prediction Mode

---

## 📌 Real-Time User Input

Users can type custom text interactively.

### Example

```python
Enter text: I absolutely love this movie!
Prediction: Positive (98.2%)
```

### Exit

```python
Type "exit"
```

to stop the session.

---

# Technologies Used

| Category | Tools |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow / Keras |
| NLP | NLTK |
| ML Utilities | scikit-learn |
| Visualization | seaborn, matplotlib |
| Platform | Google Colab |
| Dataset Source | Kaggle |

---

# Future Improvements

- Add multilingual sentiment analysis
- Integrate transformer models (BERT, RoBERTa)
- Deploy using Flask/FastAPI
- Build Streamlit web application
- Add neutral sentiment classification

---

# Final Conclusion

This project demonstrates a complete deep learning pipeline for sentiment analysis using large-scale Twitter data.

Among all tested architectures, the **CNN + BiGRU** model achieved the best overall performance by effectively combining local feature extraction with sequential contextual understanding.

The system is modular, scalable, and suitable for future NLP experimentation or deployment.



---

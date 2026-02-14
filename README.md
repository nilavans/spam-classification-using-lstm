 # 📧 Email/SMS Spam Classification using Bidirectional LSTM

 A deep learning-based text classification system that detects spam messages using Bidirectional Long Short-Term Memory (LSTM) networks, achieving **97%+ accuracy** on real-world SMS data.

 ## 🎯 Overview

This project implements a **Bidirectional LSTM neural network** for binary text classification of spam vs. ham (legitimate) messages. The model uses word embeddings and sequential processing to capture semantic meaning and context in text messages.

**Key Highlights:**
- 🎯 **97%+ accuracy** on test data
- 📊 **0.98 ROC-AUC** score
- ⚖️ Handles **class imbalance** (13% spam rate)
- 🔄 **Reproducible** results with fixed random seeds
- 📈 **Comprehensive evaluation** with multiple metrics

---

## ✨ Key Features

### **1. Advanced Text Preprocessing**
- URL and email address removal
- Number and punctuation filtering
- Stopword removal using NLTK
- Custom tokenization pipeline

### **2. Deep Learning Architecture**
- Bidirectional LSTM for context understanding
- Word embeddings (128-dimensional)
- Dropout layers for regularization
- Optimized for imbalanced datasets

### **3. Robust Training**
- Class weight balancing
- Early stopping to prevent overfitting
- Model checkpointing for best performance
- Stratified train-test split

### **4. Comprehensive Evaluation**
- Precision-Recall curves
- ROC-AUC analysis
- Confusion matrix
- Classification reports
- Training history visualization

---

## 📊 Dataset

**Source:** SMS Spam Collection Dataset
- **Total Messages:** 5,574
- **Spam Messages:** 747 (13.4%)
- **Ham Messages:** 4,827 (86.6%)
- **Train/Test Split:** 80/20 stratified

### Sample Data

| Type | Message |
|------|---------|
| Ham  | "Hi, can we meet at 5pm tomorrow?" |
| Spam | "WINNER!! You've won £1000 cash prize! Call now!" |
| Ham  | "Don't forget to pick up milk on your way home" |
| Spam | "Congratulations! Click here for your FREE iPhone!" |

---

## 🏗️ Model Architecture

```
Layer (type)                 Output Shape              Param #   
=================================================================
Embedding                    (None, 100, 128)          640,000    
Bidirectional LSTM           (None, 128)               98,816     
Dropout (0.5)                (None, 128)               0         
Dense (ReLU)                 (None, 32)                4,128      
Dropout (0.3)                (None, 32)                0         
Dense (Sigmoid)              (None, 1)                 33        
=================================================================
Total params: 742,977
Trainable params: 742,977
```

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Vocabulary Size** | 5,000 | Captures most common words while reducing noise |
| **Embedding Dimension** | 128 | Balances expressiveness with computational efficiency |
| **LSTM Units** | 64 | Sufficient capacity for sequence patterns |
| **Max Sequence Length** | 100 | Covers 95th percentile of message lengths |
| **Batch Size** | 64 | Better convergence than larger batches |
| **Dropout Rate** | 0.5, 0.3 | Prevents overfitting |
| **Optimizer** | Adam | Adaptive learning rate |
| **Learning Rate** | 0.001 (default) | Standard for Adam optimizer |

---

## 📈 Results

### Performance Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 97.31% |
| **Precision (Spam)** | 91.23% |
| **Recall (Spam)** | 94.67% |
| **F1-Score (Spam)** | 92.92% |
| **ROC-AUC** | 0.9834 |
| **Average Precision** | 0.9421 |

### Confusion Matrix

|  | Predicted Ham | Predicted Spam |
|--|---------------|----------------|
| **Actual Ham** | 961 | 4 |
| **Actual Spam** | 8 | 142 |

**Interpretation:**
- **True Negatives (961):** Ham messages correctly identified
- **False Positives (4):** Ham messages incorrectly flagged as spam (0.4%)
- **False Negatives (8):** Spam messages missed (5.3%)
- **True Positives (142):** Spam messages correctly caught (94.7%)

### Training History

The model converged after **~10 epochs** with early stopping:

![Training History](results/training_history.png)
*Training and validation metrics across epochs showing stable convergence*

![Precision-Recall Curve](results/precision_recall_curve.png)
*Precision-Recall curve demonstrating excellent performance across thresholds*

---

## 🔬 Methodology

### 1. Data Preprocessing

**Text Cleaning Pipeline:**
```python
1. Lowercase conversion
2. URL removal (http/https/www patterns)
3. Email address removal
4. Number removal
5. Punctuation removal
6. Extra whitespace removal
7. Tokenization
8. Stopword removal
```

### 2. Feature Engineering

- **Tokenization:** Convert text to sequences of integers using top 5,000 most frequent words
- **Padding:** Standardize sequence length to 100 tokens
- **Out-of-Vocabulary:** Handle unknown words with `<OOV>` token

### 3. Handling Class Imbalance

**Challenge:** Only 13.4% of messages are spam

**Solutions Implemented:**
- ✅ **Class Weighting:** Spam class weighted 3.73x more than ham
- ✅ **Stratified Splitting:** Preserves class distribution in train/test sets
- ✅ **Evaluation Metrics:** Focus on Precision-Recall over accuracy

### 4. Model Training

**Training Strategy:**
- **Optimiser:** Adam with default learning rate (0.001)
- **Loss Function:** Binary Cross-Entropy with class weights
- **Batch Size:** 64 (reduced from 512 for better convergence)
- **Early Stopping:** Patience of 3 epochs on validation loss
- **Callbacks:**
  - Early stopping to prevent overfitting
  - Model checkpoint to save best weights

### 5. Evaluation

**Comprehensive metrics:**
- Classification report (precision, recall, F1-score)
- Confusion matrix analysis
- ROC-AUC score
- Precision-Recall curves
- Average Precision score

---

## 📚 References

1. **Dataset:** [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
2. **LSTM Paper:** Hochreiter & Schmidhuber (1997) - "Long Short-Term Memory"
3. **Bidirectional RNN:** Schuster & Paliwal (1997) - "Bidirectional Recurrent Neural Networks"
4. **Class Imbalance:** He & Garcia (2009) - "Learning from Imbalanced Data"

---

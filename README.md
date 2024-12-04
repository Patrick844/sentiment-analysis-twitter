
# Sentiment Analysis with PyTorch and Attention Mechanism

## **Overview**
This project provides a comprehensive sentiment analysis pipeline using PyTorch, with a focus on integrating attention mechanisms for better interpretability. It includes data preprocessing, feature engineering, model building with LSTMs and attention, and inference pipelines.

---

## **Key Features**
1. **Data Preprocessing**:
   - Removal of missing values.
   - Handling English contractions.
   - Emoji detection and removal.
   - Text normalization (lowercasing, punctuation removal, etc.).
   - Spell correction using both rule-based methods and language models.

2. **Feature Engineering**:
   - Tokenization using BERT-based tokenizers.
   - Encoding sentiment categories into numerical labels.
   - Padding for sequence length standardization.

3. **Model Architecture**:
   - Bidirectional LSTM for contextual word representations.
   - Additive Attention Mechanism to compute the relevance of each word in a sentence.
   - Fully connected layers for classification into sentiment categories (Positive, Negative, Neutral, Irrelevant).

4. **Training and Validation**:
   - Training loop with gradient updates and loss computation.
   - Validation loop for performance evaluation.

5. **Inference Pipeline**:
   - Preprocessing of single text inputs for predictions.
   - Lightweight and optimized for real-time sentiment analysis.

---

## **Setup**

### **Requirements**
- Python 3.8 or higher
- PyTorch
- Transformers by Hugging Face
- SymSpellPy
- SpaCy
- Swifter
- TQDM
- Emoji
- SpellChecker

Install dependencies:
```bash
pip install torch transformers symspellpy spacy swifter tqdm emoji pyspellchecker
python -m spacy download en_core_web_sm
```

---

## **Data**
- **Dataset**: Twitter Entity Sentiment Analysis (downloaded via KaggleHub).
- The data contains tweets labeled with sentiment categories: Positive, Neutral, Negative, or Irrelevant.

---

## **Usage**

### **Preprocessing**
The pipeline processes the raw data by cleaning, normalizing, and tokenizing it. Key steps include:
```python
df_train, max_length = preprocess_pipeline(df_train, "train")
df_val, _ = preprocess_pipeline(df_val, "val", max_length)
```

### **Training**
Run the training loop:
```python
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### **Inference**
Use the inference pipeline to predict the sentiment of a single text:
```python
text = "This movie was fantastic!"
prediction = preprocess_text_for_inference(text, tokenizer)
print(f"Predicted Sentiment: {prediction}")
```

---

## **Model Architecture**

The core model includes:
- **Embedding Layer**: Converts tokens into dense vectors.
- **Bidirectional LSTM**: Captures contextual dependencies in both directions.
- **Attention Mechanism**: Highlights important words contributing to sentiment.
- **Fully Connected Layer**: Outputs sentiment probabilities.

---

## **Results**
The training and validation pipelines produce metrics such as accuracy and loss. You can monitor these values during training to ensure proper convergence and generalization.

---

## **Folder Structure**
```
.
├── data/
│   ├── twitter_training.csv
│   ├── twitter_validation.csv
├── model/
│   ├── sa_twitter.pt  # Saved model weights
├── pytorch_lightning_sa.py  # Main pipeline and training script
├── README.md
```

---

## **Future Improvements**
1. Integrate Transformer-based models like BERT for enhanced performance.
2. Add multilingual support for sentiment analysis.
3. Implement hyperparameter tuning for optimal results.

---

## **Acknowledgments**
Special thanks to:
- Hugging Face for Transformers.
- SymSpellPy for efficient spell-checking.
- The Kaggle community for datasets and resources.

# Rock vs Mine Detection (IEEE Project)

This project classifies sonar signals as either:

- `Rock`
- `Mine`

The implementation is provided in a Jupyter notebook and compares multiple machine learning and deep learning models on sonar feature data.

## Project Overview

The notebook includes an end-to-end workflow:

1. Data loading
2. Data preprocessing (encoding, normalization, split)
3. Model training
4. Model evaluation (classification report + confusion matrix)
5. Performance comparison across models

## Repository Structure

- `Rock_vs_Mine.ipynb`: Main notebook with complete pipeline
- `sonar_dataset.csv`: Sonar dataset used for training/testing
- `IEEE dox.pdf`: Project document/report
- `IEEE certificate.jpeg`: Certificate/image asset
- `redme.md`: Legacy file (kept for compatibility)

## IEEE Documents

### IEEE Project Document

- [Open IEEE dox.pdf](IEEE%20dox.pdf)

### IEEE Certificate

- [Open IEEE certificate.jpeg](IEEE%20certificate.jpeg)

Preview:

![IEEE Certificate](IEEE%20certificate.jpeg)

## Models Implemented

The following models are trained and evaluated in the notebook:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Convolutional Neural Network (CNN)
- Long Short-Term Memory (LSTM)
- Hybrid Stacking Ensemble (attempted)

## Results Summary

Based on the confusion matrices visible in notebook outputs:

| Model | Confusion Matrix (TP+TN over Total) | Accuracy |
|---|---|---|
| Logistic Regression | (478 + 476) / 1000 | 95.40% |
| SVM | (494 + 494) / 1000 | 98.80% |
| Random Forest | (493 + 490) / 1000 | 98.30% |
| CNN | (490 + 490) / 1000 | 98.00% |
| LSTM | (378 + 432) / 1000 | 81.00% |

Current best-performing model in this run: **SVM (98.80%)**.

## Tech Stack

- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- TensorFlow / Keras
- SciKeras
- Jupyter Notebook (designed for Google Colab workflow)

## Setup

### Option 1: Google Colab (recommended for current notebook)

1. Upload/open `Dmt.ipynb` in Colab.
2. Ensure the dataset path in the notebook points to your Drive file.
3. Run cells in order.

The notebook currently uses:

`/content/drive/MyDrive/DMT/dataset_synthetic_sonar.csv`

If your actual file is `sonar_dataset.csv`, update the `file_path` variable accordingly.

### Option 2: Local Jupyter

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow scikeras jupyter
```

3. Start Jupyter:

```bash
jupyter notebook
```

4. Open `Dmt.ipynb` and run all cells.

## Known Issue

The Hybrid Stacking Ensemble section currently fails with a scikit-learn/SciKeras compatibility error:

`AttributeError: 'super' object has no attribute '__sklearn_tags__'`

This usually indicates a version mismatch between `scikit-learn` and `scikeras`.

Suggested fix:

- Pin compatible versions, for example:

```bash
pip install "scikit-learn==1.5.2" "scikeras==0.13.0"
```

Then rerun the notebook.

## Future Improvements

- Add fixed dependency versions in a `requirements.txt`
- Export the best model for inference
- Add ROC-AUC, precision-recall curves, and cross-validation
- Build a small prediction UI (Streamlit/Flask)

## Author

Rama Krishna Reddy

# CIFAR-10 Interpretability Studio 🧠

An interactive analytical dashboard built with Streamlit and MLflow to explore, audit, and interpret deep learning models trained on the CIFAR-10 dataset.

## 🚀 Live Demo
https://cifar10-interpretability-studio-busruwlozm5gpn9zve8abn.streamlit.app

## ✨ Key Features
- **Dataset Explorer:** Visual statistics and interactive inspection of 10,000 test images.
- **Global Error Analysis:** Deep-dive into model failures across the entire test set using pre-calculated predictions.
- **Explainable AI (XAI):** Interpret model decisions using **Grad-CAM** (gradient-based) and **LIME** (perturbation-based) visualizations.
- **Model Passport:** Comparative analysis of multiple architectures (SimpleCNN vs. 8-layer ProCNN with BatchNorm).

## 🛠️ Tech Stack
- **Frameworks:** PyTorch, Streamlit, MLflow.
- **Interpretability:** Captum (Grad-CAM), LIME.
- **Visualization:** Plotly, Matplotlib.
- **Environment:** Poetry for dependency management.

## 📈 Research Insight
This project compares a baseline 2-layer CNN with an optimized 8-layer architecture. Through the dashboard, we demonstrate how the ProCNN model, trained on Cloud GPUs, achieved significantly higher accuracy and more focused attention maps compared to the baseline.

## ⚙️ Setup and Execution
1. Install dependencies: `pip install poetry && poetry install`
2. Run the dashboard: `poetry run streamlit run app.py`

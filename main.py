import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score

def run_classification_project():
    print("=== PROJECT 2: DATA CLASSIFICATION USING AI ===")
    
    # =====================================================================
    # 1. INPUT PHASE (Data Loading & Scaling)
    # =====================================================================
    # Load the built-in Iris dataset (150 samples, 3 classes, 4 features)
    iris = load_iris()
    X = iris.data  # Features: sepal length, sepal width, petal length, petal width
    y = iris.target  # Target classes: Setosa (0), Versicolor (1), Virginica (2)
    
    print(f"[*] Dataset loaded successfully: {X.shape[0]} samples with {X.shape[1]} features.")
    
    # Feature Scaling (StandardScaler: Mean = 0, Variance = 1) - Slide 9
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("[*] Feature scaling applied using StandardScaler.")

    # =====================================================================
    # 2. PROCESS PHASE (Train-Test Split & KNN Training)
    # =====================================================================
    # Split data into 80% Training and 20% Testing sets - Slide 10 & 17
    # 'stratify=y' ensures the classes are balanced in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[*] Data split completed: Training set = {X_train.shape[0]} samples, Test set = {X_test.shape[0]} samples.")
    
    # Instantiate the KNN Classifier with K=10 - Slide 11 & 13
    model = KNeighborsClassifier(n_neighbors=10)
    
    # Train the model (Memorize the map)
    model.fit(X_train, y_train)
    print("[*] Model trained successfully using K-Nearest Neighbors (K=10).")
    
    # Make predictions on test data
    predictions = model.predict(X_test)
    print("[*] Predictions computed for the test set.")

    # =====================================================================
    # 3. OUTPUT PHASE (Validation & Metrics)
    # =====================================================================
    print("\n=== EVALUATION RESULTS ===")
    
    # Confusion Matrix (TP, FP, FN, TN applied to multi-class) - Slide 15
    cm = confusion_matrix(y_test, predictions)
    print("1. Confusion Matrix:")
    print(cm)
    
    # Classification Report containing Precision, Recall, and F1-Score - Slide 16
    print("\n2. Detailed Classification Report:")
    report = classification_report(
        y_test, 
        predictions, 
        target_names=iris.target_names
    )
    print(report)
    
    # Specific F1-Score (Harmonic Mean) - Slide 16
    # 'weighted' is used to account for multi-class classification
    f1 = f1_score(y_test, predictions, average='weighted')
    print(f"3. Final Weighted F1-Score: {f1:.4f}")
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)
    plt.title('Confusion Matrix - Iris Classification')
    plt.ylabel('Actual Species')
    plt.xlabel('Predicted Species')
    plt.show()

if __name__ == "__main__":
    run_classification_project()
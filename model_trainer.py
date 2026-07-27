import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_models():
    # Load data
    df = pd.read_csv('user_data.csv')
    
    X = df.drop('fake', axis=1)
    y = df['fake']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # models
    models = {
        'SVM': SVC(probability=True),
        'Logistic Regression': LogisticRegression(max_iter=1000)
    }
    
    accuracies = {}
    
    if not os.path.exists('static'):
        os.makedirs('static')

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        accuracies[name] = acc
        print(f"{name} Accuracy: {acc*100:.2f}%")
        
        # Save model
        joblib.dump(model, f'{name.replace(" ", "_").lower()}_model.pkl')
        
    # Save accuracies for dashboard
    pd.DataFrame(list(accuracies.items()), columns=['Model', 'Accuracy']).to_csv('static/model_accuracies.csv', index=False)
    
    # Create accuracy comparison graph
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette='viridis')
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 1.1)
    plt.ylabel('Accuracy')
    plt.savefig('static/accuracy_graph.png')
    print("Accuracy graph saved to static/accuracy_graph.png")
    
    # Dataset overview plots
    plt.figure(figsize=(10, 6))
    sns.countplot(x='fake', data=df)
    plt.title('Dataset Balance (0=Genuine, 1=Fake)')
    plt.savefig('static/dataset_balance.png')
    
    print("Training complete.")

if __name__ == "__main__":
    train_models()

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load models
svm_model = None
lr_model = None
model_accuracies = {}

def load_resources():
    global svm_model, lr_model, model_accuracies
    try:
        svm_model = joblib.load('svm_model.pkl')
        lr_model = joblib.load('logistic_regression_model.pkl')
        
        # Load accuracies if available
        if os.path.exists('static/model_accuracies.csv'):
            df_acc = pd.read_csv('static/model_accuracies.csv')
            for index, row in df_acc.iterrows():
                model_accuracies[row['Model']] = row['Accuracy']
                
    except Exception as e:
        print(f"Error loading resources: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not svm_model or not lr_model:
        load_resources()
        
    try:
        # Get data from form
        data = request.form
        
        # Extract features in correct order
        # 'profile pic', 'nums/length username', 'fullname words', 
        # 'nums/length fullname', 'name==username', 'description length', 
        # 'external URL', 'private', '#posts', '#followers', '#follows'
        
        features = [
            float(data.get('profile_pic')),
            float(data.get('nums_len_username')),
            float(data.get('fullname_words')),
            float(data.get('nums_len_fullname')),
            float(data.get('name_eq_username')),
            float(data.get('desc_len')),
            float(data.get('ext_url')),
            float(data.get('private')),
            float(data.get('posts')),
            float(data.get('followers')),
            float(data.get('follows'))
        ]
        
        features_array = np.array([features])
        
        # Predict using both models (taking SVM as primary or average? User asked for "predict acounts fake or geniun")
        # Let's use SVM as primary as it's often more robust for this type of data, or display both?
        # User asked "train with sms,linear... show graphs... predict"
        # I'll return the result from SVM for the main prediction text.
        
        prediction = svm_model.predict(features_array)[0]
        # 0 = Genuine, 1 = Fake (based on my generator logic)
        
        result_text = "FAKE ACCOUNT DETECTED" if prediction == 1 else "GENUINE ACCOUNT"
        result_class = "danger" if prediction == 1 else "success" # for css styling
        
        return render_template('index.html', prediction_text=result_text, prediction_class=result_class)
        
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}", prediction_class="warning")

@app.route('/dashboard')
def dashboard():
    # Helper to read dataset csv for overview
    dataset_overview = {}
    if os.path.exists('user_data.csv'):
        df = pd.read_csv('user_data.csv')
        dataset_overview['total_samples'] = len(df)
        dataset_overview['fake_samples'] = len(df[df['fake'] == 1])
        dataset_overview['genuine_samples'] = len(df[df['fake'] == 0])
        dataset_overview['columns'] = list(df.columns)
        dataset_overview['preview'] = df.head().to_html(classes='table table-striped', index=False)
    
    return render_template('dashboard.html', accuracies=model_accuracies, overview=dataset_overview)

if __name__ == '__main__':
    load_resources()
    app.run(debug=True)

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import nltk


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Step 2: Define text preprocessing function
def simple_clean_text(text):
    """
    A simpler version of text cleaning that doesn't use NLTK tokenization
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    import re
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Simple word splitting
    words = text.split()
    
    # Remove stopwords if available
    try:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
    except:
        # If stopwords fail, just keep all words
        pass
    
    # Join words back into text
    cleaned_text = ' '.join(words)
    
    return cleaned_text

# Step 3: Create a function to label data based on keywords and patterns
def label_urgency(text):
    """
    Labels text as 'high', 'normal', or 'low' urgency based on keywords and patterns.
    This creates our initial labeled dataset.
    """
    text = text.lower()
    
    # Keywords indicating high urgency
    high_urgency = [
        'immediate', 'urgent', 'emergency', 'dangerous', 'hazard', 'critical',
        'serious', 'severe', 'life-threatening', 'fatal', 'extreme', 'disaster',
        'unacceptable', 'horrific', 'worst', 'terrible', 'harmful', 'unsafe',
        'threatening', 'illegal', 'breach', 'violation', 'lawsuit', 'legal action', 'crashed'
    ]
    
    # Keywords indicating low urgency
    low_urgency = [
        'minor', 'small', 'slight', 'trivial', 'inconvenience', 'suggestion',
        'recommend', 'consider', 'improvement', 'enhance', 'update', 'minimal',
        'little', 'somewhat', 'occasionally', 'rarely', 'sometimes'
    ]
    
    # Check for high urgency patterns
    if any(word in text for word in high_urgency) or \
       'not working' in text or \
       'doesn\'t work' in text or \
       'failed' in text or \
       'broken' in text:
        return 'high'
    
    # Check for low urgency patterns
    elif any(word in text for word in low_urgency) or \
         'would be nice' in text or \
         'could be better' in text:
        return 'low'
    
    # Default to normal urgency
    else:
        return 'normal'
    
df = pd.read_csv('C:/xampp/final year/complaint_classifier/complaints.csv', encoding='latin1')
df['Complaint'] = df['Complaint'].astype(str)

# Preprocess the data
df['clean_text'] = df['Complaint'].apply(simple_clean_text)
df['urgency'] = df['Complaint'].apply(label_urgency)

#Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], 
    df['urgency'],
    test_size=0.2, 
    random_state=42,
    stratify=df['urgency']  
)


# Naive Bayes pipeline
nb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1020)),
    ('classifier', MultinomialNB())
])

# Random Forest pipeline
rf_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1020)),
    ('classifier', RandomForestClassifier(random_state=42))
])

nb_param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'tfidf__max_df': [0.5, 0.75, 1.0],
    'classifier__alpha': [0.1, 0.5, 1.0]
}

nb_grid_search = GridSearchCV(
    nb_pipeline,
    nb_param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

# Training Naive Bayes model with grid search
print("Training Naive Bayes model with hyperparameter tuning...")
nb_grid_search.fit(X_train, y_train)
nb_best_model = nb_grid_search.best_estimator_
print(f"Best Naive Bayes parameters: {nb_grid_search.best_params_}")

rf_param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'tfidf__max_df': [0.75, 1.0],
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20]
}

rf_grid_search = GridSearchCV(
    rf_pipeline,
    rf_param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

# Train Random Forest model with grid search
print("Training Random Forest model with hyperparameter tuning...")
rf_grid_search.fit(X_train, y_train)
rf_best_model = rf_grid_search.best_estimator_
print(f"Best Random Forest parameters: {rf_grid_search.best_params_}")


# Evaluate Naive Bayes
nb_y_pred = nb_best_model.predict(X_test)
print("\nNaive Bayes Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, nb_y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, nb_y_pred))

# Create confusion matrix for Naive Bayes
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, nb_y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['high', 'low', 'normal'], 
            yticklabels=['high', 'low', 'normal'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Naive Bayes')
plt.tight_layout()
plt.savefig('nb_confusion_matrix.png')
plt.close()

# Evaluate Random Forest
rf_y_pred = rf_best_model.predict(X_test)
print("\nRandom Forest Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, rf_y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_y_pred))

# Create confusion matrix for Random Forest
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, rf_y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['high', 'low', 'normal'], 
            yticklabels=['high', 'low', 'normal'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest')
plt.tight_layout()
plt.savefig('rf_confusion_matrix.png')
plt.close()

# Compare model performance
models = ['Naive Bayes', 'Random Forest']
accuracies = [accuracy_score(y_test, nb_y_pred), accuracy_score(y_test, rf_y_pred)]

plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=accuracies)
plt.ylim(0, 1)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.savefig('model_comparison.png')
plt.close()

# Feature importance analysis for Random Forest
if hasattr(rf_best_model['classifier'], 'feature_importances_'):
    feature_names = rf_best_model['tfidf'].get_feature_names_out()
    feature_importances = rf_best_model['classifier'].feature_importances_
    
    # Get top 20 features
    indices = np.argsort(feature_importances)[-20:]
    plt.figure(figsize=(12, 10))
    plt.title('Top 20 Feature Importances - Random Forest')
    plt.barh(range(len(indices)), feature_importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

# Saving the best model
import joblib
joblib.dump(rf_best_model if accuracy_score(y_test, rf_y_pred) > accuracy_score(y_test, nb_y_pred) else nb_best_model, 
            'complaint_urgency_classifier.joblib')

# The Function to classify new complaints
def predict_urgency(complaint_text, model):
    """
    Predicts the urgency level of a new complaint using the trained model.
    """
    # Clean the text
    cleaned_text =simple_clean_text(complaint_text)
    
    # Make prediction
    prediction = model.predict([cleaned_text])[0]
    probabilities = model.predict_proba([cleaned_text])[0]
    
    # Get confidence score for the prediction
    confidence = max(probabilities)
    
    return {
        'urgency': prediction,
        'confidence': confidence,
        'probabilities': {
            class_label: prob for class_label, prob in zip(model.classes_, probabilities)
        }
    }

# Example of the prediction function in the works
best_model = rf_best_model if accuracy_score(y_test, rf_y_pred) > accuracy_score(y_test, nb_y_pred) else nb_best_model
test_complaint = "The software is somewhat low and I've lost all my work!"
prediction_result = predict_urgency(test_complaint, best_model)
print("\nExample Prediction:")
print(f"Complaint: {test_complaint}")
print(f"Predicted Urgency: {prediction_result['urgency']}")
print(f"Confidence: {prediction_result['confidence']:.4f}")
print("Class Probabilities:", prediction_result['probabilities'])

print("\nModel training and evaluation complete!")
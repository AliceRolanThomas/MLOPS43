from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd

# Select 5 categories
CATEGORIES = [
    'alt.atheism',
    'comp.graphics',
    'sci.med',
    'rec.sport.hockey',
    'talk.politics.misc'
]

def prepare_dataset():
    # Fetch data
    dataset = fetch_20newsgroups(subset='all', categories=CATEGORIES, remove=('headers', 'footers', 'quotes'))
    
    # Create a DataFrame
    data = pd.DataFrame({'text': dataset.data, 'category': dataset.target})
    data['category'] = data['category'].map(lambda x: dataset.target_names[x])
    
    # Sample 10 news for training and 1 for testing from each category
    sampled_data = data.groupby('category').apply(
        lambda group: pd.concat([group.head(10), group.tail(1)])
    ).reset_index(drop=True)
    return sampled_data

def train_model():
    # Prepare dataset
    data = prepare_dataset()
    X = data['text']
    y = data['category']

    # Split data into training (10 per category) and testing (1 per category)
    train_data = data.groupby('category').head(10)
    test_data = data.groupby('category').tail(1)

    X_train, y_train = train_data['text'], train_data['category']
    X_test, y_test = test_data['text'], test_data['category']

    # Vectorize text
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_vec, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_vec)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred)}")

    # Save model and vectorizer
    joblib.dump(model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

if __name__ == '__main__':
    train_model()

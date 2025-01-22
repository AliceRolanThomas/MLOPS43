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
    dataset = fetch_20newsgroups(subset='all', 
                                 categories=CATEGORIES, remove=('headers', 'footers', 'quotes'))
    # Create a DataFrame
    data = pd.DataFrame({'text': dataset.data, 'category': dataset.target})
    data['category'] = data['category'].map(lambda x: dataset.target_names[x])

    # Sample 10 entries per category
    sampled_data = data.groupby('category').head(10).reset_index(drop=True)
    return sampled_data


def split_data(data):
    # Split each category into 80% train and 20% test
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()

    for category, group in data.groupby('category'):
        train, test = train_test_split(group, test_size=0.2, random_state=42)
        train_data = pd.concat([train_data, train])
        test_data = pd.concat([test_data, test])

    return train_data, test_data


def train_model():
    # Prepare dataset
    data = prepare_dataset()
    train_data, test_data = split_data(data)

    # Separate features and labels
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

import joblib


def predict_category(text):
    # Load model and vectorizer
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    # Predict category
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return prediction


if __name__ == '__main__':
    sample_text = "The latest advancements in computer graphics"
    print(f"Predicted Category: {predict_category(sample_text)}")

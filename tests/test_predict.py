from src.predict import predict_category

def test_predict():
    text = "A significant political event happened today"
    prediction = predict_category(text)
    valid_categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'rec.sport.hockey', 'talk.politics.misc']
    assert prediction in valid_categories

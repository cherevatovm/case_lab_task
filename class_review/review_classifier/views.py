from django.shortcuts import render
from django.http import JsonResponse
import pickle
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

loaded_model = pickle.load(open('review_classifier/ml_models/sgd_model.sav', 'rb'))
loaded_vect = pickle.load(open('review_classifier/ml_models/vectorizer.sav', 'rb'))
snowball = SnowballStemmer('english')

def preprocessor(text):
    text = re.sub(r'<[^>]*>', '', text)
    emotes = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub(r'[\W]+', ' ', text.lower())) + ' '.join(emotes).replace('-', '')
    st_words = stopwords.words('english')
    text = ' '.join([snowball.stem(w) for w in text.split() if w not in st_words])
    return text

def index(request):
    return render(request, 'review_classifier/index.html')

def classify(request):
    if request.method == 'POST':
        review = preprocessor(request.POST.get('review_text', ''))
        review_tfidf = loaded_vect.transform([review])
        pred = loaded_model.predict(review_tfidf)
        proba = loaded_model.predict_proba(review_tfidf)[0, 1]
        label = 'позитивное' if pred[0] else 'негативное'
        rating = round(proba * 10, 2)
        if rating < 1:
            rating = 1
        return JsonResponse({'label': label, 'rating': rating})
    return render(request, 'review_classifier/index.html')
from django.shortcuts import render
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle
import nltk
from nltk.stem.snowball import SnowballStemmer

#loaded_model = pickle.load(open('review_classifier/models/log_res_model.sav', 'rb'))

# Create your views here.
def index(request):
    return render(request, 'review_classifier/index.html')

def classify(request):
    return render(request, 'review_classifier/index.html')
# Imports
from imports import *
from data_loader import *
from transformers import pipeline # Bert package

# Define file name
file_name = 'star_reviews.db'

def map_polarity(polarity_score):
    '''
    Given a polarity score, return value from 1 to 5 inclusive.
    Input: (Float) polarity score provided by sentiment analysis dictionary
    Output: (Int) Polarity score, remapped on a scale of 1 to 5
    '''
    if polarity_score > 0.6:
        return 5  # Strongly positive sentiment
    elif polarity_score > 0.2:
        return 4  # Positive sentiment
    elif polarity_score > -0.2:
        return 3  # Neutral sentiment
    elif polarity_score > -0.6:
        return 2  # Negative sentiment
    else:
        return 1  # Strongly negative sentiment
    

def sentiment_analysis_text_blob(db_file):
    X, Y = sql_query_raw(db_file=db_file)
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Function to calculate sentiment using TextBlob
    def analyze_sentiment(text):
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # Polarity score between -1 (negative) and +1 (positive)
        
        # Classify the sentiment on a scale of 1 to 5
        remapped_polarity = map_polarity(polarity_score=polarity)
        
        return remapped_polarity
    
    # Retrieve training predictions
    train_pred = [analyze_sentiment(text) for text in trainX]
    test_pred = [analyze_sentiment(text) for text in testX]
    
    # Return accuracy of the sentiment classification
    train_accuracy = accuracy_score(trainY, train_pred)
    test_accuracy = accuracy_score(testY, test_pred)
    
    print(f'Training accuracy: {train_accuracy * 100:.2f}%')
    print(f'Testing accuracy: {test_accuracy * 100:.2f}%')

sentiment_analysis_text_blob(db_file=file_name)

def sentiment_analysis_VADER(db_file):
    X, Y = sql_query_raw(db_file=db_file)
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Function to calculate sentiment using TextBlob
    def analyze_sentiment(text):
        # Initialize the SentimentIntensityAnalyzer
        sid_obj = SIA()
        VADER_dict = sid_obj.polarity_scores(text)  # Polarity score between -1 (negative) and +1 (positive)
        polarity = VADER_dict['compound']
        # Classify the sentiment on a scale of 1 to 5
        remapped_polarity = map_polarity(polarity_score=polarity)
        
        return remapped_polarity
    
    # Retrieve training predictions
    train_pred = [analyze_sentiment(text) for text in trainX]
    test_pred = [analyze_sentiment(text) for text in testX]
    
    # Return accuracy of the sentiment classification
    train_accuracy = accuracy_score(trainY, train_pred)
    test_accuracy = accuracy_score(testY, test_pred)
    
    print(f'Training accuracy: {train_accuracy * 100:.2f}%')
    print(f'Testing accuracy: {test_accuracy * 100:.2f}%')

sentiment_analysis_VADER(db_file=file_name)

def sentiment_analysis_BERT(db_file):
    X, Y = sql_query_raw(db_file=db_file)
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Function to calculate sentiment using TextBlob
    def analyze_sentiment(text):
        # Initialize the sentiment-analysis pipeline using a pre-trained BERT model
        model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        sentiment_analyzer = pipeline('sentiment-analysis', device=0, model=model_id)

        BERT_dict = sentiment_analyzer(text)  # Polarity score between -1 (negative) and +1 (positive)
        polarity = BERT_dict[0]['score']
        # Classify the sentiment on a scale of 1 to 5
        remapped_polarity = map_polarity(polarity_score=polarity)
        
        return remapped_polarity
    
    # Retrieve training predictions
    train_pred = [analyze_sentiment(text) for text in trainX]
    test_pred = [analyze_sentiment(text) for text in testX]
    
    # Return accuracy of the sentiment classification
    train_accuracy = accuracy_score(trainY, train_pred)
    test_accuracy = accuracy_score(testY, test_pred)
    
    print(f'Training accuracy: {train_accuracy * 100:.2f}%')
    print(f'Testing accuracy: {test_accuracy * 100:.2f}%')

sentiment_analysis_BERT(db_file=file_name)
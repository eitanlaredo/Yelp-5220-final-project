# Yelp_5220_final_project
Comparing machine learning models accuracy for star rating classification of Yelp Reviews         
Project Report: [Report](https://www.overleaf.com/6691312184zdsnyfvqtdtn#51b698)

Yelp Review Sentiment Classification

Files:
Preprocessing: Handles all data cleaning and preprocessing
Models: Handles all modeling with reported accuracies

This project focuses on sentiment analysis of Yelp reviews, aiming to classify them based on polarity (positive or negative). Yelp, a popular platform for discovering local businesses, provides a wealth of customer feedback that can offer valuable insights for business owners and potential customers alike.

The primary goal is to identify the sentiment of Yelp reviews and extract key features that contribute to either positive or negative sentiments. This analysis will help restaurant owners, new businesses, and existing merchants gain a better understanding of customer experiences and improve their offerings.

Objectives Sentiment Classification: Use machine learning models to determine whether Yelp reviews reflect positive or negative sentiment. Model Comparison: Compare accuracy of models and summarize their effiacy via variosu ML scores and metrics

Future work: 
* Keyword Extraction: Identify words and features that contribute to sentiment polarity using techniques like word clouds and regression analysis. Emplying these words to better enhance data preprocessing and intermediary data curation to improve the accuracy of the model.
* Applying machine learning to sentiment analysis lexicons to change polarity weights of words based on training with the labeled data.
        

Data Yelp Dataset: The dataset contains over 7 million records of yelp reviews, including the review_ids, restraunts, food category, funny/cool/useful tags, and 5 star ratings. To make the project more straightforward, accessory features were trimmed, leaving just the written yelp reviews and their star ratings.

Models explored: Naive Bayes, Random Forest, Convolutional Neural Networks, Sentiment Analysis Lexicons using VADER and TextBlob

Tools & Libraries The analysis was carried out in a Jupyter Notebook using Python, with the following libraries:

Pandas for data manipulation PyTorch and Scikit-learn for machine learning models NLTK for natural language processing tasks matplotlib for visualization

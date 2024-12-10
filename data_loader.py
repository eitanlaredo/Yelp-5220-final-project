'''
Vectorization and data loaderfunction 
*Import this file for all model trials*

Takes SQL database file (star_rewview.db) and returns a word to number vectorized dataframe of each review
Also used to house mainline functions to easily query to SQL database.
'''

# Import necessary libraries from Imports.py
from imports import *

# Define file name
file_name = 'star_reviews_record_ids.db'

def sql_query_raw(db_file, query_limit=40000):

    # First validate db file name:
    if os.path.isfile(db_file) is False:
        print('Invalid db file name')
        return None, None
        
    # Establish SQL connection and query data
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Fetch data from SQL
    cursor.execute(f"SELECT * FROM data LIMIT {query_limit}")
    tables = cursor.fetchall()

    column_names = [description[0] for description in cursor.description]
    df = pd.DataFrame(tables, columns=column_names)
    
    # Define X and Y for vectorization and train/test split
    X_raw = df['processed_text']  
    Y_raw = df['stars']      
    

    return X_raw, Y_raw

def sql_vectorized(db_file, query_limit=40000):

    X, Y = sql_query_raw(db_file, query_limit=query_limit)

    # Initialize the CountVectorizer
    vectorizer = CountVectorizer(stop_words='english')

    # Vectorize the text data
    VectorizedX = vectorizer.fit_transform(X)
    
    # Split the data into training and testing sets
    trainX, testX, trainY, testY = train_test_split(VectorizedX, Y, test_size=0.2, random_state=42)
    
    return trainX, testX, trainY, testY

def sql_embedded(db_file, query_limit=40000):
    X, Y = sql_query_raw(db_file, query_limit=query_limit)

    # Tokenize the text data
    tokenized_X = [text.split() for text in X]

    # Train a Word2Vec model
    model = Word2Vec(sentences=tokenized_X, vector_size=100, window=5, min_count=1, workers=4)

    # Get the average Word2Vec embedding for each review
    def get_embedding(text):
        word_vectors = [model.wv[word] for word in text.split() if word in model.wv]
        return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

    # Convert text to embeddings
    embedded_X = np.array([get_embedding(text) for text in X])

    return embedded_X, Y

    


def Sparse(input):
    inputSparse = csr_matrix(input)
    return inputSparse
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Read intents and responses from the JSON file
with open('intents.json') as file:
    data = json.load(file)

training_data = []
responses = {}
for i, ech in enumerate(data['intents']):

    for j, ptn in enumerate(ech['patterns']):
        tpl = ()
        tpl = tpl + (ptn,ech['tag'],)
        training_data.append(tpl)
    
    rpd = []    
    for k, res in enumerate(ech['responses']):
        rpd.append(res)
    responses[ech['tag']] = rpd   



# Preprocess the training data
corpus = [data[0] for data in training_data]
intents = [data[1] for data in training_data]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# Train the intent recognition model
naive_bayes = MultinomialNB()
naive_bayes.fit(X, intents)

def nbAnswer(user_input=''):

    # Preprocess the user input
    input_vector = vectorizer.transform([user_input])

    # Predict the intent using the trained model
    predicted_intent = naive_bayes.predict(input_vector)[0]

    # Generate and display the response
    response_array = responses.get(predicted_intent)
    if response_array:
        response = random.choice(response_array)
    else:
        response = "I'm sorry, I don't have the answer to that question."

    return response



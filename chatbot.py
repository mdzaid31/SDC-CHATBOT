from flask import Flask , request , jsonify, render_template
import random
import json
import numpy as np
import pickle 
import nltk
from nltk import WordNetLemmatizer 
import tensorflow as tf

lemmatizer = WordNetLemmatizer()

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
intents = json.loads(open('intents.json').read())

model = tf.keras.models.load_model('chatbot_model.keras')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_Words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_Words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0] 
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent' : classes[r[0]] ,'probability':str(r[1])}) 
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:
        # If no intent is recognized, return a default response
        return "I'm sorry, I didn't understand that.<br>Is there anything else I can assist you with?"
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            result = result.replace("\n","<br>")
            break
    return result

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/getresponse', methods=['POST'])
def getresponse():
    user_message = request.json.get('message')
    
    # Replace this with your chatbot logic
    ints = predict_class(user_message)
    response = get_response(ints, intents)
    return jsonify({'response': response})
# Define a route for the fines popup
if __name__ == '__main__':
    app.run(debug=True)


"""
while True:
    message = input("Enter>>   ")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)

"""
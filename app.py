import json
from flask import Flask, request
from bot import Bot

import nltk  
nltk.data.path.append('./nltk_data/')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json

PAGE_ACCESS_TOKEN = 'EAAHj1Dy0fDABAHv3TkenFvwXAQLOfZAtyZBYxg7vIIQjek9xb25PqHpeBsJr0RTG0ZB59Cjb8qZBB4FSNZBByTVZAtOFW7zd6wZBxgq0HiNAZAJd31vV6xnzCU0DjJjE3FxQuG4lXTXrMxiIKKV7RZASLY4zR2i48MdUmm8CQgOgPWZCHKrF5ZCk7QT'

app = Flask(__name__)
bot = Bot(PAGE_ACCESS_TOKEN)

with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)


training = numpy.array(training)
output = numpy.array(output)


tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

# model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
# model.save("model.tflearn")

@app.route('/', methods=['GET', 'POST'])
def webhook():
    if request.method == 'GET':
        token = request.args.get('hub.verify_token')
        challenge = request.args.get('hub.challenge')
        if token == 'secret':
            return str(challenge)
        return '400'

    else:
        data = json.loads(request.data)
        messaging_events = data['entry'][0]['messaging']
        for message in messaging_events:
            user_id = message['sender']['id']
            text_input = message['message'].get('text')
            getResp(user_id, text_input)
            print('sending')
        return '200'

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

def getResp(user_id, inp):
    results = model.predict([bag_of_words(inp, words)])[0]
    results_index = numpy.argmax(results)
    tag = labels[results_index]

    if results[results_index] > 0.6:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        resp = random.choice(responses)
    else:
       resp = "I didn't get that, please try again"
       
    bot.send_text_message(user_id,resp)

if __name__ == '__main__':
    app.run(debug=True)

    
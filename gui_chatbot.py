# we will be using the Tkinter module to build the structure of the desktop application and then we will capture the user message and again perform some preprocessing before we input the message into our trained model.
# The model will then predict the tag of the user’s message, and we will randomly select the response from the list of responses in our intents file.
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - splitting words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stemming every word - reducing to base form
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for words that exist in sentence
def bag_of_words(sentence, words, show_details=True):
    # tokenizing patterns
    sentence_words = clean_up_sentence(sentence)
    # bag of words - vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,word in enumerate(words):
            if word == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % word)
    return(np.array(bag))

def predict_class(sentence):
    # Convert sentence to bag of words representation
    p = bag_of_words(sentence, words, show_details=False)
    
    # Reshape p to match the model's expected input shape
    p = np.array([p])  # Shape: (1, 87)
    p = np.expand_dims(p, axis=1)  # New shape: (1, 1, 87) if required by the model
    
    # Predict using the model
    res = model.predict(p)[0]  # Assuming res is a 1D array of probabilities
    
    # Ensure res is a 1D array
    if len(res.shape) > 1:
        res = res.flatten()
    
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    # Sort results by probability strength
    results.sort(key=lambda x: x[1], reverse=True)
    
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result


#Creating tkinter GUI
import tkinter as tk

def send(event=None):
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("1.0", tk.END)

    if msg != '':
        ChatBox.config(state=tk.NORMAL)
        ChatBox.insert(tk.END, "You: " + msg + '\n\n')
        ChatBox.config(foreground="#333333", font=("Verdana", 12))

        # Simulate a response
        ints = predict_class(msg)
        res = getResponse(ints, intents)
        
        ChatBox.insert(tk.END, "Bot: " + res + '\n\n')
            
        ChatBox.config(state=tk.DISABLED)
        ChatBox.yview(tk.END)
 

# Initialize main window
root = tk.Tk()
root.title("Chatbot")
root.geometry("400x500")
root.resizable(width=False, height=False)


#Create Chat window
ChatBox = tk.Text(root, bd=0, bg="white", height="8", width="50", font="Arial",)

ChatBox.config(state=tk.DISABLED)

#Bind scrollbar to Chat window
scrollbar = tk.Scrollbar(root, command=ChatBox.yview, cursor="heart")
ChatBox['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = tk.Button(root, font=("Verdana",12,'bold'), text="Send", width="7", height=5,
                    bd=0, bg="#f9a602", activebackground="#3c9d9b",fg='#000000',
                    command= send )

#Create the box to enter message
EntryBox = tk.Text(root, bd=0, bg="white",width="29", height="5", font="Arial")
# Bind Enter key to send function
EntryBox.bind("<Return>", send)
EntryBox.bind("<Shift-Return>", lambda e: EntryBox.insert(tk.INSERT, '\n'))


#Place all components on the screen
scrollbar.place(x=376, y=6, height=386)
ChatBox.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

root.mainloop()
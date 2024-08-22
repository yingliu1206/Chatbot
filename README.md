# Chatbot Project

This project is a simple AI-powered chatbot built using Python, Keras, and Tkinter. The chatbot is designed to respond to user inputs based on predefined intents and patterns. It demonstrates natural language processing (NLP) techniques, machine learning model training, and a graphical user interface (GUI) for user interaction.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This chatbot project uses a neural network model to classify user inputs into predefined intents and generate appropriate responses. The model is trained on a dataset of intents, and the chatbot can be interacted with through a simple Tkinter-based GUI.

## Features

- **Natural Language Processing**: Tokenization, lemmatization, and bag-of-words techniques are used to preprocess user inputs.
- **Machine Learning**: The chatbot uses a neural network model built with Keras to classify user inputs.
- **Graphical User Interface**: A Tkinter-based GUI allows for easy interaction with the chatbot.
- **Customizable Intents**: Easily modify the chatbot's behavior by updating the `intents.json` file.

## Installation

### Prerequisites

- Python 3.x
- Required Python packages (see `requirements.txt`)

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yingliu1206/Chatbot.git
   cd Chatbot

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt

3. **Download NLTK Data**
   ```bash
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')

4. **Train the Model**: Run the training script to train the chatbot model
   ```bash
   python train_chatbot.py

5. **Run the Chatbot**: Start the chatbot with the GUI:
   ```bash
   python chatbot_gui.py


### Usage
1. Launch the chatbot GUI by running chatbot_gui.py.
2. Type your message into the text box and press "Send" or hit Enter.
3. The chatbot will respond based on the trained model.


### Project Structure
chatbot-project/
│
├── intents.json          # Contains the intents, patterns, and responses
├── train_chatbot.py      # Script to preprocess data and train the model
├── chatbot_gui.py        # Script to launch the Tkinter GUI for the chatbot
├── chatbot_model.h5      # Saved Keras model after training
├── words.pkl             # Pickle file containing the preprocessed words
├── classes.pkl           # Pickle file containing the list of classes (intents)
├── README.md             # Project documentation
└── requirements.txt      # Python dependencies


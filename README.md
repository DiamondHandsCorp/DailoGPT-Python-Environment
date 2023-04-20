DialoGPT Chatbot
This is a simple chatbot based on the DialoGPT language model using the Hugging Face Transformers library. This chatbot can generate conversational responses to user inputs in natural language.

Author
This code was written by Sean Pepper.

Requirements
To run this chatbot, you will need to install the following:

Python 3.6 or higher
PyTorch 1.0 or higher
Transformers library by Hugging Face (pip install transformers)
Usage
To start the chatbot, run the chatbot.py script. The chatbot will greet the user and prompt for input. Type in your message and hit enter to receive a response.

To stop the chatbot, type "quit" and hit enter.

Customization
You can customize the chatbot's behavior by modifying the following parameters in the chatbot.py script:

model_name: The name of the pre-trained DialoGPT model to use. You can find a list of available models on the Hugging Face website.
max_length: The maximum number of tokens to generate for each response.
no_repeat_ngram_size: The size of n-grams to avoid repeating in the generated response.
License
This code is released under the MIT License. Feel free to use and modify this code for your own projects.

# DialoGPT Chatbot

This is a Python implementation of a chatbot based on the [DialoGPT](https://github.com/microsoft/DialoGPT) model, using the Hugging Face [Transformers](https://github.com/huggingface/transformers) library. 

## Requirements

* Python 3.6 or higher
* PyTorch 1.6.0 or higher
* Transformers 4.0.0 or higher

## Usage

1. Clone this repository: `git clone https://github.com/seanpepper/dialoGPT-chatbot.git`
2. Install the required packages: `pip install -r requirements.txt`
3. Run the chatbot: `python chatbot.py`

When you run the chatbot, it will greet you with a welcome message and prompt you for input. You can type anything you want and the chatbot will respond based on the DialoGPT model's training data.

You can exit the chatbot by typing "quit" as your input.

## Customization

If you want to modify the behavior of the chatbot, you can adjust the parameters in the `chatbot.py` file:

* `model_name`: the name of the DialoGPT model to use, e.g. 'microsoft/DialoGPT-large'.
* `max_length`: the maximum length (in tokens) of the chatbot's responses.
* `no_repeat_ngram_size`: the number of tokens in a row that the chatbot's responses should avoid repeating.
* `device`: the device (e.g. 'cpu', 'cuda') to use for running the model.

You can experiment with different values for these parameters to see how they affect the chatbot's behavior.

## Acknowledgments

This chatbot implementation is based on the following resources:

* [Transformers documentation](https://huggingface.co/transformers/)
* [DialoGPT paper](https://arxiv.org/abs/1911.00536)
* [Hugging Face DialoGPT example](https://github.com/huggingface/transformers/blob/master/examples/conversational/DialoGPT.py)

## Author

This implementation was created by Sean Pepper.

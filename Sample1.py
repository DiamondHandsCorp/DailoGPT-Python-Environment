#Simple Python Chat Environment using DialogPT
#Authored by Sean Pepper April 19th, 2023
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set up model and tokenizer
model_name = 'microsoft/DialoGPT-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Set up response generation function
def generate_response(input_text, model, tokenizer, chat_history_ids=[], max_length=1024, no_repeat_ngram_size=3):
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    new_input_tokens = input_ids.clone().detach()
    attention_mask = torch.ones(new_input_tokens.shape, dtype=torch.long, device=device)
    padded_input_tokens = torch.nn.functional.pad(new_input_tokens, (0, max_length - len(new_input_tokens)), value=tokenizer.pad_token_id)
    padded_attention_mask = torch.nn.functional.pad(attention_mask, (0, max_length - len(new_input_tokens)), value=0)
    output_tokens = model.generate(padded_input_tokens, max_length=max_length, attention_mask=padded_attention_mask, no_repeat_ngram_size=no_repeat_ngram_size)
    # Concatentate the chat history ids to the input ids, if any
    if chat_history_ids is not None:
        input_ids = torch.cat([chat_history_ids, input_ids], dim=-1)

    # Generate a response
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids.to(device), attention_mask=input_ids.ne(tokenizer.pad_token_id).to(device))
        logits = outputs.logits[:, -1, :]
        new_tokens = torch.argmax(logits, dim=-1).unsqueeze(-1)
        response_text = tokenizer.decode(new_tokens)

    # Concatenate the chat history with the new tokens
    chat_history_ids = torch.cat([chat_history_ids, new_tokens], dim=-1) if chat_history_ids is not None else new_tokens

    return response_text, chat_history_ids


# Set up chatbot function
def chatbot():
    print("Chatbot: Hello! How can I help you?")
    chat_history_ids = None
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Chatbot: Goodbye!")
            break
        response_text, chat_history_ids = generate_response(user_input, model, tokenizer, chat_history_ids, device=device)
        print("Chatbot:", response_text)


# Run chatbot
chatbot()

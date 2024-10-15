import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, Counter
import random
import re
import json
import os

# Hyperparameters
hidden_size = 512
num_layers = 3
lr = 0.001
batch_size = 32
seq_length = 20
epochs = 20

# Initial text corpus
with open('instructions.txt', 'r', encoding='utf-8') as file:
    initial_text = file.read()

# Simple tokenizer function
def simple_tokenizer(text):
    return re.findall(r'\w+|[^\w\s]', text.lower())

def load_words_from_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return [word.strip().lower() for word in file.readlines() if word.strip()]
    except FileNotFoundError:
        print(f"File {filename} not found. Using default vocabulary.")
        return []

# Function to initialize vocabulary
def initialize_vocabulary(initial_text, word_file=None):
    words = set(simple_tokenizer(initial_text))
    if word_file:
        words.update(load_words_from_file(word_file))
    return sorted(list(words))

# Initialize vocabulary (optionally with a word file)
word_file = 'vocabulary.txt'  # Set to None if you don't want to use a file
words = initialize_vocabulary(initial_text, word_file)

word_to_idx = {word: i for i, word in enumerate(words)}
idx_to_word = {i: word for i, word in enumerate(words)}
n_words = len(words)

print(f"Vocabulary size: {n_words}")

class ConversationalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(ConversationalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        embeds = self.embedding(x)
        out, hidden = self.lstm(embeds, hidden)
        out = self.fc(out)
        return out, hidden
    
    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))

model = ConversationalLSTM(n_words, hidden_size, num_layers, n_words)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

class ConversationMemory:
    def __init__(self, max_length=999999999999):
        self.conversations = []
        self.votes_file = 'votes.json'
        self.user_words_file = 'user_words.json'
        self.votes = self.load_json(self.votes_file, {})
        self.user_words = Counter(self.load_json(self.user_words_file, {}))
        json.dump(self.conversations, open('assistant_data.json', 'w'))

    def load_json(self, filename, default):
        """Load data from a JSON file or return a default value."""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
        return default

    def save_json(self, filename, data):
        """Save data to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(data, f)

    def log_vote(self, response, vote):
        """Log a vote for a response and save it to the JSON file."""
        self.votes[response] = vote
        self.save_json(self.votes_file, self.votes)

    def add_exchange(self, user_input, bot_response):
        """Store the exchange and update the word counter."""
        self.conversations.append((user_input.lower(), bot_response.lower()))
        self.update_user_words(user_input)

    def update_user_words(self, user_input):
        """Update the word counter with words from the user's input."""
        words = re.findall(r'\w+', user_input.lower())
        self.user_words.update(words)
        self.save_json(self.user_words_file, dict(self.user_words))

    def get_recent_conversations(self, n=3):
        """Get the last N conversations."""
        return ''.join([f'User: {conv[0]}\nBot: {conv[1]}\n' for conv in self.conversations[-n:]])

    def get_all_conversations(self):
        """Get all conversations as a single string."""
        return ''.join([f'User: {conv[0]}\nBot: {conv[1]}\n' for conv in self.conversations])

    def get_common_user_words(self, n=10):
        """Get the most common words used by the user."""
        return [word for word, _ in self.user_words.most_common(n)]

    def get_average_vote(self):
        """Calculate the average vote from the JSON file."""
        if self.votes:
            total_votes = sum(vote for vote in self.votes.values())
            return total_votes / len(self.votes)
        return 0

def train_on_text_batch(data, epochs=5, batch_size=32):
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(data) - seq_length, seq_length):
            chunk = data[i:i + seq_length + 1]
            if len(chunk) < seq_length + 1:
                continue

            inputs = torch.tensor(chunk[:-1]).unsqueeze(0)
            targets = torch.tensor(chunk[1:])

            hidden = model.init_hidden(inputs.size(0))
            model.zero_grad()

            output, hidden = model(inputs, hidden)
            loss = criterion(output.view(-1, n_words), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}')

def train_on_all_conversations(memory, epochs=5):
    all_conversations = memory.get_all_conversations()
    if all_conversations:
        tokenized_data = simple_tokenizer(all_conversations)
        data = [word_to_idx.get(word, random.randint(0, n_words-1)) for word in tokenized_data]
        print("Training on all conversations...")
        train_on_text_batch(data, epochs=epochs, batch_size=batch_size)
    else:
        print("No conversations to train on.")

def generate_response(prompt, length=20):
    model.eval()
    h = model.init_hidden(1)
    generated = simple_tokenizer(prompt.lower())

    for word in generated:
        x = torch.tensor([[word_to_idx.get(word, random.randint(0, n_words-1))]]).long()
        _, h = model(x, h)

    user_words = set(memory.get_common_user_words())
    for i in range(length):
        x = torch.tensor([[word_to_idx.get(generated[-1], random.randint(0, n_words-1))]]).long()
        output, h = model(x, h)
        output_dist = output.data.view(-1).div(0.8).exp()

        for word in user_words:
            if word in word_to_idx:
                output_dist[word_to_idx[word]] *= 1.5

        top_word = torch.multinomial(output_dist, 1)[0]
        generated_word = idx_to_word[top_word.item()]
        generated.append(generated_word)

        if generated_word == '.' or len(generated) - len(simple_tokenizer(prompt)) >= length:
            break

    return ' '.join(generated[len(simple_tokenizer(prompt)):]).capitalize()

def chat():
    print("Chatbot: Hello! I'm ready to chat. Type 'quit' to exit.")
    memory = ConversationMemory()
    conversation_turn = 0
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        response = generate_response(user_input)
        print("Chatbot:", response)

        memory.add_exchange(user_input, response)
        conversation_turn += 1

        while True:
            vote = input("Rate the response (1-5) or 'skip': ").lower()
            if vote == 'skip':
                break
            try:
                vote = int(vote)
                if 1 <= vote <= 5:
                    memory.log_vote(response, vote)
                    break
            except ValueError:
                print("Invalid input. Try again.")

        if conversation_turn % 3 == 0:
            avg_vote = memory.get_average_vote()
            if avg_vote < 3:
                print("Average rating is low. Retraining...")
            train_on_all_conversations(memory, epochs=2)

# Initial training
memory = ConversationMemory()
train_on_all_conversations(memory, epochs=5)

# Start chatting
chat()
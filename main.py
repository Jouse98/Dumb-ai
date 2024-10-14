import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, Counter
import random
import re

# Hyperparameters
hidden_size = 512
num_layers = 3
lr = 0.001
batch_size = 32  # Increased batch size for faster training
seq_length = 100
epochs = 20

# Initial text corpus
initial_text = """
Hello! I'm a chatbot. I can learn from our conversations.
I am still stupid, but I'm learning as we speak.
Feel free to ask me questions or share information with me.
"""

# Preprocess the data
chars = sorted(list(set(initial_text.lower())))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
n_chars = len(chars)

class ConversationalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(ConversationalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden
    
    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))

model = ConversationalLSTM(n_chars, hidden_size, num_layers, n_chars)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

class ConversationMemory:
    def __init__(self, max_length=1000):
        self.conversations = []  # Store all conversations
        self.user_words = Counter()
        self.votes = []  # Store votes for each response

    def add_exchange(self, user_input, bot_response):
        # Store the entire exchange as a tuple
        self.conversations.append((user_input.lower(), bot_response.lower()))
        self.update_user_words(user_input)

    def update_user_words(self, user_input):
        words = re.findall(r'\w+', user_input.lower())
        self.user_words.update(words)

    def log_vote(self, response, vote):
        self.votes.append((response, vote))

    def get_recent_conversations(self, n=3):
        return ''.join(['User: ' + conv[0] + '\nBot: ' + conv[1] + '\n' for conv in self.conversations[-n:]])

    def get_all_conversations(self):
        return ''.join(['User: ' + conv[0] + '\nBot: ' + conv[1] + '\n' for conv in self.conversations])

    def get_common_user_words(self, n=10):
        return [word for word, _ in self.user_words.most_common(n)]

    def get_average_vote(self):
        if self.votes:
            total_votes = sum(vote for _, vote in self.votes)
            return total_votes / len(self.votes)
        return 0

def train_on_text_batch(data, epochs=5, batch_size=32):
    for epoch in range(epochs):
        total_loss = 0
        
        # Prepare batches
        for i in range(0, len(data) - seq_length, seq_length):
            chunk = data[i:i + seq_length + 1]
            if len(chunk) < seq_length + 1:
                continue

            inputs = nn.functional.one_hot(torch.tensor(chunk[:-1]), num_classes=n_chars).float().unsqueeze(0)
            targets = torch.tensor(chunk[1:])

            # Initialize hidden state for the current batch size
            hidden = model.init_hidden(inputs.size(0))  # inputs.size(0) gives the batch size (1 for one input)
            model.zero_grad()

            output, hidden = model(inputs, hidden)
            loss = criterion(output.view(-1, n_chars), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}')

def train_on_all_conversations(memory, epochs=5):
    all_conversations = memory.get_all_conversations()
    if all_conversations:
        data = [char_to_idx.get(ch, random.randint(0, n_chars-1)) for ch in all_conversations.lower()]
        print("Training on all conversations...")
        # Train using the batch size
        train_on_text_batch(data, epochs=epochs, batch_size=batch_size)
    else:
        print("No conversations to train on.")

def generate_response(prompt, length=100):
    model.eval()
    h = model.init_hidden(1)
    generated = prompt.lower()

    for ch in generated:
        x = torch.tensor([[char_to_idx.get(ch, random.randint(0, n_chars-1))]]).long()
        x = nn.functional.one_hot(x, num_classes=n_chars).float()
        _, h = model(x, h)
    
    user_words = set(memory.get_common_user_words())
    current_word = ""
    for i in range(length):
        x = torch.tensor([[char_to_idx.get(generated[-1], random.randint(0, n_chars-1))]]).long()
        x = nn.functional.one_hot(x, num_classes=n_chars).float()
        output, h = model(x, h)
        output_dist = output.data.view(-1).div(0.8).exp()
        
        # Boost probabilities of characters that could continue a user's common word
        for word in user_words:
            if word.startswith(current_word):
                next_char = word[len(current_word)]
                if next_char in char_to_idx:
                    output_dist[char_to_idx[next_char]] *= 1.5

        top_char = torch.multinomial(output_dist, 1)[0]
        generated_char = idx_to_char[top_char.item()]
        generated += generated_char

        if generated_char.isalpha():
            current_word += generated_char
        else:
            current_word = ""

        if generated_char == '\n' or len(generated) - len(prompt) >= length:
            break
    
    return generated[len(prompt):].capitalize()

def chat():
    print("Chatbot: Hello! I'm ready to chat. Type 'quit' to exit.")
    memory = ConversationMemory()  # Create an instance of the memory within the chat function
    conversation_turn = 0
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        response = generate_response(user_input)
        print("Chatbot:", response)
        
        memory.add_exchange(user_input, response)
        conversation_turn += 1
        
        # Voting on the chatbot's response
        while True:
            try:
                vote = input("Please rate the response (1-5) or type 'skip' to skip: ").lower()
                if vote == 'skip':
                    print("Skipping vote.")
                    break
                vote = int(vote)
                if 1 <= vote <= 5:
                    print(f"Thank you for voting {vote} for the response!")
                    memory.log_vote(response, vote)  # Log the vote
                    break
                else:
                    print("Please enter a valid vote (1-5) or type 'skip'.")
            except ValueError:
                print("Please enter a valid vote (1-5) or type 'skip'.")

        # Train on all conversations if needed
        if conversation_turn % 3 == 0:  # Train every 3 turns, you can adjust this
            avg_vote = memory.get_average_vote()
            if avg_vote < 3:
                print("Average response rating is low. Adjusting training focus.")
            train_on_all_conversations(memory, epochs=2)
# Initial training
memory = ConversationMemory()  # Create an instance of the memory for the initial training
train_on_all_conversations(memory, epochs=5)

# Start chatting
chat()
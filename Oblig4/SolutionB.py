import torch
import torch.nn as nn


class LongShortTermMemoryModel(nn.Module):
    def __init__(self, encoding_size, em_encoding_size):
        super(LongShortTermMemoryModel, self).__init__()
        
        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, em_encoding_size)  # 128 is the state size

    def reset(self):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, 1, 128)  # Shape: (number of layers, batch size, state size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, 128))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        x_logits = self.logits(x)
        y_train = (torch.ones(len(x_logits))*y).type(torch.LongTensor)
        return nn.functional.cross_entropy(x_logits, y_train)

char_encodings = [
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # ' ' 0
    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'h' 1
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'a' 2
    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 't' 3
    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'r' 4
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],  # 'c' 5
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],  # 'f' 6
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],  # 'l ' 7
    [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],  # 'm' 8
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],  # 's' 9
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],  # 'o' 10
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],  # 'p' 11
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],  # 'n' 12
]

encoding_size =  len(char_encodings)

x_train = torch.tensor([
[[char_encodings[1]], [char_encodings[2]], [char_encodings[3]], [char_encodings[0]]],
[[char_encodings[4]], [char_encodings[2]], [char_encodings[3]], [char_encodings[0]]], 
[[char_encodings[5]], [char_encodings[2]], [char_encodings[3]], [char_encodings[0]]], 
[[char_encodings[9]], [char_encodings[10]], [char_encodings[12]], [char_encodings[0]]],
[[char_encodings[8]], [char_encodings[2]], [char_encodings[3]], [char_encodings[3]]],
[[char_encodings[5]], [char_encodings[2]], [char_encodings[11]],[char_encodings[0]]],
[[char_encodings[6]], [char_encodings[7]], [char_encodings[2]], [char_encodings[3]]]
])

emoji_index = ["🎩", "🐀", "🐈", "🧒", "👨", "🧢","🏢"]

y_train = torch.tensor([0, 1, 2, 3, 4, 5, 6])

model = LongShortTermMemoryModel(encoding_size, len(y_train))

optimizer = torch.optim.RMSprop(model.parameters(), 0.001)
for epoch in range(100):
    for i in range(len(x_train)):
        model.reset()
        model.loss(x_train[i], y_train[i]).backward()
        optimizer.step()
        optimizer.zero_grad()

model.reset()
y = model.f(torch.tensor([[char_encodings[1]], [char_encodings[2]], [char_encodings[3]], [char_encodings[0]]]))
print("hat: ", emoji_index[y.argmax(1)[-1]])

model.reset()
y = model.f(torch.tensor([[char_encodings[4]], [char_encodings[2]], [char_encodings[3]], [char_encodings[0]]]))
print("rat: ", emoji_index[y.argmax(1)[-1]])

model.reset()
y = model.f(torch.tensor([[char_encodings[5]], [char_encodings[2]], [char_encodings[3]], [char_encodings[0]]]))
print("cat: ", emoji_index[y.argmax(1)[-1]])

model.reset()
y = model.f(torch.tensor([[char_encodings[6]], [char_encodings[7]], [char_encodings[2]], [char_encodings[3]]]))
print("flat: ", emoji_index[y.argmax(1)[-1]])

model.reset()
y = model.f(torch.tensor([[char_encodings[8]], [char_encodings[2]], [char_encodings[3]], [char_encodings[3]]]))
print("matt: ", emoji_index[y.argmax(1)[-1]])

model.reset()
y = model.f(torch.tensor([[char_encodings[5]], [char_encodings[11]]]))
print("cp: ", emoji_index[y.argmax(1)[-1]])

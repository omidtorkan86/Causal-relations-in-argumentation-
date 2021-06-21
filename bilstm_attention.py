
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_processor import DataProcessor #from the data_processor file i have imported DataProcessor class.
import matplotlib.pyplot as plt

torch.manual_seed(123)

vocab_size = 500
embedding_size = 50
num_classes = 6
sentence_max_len = 64
hidden_size = 32

num_layers = 1# lstm layer
num_directions = 2 #BiLSTM
lr = 1e-3 #learning rate
batch_size = 32
epochs = 300

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Bi-LSTM: I have created a class for Bilstm that i have got a constructor function from pytorch.
class BiLSTMModel(nn.Module):
    def __init__(self, embedding_size,hidden_size, num_layers, num_directions, num_classes):
        super(BiLSTMModel, self).__init__()

        #initialization: I have defined a series of parameter for that.
        self.input_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = num_directions
        
        #create BiLSTM MOdel: In fact i have created a LSTM layer with considering some parameters.
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers = num_layers, bidirectional = (num_directions == 2))

        #add attention layer
        self.attention_weights_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        )
        #softmax layer
        self.liner = nn.Linear(hidden_size, num_classes)
        self.act_func = nn.Softmax(dim=1)
    
    def forward(self, x):
        #lstm [seq_len, batch, input_size]
        #x [batch_size, sentence_length, embedding_size]
        x = x.permute(1, 0, 2)         #[sentence_length, batch_size, embedding_size]
        
        #batch_size
        batch_size = x.size(1)
        

        h_0 = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
        
        #out[seq_len, batch, num_directions * hidden_size]。lstm，out h_t
        #h_n, c_n [num_layers * num_directions, batch, hidden_size]
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        

        (forward_out, backward_out) = torch.chunk(out, 2, dim = 2)
        out = forward_out + backward_out  #[seq_len, batch, hidden_size]
        out = out.permute(1, 0, 2)  #[batch, seq_len, hidden_size]
        

        h_n = h_n.permute(1, 0, 2)  #[batch, num_layers * num_directions,  hidden_size]
        h_n = torch.sum(h_n, dim=1) #[batch, 1,  hidden_size]
        h_n = h_n.squeeze(dim=1)  #[batch, hidden_size]
        
        attention_w = self.attention_weights_layer(h_n)  #[batch, hidden_size]
        attention_w = attention_w.unsqueeze(dim=1) #[batch, 1, hidden_size]
        
        attention_context = torch.bmm(attention_w, out.transpose(1, 2))  #[batch, 1, seq_len]
        softmax_w = F.softmax(attention_context, dim=-1)  #[batch, 1, seq_len]
        
        x = torch.bmm(softmax_w, out)  #[batch, 1, hidden_size]
        x = x.squeeze(dim=1)  #[batch, hidden_size]
        x = self.liner(x)
        x = self.act_func(x)
        return x


def test_agro(model, agro_dataset):
    model.eval()
    loss_val = 0.0
    corrects = 0.0
    for datas, labels in agro_dataset:
        datas = datas.to(device)

        preds = model(datas) # predic a class for each data
        preds = torch.argmax(preds, dim=1)
        print(preds.size())
        print(" {}".format( preds))
        print(datas.size())

#validation
def test(model, test_loader, loss_func):
    model.eval()
    loss_val = 0.0
    corrects = 0.0
    for datas, labels in test_loader:
        datas = datas.to(device)
        labels = labels.to(device)
        
        preds = model(datas)
        loss = loss_func(preds, labels)
        
        loss_val += loss.item() * datas.size(0)

        preds = torch.argmax(preds, dim=1)
        labels = torch.argmax(labels, dim=1)
        corrects += torch.sum(preds == labels).item()
    test_loss = loss_val / len(test_loader.dataset)
    test_acc = corrects / len(test_loader.dataset)
    print("Val Loss: {}, val Acc: {}".format(test_loss, test_acc))
    return test_acc,test_loss

#train: I have ipmlemented afunction (train())so that it can trains the model then i will get it data and labels so that it can do prediction operation and it can calculate loss function
def train(model, train_loader,test_loader, agro_loader, optimizer, loss_func, epochs):
    best_val_acc = 0.0
    best_model_params = copy.deepcopy(model.state_dict())
    trian_accs,test_accs,train_losses,test_losses = [],[],[],[]
    for epoch in range(epochs):
        model.train()
        loss_val = 0.0
        corrects = 0.0
        for datas, labels in train_loader:
            datas = datas.to(device)
            labels = labels.to(device)
            
            preds = model(datas)
            loss = loss_func(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_val += loss.item() * datas.size(0)
            

            preds = torch.argmax(preds, dim=1)
            labels = torch.argmax(labels, dim=1)
            corrects += torch.sum(preds == labels).item()
        train_loss = loss_val / len(train_loader.dataset)
        train_acc = corrects / len(train_loader.dataset)
        trian_accs.append(train_acc)
        train_losses.append(train_loss)
        if(epoch % 10 == 0):
            print("Epoch: {}, Train Loss: {}, Train Acc: {}".format(epoch,train_loss, train_acc))
            test_acc,test_loss = test(model, test_loader, loss_func)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            if(best_val_acc < test_acc):
                best_val_acc = test_acc
                best_model_params = copy.deepcopy(model.state_dict())

    test_acc, test_loss = test(model, test_loader, loss_func)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    if (best_val_acc < test_acc):
        best_val_acc = test_acc
        best_model_params = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_params)
    print("best_val_accuracy: {}".format(best_val_acc))
    test_agro(model,agro_loader)

    return model,trian_accs,train_losses,test_accs,test_losses

processor = DataProcessor()
acc = 0
v_acc = 0
for i in range(10):
    train_datasets, test_datasets, agro_datasets = processor.get_datasets(vocab_size=vocab_size, embedding_size=embedding_size, max_len=sentence_max_len)
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=True)
    agro_loader = torch.utils.data.DataLoader(agro_datasets, batch_size=batch_size, shuffle = False)
    model = BiLSTMModel(embedding_size, hidden_size, num_layers, num_directions, num_classes)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.BCELoss()
    model,trian_accs,train_losses,test_accs,test_losses = train(model, train_loader, test_loader,agro_loader, optimizer, loss_func, epochs)
    acc = acc +trian_accs [-1]
    v_acc = v_acc + test_accs[-1]

print("train acc: {}".format(acc / 10))
print("valid acc: {}".format(v_acc/10))
"""print("plotting the results...")
plt.plot(range(epochs), trian_accs, label="train acc")
plt.plot(range(0, epochs + 10, 10), test_accs, label="val acc")
plt.plot(range(0, epochs + 10, 10), test_losses, label="val loss")
plt.plot(range(epochs), train_losses, label="train loss")
plt.legend()
#plt.savefig("500_3_1L_e70.png")
plt.savefig("500_3_{}.png".format(i))
plt.show()"""



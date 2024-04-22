import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset

class NNSurrogate(nn.Module):
    def __init__(self, input_size=2, hidden_1=4, hidden_2=8):
        super(NNSurrogate, self).__init__()
        
        
        self.fc1 = nn.Linear(input_size, hidden_1)
        self.bn1 = nn.BatchNorm1d(hidden_1)  # Batch normalization after first layer
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.bn2 = nn.BatchNorm1d(hidden_2)  # Batch normalization after second layer
        self.final_fc = nn.Linear(hidden_2, 1)
        self.relu = nn.ReLU()
        return

    def forward(self, x):
        x = x.to(self.fc1.weight.dtype)
        
        h1_norm = self.relu(self.bn1(self.fc1(x)))
        h2_norm = self.relu(self.bn2(self.fc2(h1_norm)))
        y = self.final_fc(h2_norm)
        return y
    
    def fit(self, train_X, train_Y):
        # Initialize your neural network
        model = self
        
        # Set hyperparameters
        learning_rate = 0.01
        num_epochs = 100
        batch_size = len(train_X)
        
        # Build dataloader 
        train_X, train_Y = train_X.float(), train_Y.float()  # Cast labels to float32
        train_dataset = TensorDataset(train_X, train_Y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        #optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)

        # Training loop
        for epoch in range(num_epochs):
            model.train()  # Set the model to training mode
            running_loss = 0.0
            for inputs, labels in train_loader:  # Iterate over batches of training data
                optimizer.zero_grad()  # Zero the parameter gradients
                outputs = model(inputs) # Forward pass
                loss = criterion(outputs, labels) # Compute loss

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Print progress every epoch
            #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

#         # Optionally, save the trained model
#         torch.save(model.state_dict(), 'model.pth')
        model.train(mode=False)
        return
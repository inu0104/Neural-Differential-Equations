import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from main import AttentiveNCDE, SimpleNCDE
import torch.nn as nn

# Experiment setup
input_dim = 3
hidden_dim = 32
attention_dim = 32
output_dim = 1

data_size = 1000
sequence_length = 50

# Generate synthetic data
X = torch.randn(data_size, sequence_length, input_dim)
y = torch.randint(0, 2, (data_size, output_dim)).float()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models, loss, and optimizer
ancde_model = AttentiveNCDE(input_dim, hidden_dim, attention_dim, output_dim)
ncde_model = SimpleNCDE(input_dim, hidden_dim, output_dim)
criterion = nn.BCEWithLogitsLoss()
ancde_optimizer = torch.optim.Adam(ancde_model.parameters(), lr=0.001)
ncde_optimizer = torch.optim.Adam(ncde_model.parameters(), lr=0.001)

# Training loop
epochs = 10

# Train models
def train_model(model, optimizer, X_train, y_train, epochs, model_name):
    print(f"Training {model_name} Model")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i in range(len(X_train)):
            time_points = torch.linspace(0, 1, steps=sequence_length)
            initial_state = torch.zeros(hidden_dim)
            input_series = X_train[i]
            target = y_train[i]

            optimizer.zero_grad()
            output = model(time_points, initial_state, input_series)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(X_train)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# Train AttentiveNCDE
train_model(ancde_model, ancde_optimizer, X_train, y_train, epochs, "AttentiveNCDE")

# Train SimpleNCDE
train_model(ncde_model, ncde_optimizer, X_train, y_train, epochs, "SimpleNCDE")

# Evaluation function
def evaluate_model(model, X_test, y_test):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(len(X_test)):
            time_points = torch.linspace(0, 1, steps=sequence_length)
            initial_state = torch.zeros(hidden_dim)
            input_series = X_test[i]
            target = y_test[i]

            output = model(time_points, initial_state, input_series)
            predicted = (torch.sigmoid(output) > 0.5).float()
            correct += (predicted == target).sum().item()
            total += target.size(0)
    accuracy = correct / total
    return accuracy

# Evaluate models
ancde_accuracy = evaluate_model(ancde_model, X_test, y_test)
ncde_accuracy = evaluate_model(ncde_model, X_test, y_test)

print(f"\nTest Accuracy of AttentiveNCDE Model: {ancde_accuracy * 100:.2f}%")
print(f"Test Accuracy of SimpleNCDE Model: {ncde_accuracy * 100:.2f}%")

# Plotting results
fig, ax = plt.subplots()
x = ['AttentiveNCDE', 'SimpleNCDE']
y = [ancde_accuracy, ncde_accuracy]
ax.bar(x, y, color=['blue', 'orange'])
ax.set_ylabel('Accuracy')
ax.set_title('Model Comparison')
plt.show()

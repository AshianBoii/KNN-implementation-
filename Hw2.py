import numpy as np
import pandas as pd

# Load the datasets
train_data = pd.read_csv('MNIST_training.csv')
test_data = pd.read_csv('MNIST_test.csv')

# Function to compute Euclidean distance between two vectors
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# KNN function
def knn_classifier(train_data, test_instance, k=3):
    # Extract features and labels
    train_features = train_data.iloc[:, 1:].values  # All columns except 'label'
    train_labels = train_data.iloc[:, 0].values  # The 'label' column
    
    # Compute distances between the test instance and all training data points
    distances = np.array([euclidean_distance(test_instance, train_feature) for train_feature in train_features])
    
    # Get the indices of the k smallest distances
    k_neighbors_idx = distances.argsort()[:k]
    
    # Get the labels of the k neighbors
    k_neighbor_labels = train_labels[k_neighbors_idx]
    
    # Find the majority class
    majority_class = np.bincount(k_neighbor_labels).argmax()
    
    return majority_class

# Function to evaluate the KNN algorithm and print results for each test instance
def evaluate_knn(train_data, test_data, k=3):
    correct = 0
    total = len(test_data)
    
    for i in range(total):
        test_instance = test_data.iloc[i, 1:].values  # Features of the test instance
        true_label = test_data.iloc[i, 0]  # True label of the test instance
        
        predicted_label = knn_classifier(train_data, test_instance, k)
        
        # Compare predicted label with the true label
        if predicted_label == true_label:
            correct += 1
        
        # Print output for each instance
        print(f'Test Instance {i+1}: True Label = {true_label}, Predicted Label = {predicted_label}')
    
    # Calculate accuracy
    accuracy = correct / total * 100
    return accuracy, correct, total

# Run KNN on the entire test set
k_value = 15  
accuracy, correct, total = evaluate_knn(train_data, test_data, k_value)

print(f'\nFinal Accuracy: {accuracy:.2f}%')
print(f'Correctly classified: {correct}/{total}')

import torch
from kan import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class KAN(torch.nn.Module):
    ...

torch.set_default_dtype(torch.float64)

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

print(device)

def load_fluid_dataset():
    """
    Load fluid simulation dataset from HDF5 file
    Expected structure:
    - 'low_res_states': (N, 2, 64, 64) - low resolution fluid states  
    - 'high_res_states': (N, 2, 64, 64) - high resolution states downsampled to 64x64
    """
    data_file = ...
    
    try:
        with ... as f:
            low_res_data = f['low_res_states'][:]
            high_res_data = f['high_res_states'][:]
            
        # Calculate errors (Difference in velocities)
        errors = high_res_data - low_res_data
        
        # Change to 1D for KAN input: (N, 2, 64, 64) -> (N, 2*64*64)
        low_res_flat = low_res_data.reshape(low_res_data.shape[0], -1)
        errors_flat = errors.reshape(errors.shape[0], -1)
        
        # Optional: normalize the data
        scaler_input = StandardScaler()
        scaler_output = StandardScaler()
        low_res_flat = scaler_input.fit_transform(low_res_flat)
        errors_flat = scaler_output.fit_transform(errors_flat)
        
        # Convert to PyTorch tensors
        data_tensor = torch.tensor(low_res_flat, dtype=torch.float64) # Or float 32?
        target_tensor = torch.tensor(errors_flat, dtype=torch.float64) # Or float 32?

    # Use to check if code is working
    except FileNotFoundError:
        print(f"File {data_file} not found. Creating dummy data for testing...")
        # Create dummy data for testing
        n_samples = 1000
        # Input: flattened (2, 64, 64) = 8192 features
        data_tensor = torch.randn(n_samples, 2 * 64 * 64, dtype=torch.float32)
        # Output: flattened error correction (2, 64, 64) = 8192 features  
        target_tensor = 0.1 * torch.randn(n_samples, 2 * 64 * 64, dtype=torch.float32)

    # Split dataset into train and test sets
    train_data, test_data, train_target, test_target = train_test_split(
        data_tensor, target_tensor, test_size=0.2, random_state=42)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_data, train_target), 
        batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_data, test_target), 
        batch_size=1, shuffle=False)

    # Input dimension: 2 channels * 64 * 64 = 8192
    input_dim = 2 * 64 * 64
    train_inputs = torch.empty(0, input_dim, device=device)
    train_labels = torch.empty(0, input_dim, device=device)  # Output same size as input
    test_inputs = torch.empty(0, input_dim, device=device)
    test_labels = torch.empty(0, input_dim, device=device)

    # Concatenate all data into a single tensor on the specified device
    for data, labels in tqdm(train_loader):
        train_inputs = torch.cat((train_inputs, data.to(device)), dim=0)
        train_labels = torch.cat((train_labels, labels.to(device)), dim=0)

    for data, labels in tqdm(test_loader):
        test_inputs = torch.cat((test_inputs, data.to(device)), dim=0)
        test_labels = torch.cat((test_labels, labels.to(device)), dim=0)

    dataset = {}
    dataset['train_input'] = train_inputs
    dataset['test_input'] = test_inputs
    dataset['train_label'] = train_labels
    dataset['test_label'] = test_labels

    return dataset

fluid_dataset = load_fluid_dataset()

print("Train data shape: {}".format(fluid_dataset['train_input'].shape))
print("Train target shape: {}".format(fluid_dataset['train_label'].shape))
print("Test data shape: {}".format(fluid_dataset['test_input'].shape))
print("Test target shape: {}".format(fluid_dataset['test_label'].shape))
print("====================================")

# Model dimensions: input=8192 (2*64*64), hidden layers customisable, output=8192
# Customize these layers as needed:
hidden_layers = [1024, 512, 256]  # Adjust these as desired
input_dim = 2 * 64 * 64  # 8192
output_dim = 2 * 64 * 64

model_width = [input_dim] + hidden_layers + [output_dim]
model = KAN(width=model_width, grid=10, k=3, seed=0, device=device)

def train_mse():
    with torch.no_grad():
        predictions = model(fluid_dataset['train_input'])
        mse = torch.nn.functional.mse_loss(predictions, fluid_dataset['train_label'])
    return mse

def test_mse():
    with torch.no_grad():
        predictions = model(fluid_dataset['test_input'])
        mse = torch.nn.functional.mse_loss(predictions, fluid_dataset['test_label'])
    return mse

results = model.fit(fluid_dataset, opt="Adam", metrics=(train_mse, test_mse), # LBFGS (apparently better for small datasets) or Adam (for large data sets)?
                      loss_fn=torch.nn.MSELoss(), steps=25, lamb=0.01, lamb_entropy=2.) 
# General model tuning
# If underfitting: Decrease lamb, increase steps, use more complex architecture
# If overfitting: Increase lamb or lamb_entropy, add more training data
# If loss plateaus: Decrease regularisation, check data quality

print(results['train_mse'][-1], results['test_mse'][-1])

# Helper function to reshape predictions back to spatial format for visualisation
def reshape_prediction(flat_prediction, batch_size=1):
    """Reshape flattened prediction back to (batch_size, 2, 64, 64)"""
    return flat_prediction.view(batch_size, 2, 64, 64)

# Example usage after training:
# with torch.no_grad():
#     sample_input = fluid_dataset['test_input'][:1]  # Take first test sample
#     error_prediction = model(sample_input)
#     error_spatial = reshape_prediction(error_prediction)  # Shape: (1, 2, 64, 64)
#     print("Error prediction shape:", error_spatial.shape)
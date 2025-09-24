import torch
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
from .model import Model

class NeuralNetwork(nn.Module, Model):
    '''
    Neural network model for regression tasks.

    Attributes
    ----------
    activation : str
        Activation function to be used in the hidden layers.
    complexity : int
        Complexity factor for determining the width of the hidden layers.
    lr : float
        Learning rate for the optimizer.
    loss : str
        Loss function to be used for training.
    layers : int
        Number of hidden layers.
    input : torch.nn.Linear
        Input layer of the neural network.
    hidden_layers : torch.nn.ModuleList
        List of hidden layers in the neural network.
    output : torch.nn.Linear
        Output layer of the neural network.
    activation_func : torch.nn.Module
        Activation function to be used in the hidden layers.
    output_activation : str or None
        Activation function to be used in the output layer.

    Methods
    -------
    initialize_weights()
        Initialize the weights of the model.
    forward(x)
        Forward pass of the model.
    fit(Xtrain, ytrain, epochs=1000, batch_size=16, verbose=0, val_split=0.25, reset_weights=True, optimizer='Adam', patience=None)
        Train the model.
    predict(X)
        Make predictions using the trained model.
    '''
    
    
    def __init__(self, Nfeatures, activation='ReLU', complexity=16, lr=0.001, 
                 loss='L1Loss', layers=2, output_activation=None, seed=None,
                 weight_initialization='xavier_uniform_', bias_initialization='zeros_', 
                 use_batch_norm=True, scale_data=False):
        """
        Initialize a NeuralNetwork object.

        Parameters
        ----------
        Nfeatures : int
            Number of input features.
        activation : str, optional
            Activation function to be used in the hidden layers, by default 'ReLU'.
        complexity : int, optional
            Complexity factor for determining the width of the hidden layers, by default 4.
        lr : float, optional
            Learning rate for the optimizer, by default 0.005.
        loss : str, optional
            Loss function to be used for training, by default 'L1Loss'.
        layers : int, optional
            Number of hidden layers, by default 3.
        output_activation : str or None, optional
            Activation function to be used in the output layer, by default None.
        seed : int or None, optional
            Random seed for reproducibility, by default None.
        weight_initialization : str or None, optional
            Weight initialization method, by default None.
        bias_initialization : str or None, optional
            Bias initialization method, by default None.

        Returns
        -------
        None
        """

        super(NeuralNetwork, self).__init__()
        
        self.Nfeatures = Nfeatures
        self.activation = activation
        self.complexity = complexity
        self.lr = lr
        self.loss = loss
        self.layers = layers
        self.generator = torch.Generator().manual_seed(seed) if seed else torch.Generator()
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.weight_initialization = getattr(nn.init, weight_initialization)
        self.bias_initialization = getattr(nn.init, bias_initialization)
        self.use_batch_norm = use_batch_norm
        self.scale_data = scale_data

        if self.use_batch_norm and self.scale_data:
            print('Warning: Are you sure you want to scale the data and use batch normalization?')

        # Add the proper number of layers
        width = 2**(layers-1)*complexity
        self.input = nn.Linear(Nfeatures, width)
        self.input_bn = nn.BatchNorm1d(width) if use_batch_norm else None
        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList() if use_batch_norm else None
        for _ in range(layers-1):
            self.hidden_layers.append(nn.Linear(width, width//2))
            if use_batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(width // 2))
            width //= 2
        self.output = nn.Linear(width, 1)
        
        # Set activation functions
        self.activation_func = getattr(nn, activation)()
        self.output_activation = output_activation
        if self.output_activation is not None:
            self.output_activation_func = getattr(nn, output_activation)()

        # Initialize weights
        self.initialize_weights()

        
    def initialize_weights(self):
        '''Initialize the weights of the model'''
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.weight_initialization == nn.init.zeros_:
                    self.weight_initialization(module.weight)
                else:
                    self.weight_initialization(module.weight, generator=self.generator)
                if self.bias_initialization == nn.init.zeros_:
                    self.bias_initialization(module.bias)
                else:
                    self.bias_initialization(module.bias, generator=self.generator)

                    
    def get_weights(self):
        '''Get the weights of the model'''
        weights = [module.weight for module in self.modules() if isinstance(module, nn.Linear)]
        return weights

    
    def get_biases(self):
        '''Get the biases of the model'''
        biases = [module.bias for module in self.modules() if isinstance(module, nn.Linear)]
        return biases
        

    def forward(self, x):
        '''Forward pass of the model'''

        # Input layer
        x = self.input(x)
        if self.use_batch_norm and x.size(0) > 1:
            x = self.input_bn(x)
        x = self.activation_func(x)

        # Hidden layers
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.use_batch_norm and x.size(0) > 1:
                x = self.bn_layers[i](x)
            x = self.activation_func(x)
        
        # Output layer
        x = self.output(x)
        if self.output_activation is not None:
            x = self.output_activation_func(x)

        return x

        
    def fit(self, Xtrain, ytrain, epochs=1000, batch_size=32, verbose=0, 
            val_split=0.2, reset_weights=False, optimizer='Adam', patience=50):   
        '''Train the model

        Training will default to resetting the weights to their initial values.
        
        Parameters
        ----------
        Xtrain : array_like
            Array containing the datapoint features in the shape (n x m)
            where n is the number of datapoints and m is the number of
            features.
        ytrain : array_like
            Array with labels. Should contain n datapoints.
        epochs : int, optional
            Limit of total epochs to run. Default is 1000.
        batch_size : int, optional
            Batch size to use in training. Default is 16.
        verbose : int, optional
            Output text (0=none, 1=full, 2=minimal). Default is 0.
        val_split : float, optional
            Split to use for validation during training. Default is 0.25.
        reset_weights : bool, optional
            Reset weights to their initial value. Default is True.
        optimizer : str, optional
            Name of the optimizer to use. Default is 'Adam'.
        patience : int, optional
            Number of epochs to wait for improvement in validation loss before stopping training. Default is None.
        '''

        # Place model in train mode
        self.train()

        # Convert data to tensors
        Xtrain = torch.tensor(Xtrain, dtype=torch.float32)
        ytrain = torch.tensor(ytrain, dtype=torch.float32)

        # Initialize model weights
        if reset_weights:
            self.initialize_weights()
        
        # Set up loss function and optimizer
        criterion = getattr(nn, self.loss)()
        optimizer = getattr(optim, optimizer)(self.parameters(), lr=self.lr)
        
        # Split data into training and validation sets, set loaders
        dataset = torch.utils.data.TensorDataset(Xtrain, ytrain)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Fit scaler on train_data and transform both train_data and val_data
        if self.scale_data:
            train_X = torch.stack([x for x, _ in train_data])
            train_y = torch.stack([y for _, y in train_data])
            
            self.scaler_X.fit(train_X)
            self.scaler_y.fit(train_y.unsqueeze(1))
            
            train_data = [(torch.tensor(self.scaler_X.transform(x.unsqueeze(0)), dtype=torch.float32).squeeze(0), 
                    torch.tensor(self.scaler_y.transform(y.reshape(-1,1)), dtype=torch.float32).squeeze(0)) for x, y in train_data]
            val_data = [(torch.tensor(self.scaler_X.transform(x.unsqueeze(0)), dtype=torch.float32).squeeze(0), 
                    torch.tensor(self.scaler_y.transform(y.reshape(-1,1)), dtype=torch.float32).squeeze(0)) for x, y in val_data]

        train_loader = torch.utils.data.DataLoader(
            train_data, 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=True if train_size > batch_size else False,
        )
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True if val_size > batch_size else False,
        )
        
        # Training loop
        if patience:
            no_improvement = 0
            best_loss = float('inf')

        for epoch in range(epochs):
            # Compute loss and gradients
            total_loss = 0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(inputs).squeeze()
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Compute val loss
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = self.forward(inputs).squeeze()
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

            if verbose:
                print(f'Epoch: {epoch+1:4d}/{epochs}, Loss: {total_loss/train_size:.5f}, Val Loss: {val_loss/val_size:.5f}')
                    
            # Check for early stopping
            if patience:
                # Check if val loss has stopped decreasing
                if val_loss >= best_loss:
                    no_improvement += 1
                else:
                    best_loss = val_loss
                    no_improvement = 0
                    # Record the model parameters from the best loss round
                    best_model_state_dict = self.state_dict()

                # Exit the loop if val loss has stopped decreasing for patience iterations
                if no_improvement >= patience:
                    fstring = 'no improvement'
                    break
            fstring = 'max epochs'

        print(f'Trained for {epoch+1} epochs ({fstring})')

        if patience:
            # Reset the model to parameters from best val loss
            self.load_state_dict(best_model_state_dict)

                
    def predict(self, X, verbose=False):
        self.eval()

        # Scale the input data
        if self.scale_data:
            X = self.scaler_X.transform(X)

        # Convert data to tensors
        X = torch.tensor(X, dtype=torch.float32)

        # Make predictions
        with torch.no_grad():
            predictions = self(X).detach().numpy()

        # Inverse transform the predictions
        if self.scale_data:
            predictions = self.scaler_y.inverse_transform(predictions)

        if verbose:
            print(f'Predictions: {predictions}')
        
        return predictions

        
    def __repr__(self):
        model_architecture = f'Input Layer: {self.input.in_features} -> {self.input.out_features}\n'
        for i, layer in enumerate(self.hidden_layers):
            model_architecture += f'Hidden Layer {i+1}: {layer.in_features} -> {layer.out_features}\n'
        model_architecture += f'Output Layer: {self.output.in_features} -> {self.output.out_features}\n'
        
        return (f'NeuralNetwork(\n'
                f'  Number of tunable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)},\n'
                f'  Architecture:\n{model_architecture}'
                f'  Activation: {self.activation},\n'
                f'  Output Activation: {self.output_activation}\n'
                f')')
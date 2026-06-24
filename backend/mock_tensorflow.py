"""
Mock TensorFlow module for testing without actual TensorFlow installation
"""

import numpy as np
import json
import os

class MockModel:
    """Mock TensorFlow model for testing."""
    
    def __init__(self, num_classes=30):
        self.num_classes = num_classes
        self.input_shape = (None, 224, 224, 3)
        self.output_shape = (None, num_classes)
        self._params = 1000000  # Mock parameter count
    
    def predict(self, x, verbose=0):
        """Mock prediction that returns realistic probabilities."""
        batch_size = x.shape[0]
        
        # Generate realistic predictions based on input characteristics
        predictions = []
        for i in range(batch_size):
            img = x[i]
            
            # Analyze image characteristics
            mean_brightness = np.mean(img)
            green_channel = np.mean(img[:, :, 1])
            red_channel = np.mean(img[:, :, 0])
            
            # Create base probabilities
            probs = np.random.exponential(0.05, size=self.num_classes)
            
            # Make some classes more likely based on image characteristics
            if green_channel > 0.4:  # Greenish image - more likely healthy
                healthy_indices = [3, 6, 10, 13, 17]  # Some healthy class indices
                for idx in healthy_indices:
                    if idx < self.num_classes:
                        probs[idx] *= 2.0
            
            if red_channel > green_channel:  # Reddish - more likely diseased
                disease_indices = [0, 1, 2, 4, 5, 8, 9, 11, 12, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24]
                for idx in disease_indices:
                    if idx < self.num_classes:
                        probs[idx] *= 1.5
            
            # Normalize
            probs = probs / np.sum(probs)
            predictions.append(probs)
        
        return np.array(predictions, dtype=np.float32)
    
    def count_params(self):
        return self._params

class MockSequential:
    """Mock Sequential model."""
    
    def __init__(self, layers=None):
        self.layers = layers or []
        self.num_classes = 30
        self.input_shape = (None, 224, 224, 3)
        self.output_shape = (None, self.num_classes)
        self._compiled = False
    
    def compile(self, optimizer=None, loss=None, metrics=None):
        self._compiled = True
    
    def predict(self, x, verbose=0):
        return MockModel(self.num_classes).predict(x, verbose)
    
    def count_params(self):
        return 1000000

class MockModels:
    """Mock keras.models module."""
    
    @staticmethod
    def load_model(filepath, **kwargs):
        """Mock model loading."""
        print(f"Mock loading model from {filepath}")
        
        # Try to determine number of classes from labels
        try:
            backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            labels_path = os.path.join(backend_dir, 'labels.json')
            if os.path.exists(labels_path):
                with open(labels_path, 'r') as f:
                    labels = json.load(f)
                num_classes = len(labels)
            else:
                num_classes = 30
        except:
            num_classes = 30
        
        return MockModel(num_classes)
    
    Sequential = MockSequential

class MockLayers:
    """Mock keras.layers module."""
    
    class Input:
        def __init__(self, shape=None):
            self.shape = shape
    
    class Dense:
        def __init__(self, units, activation=None):
            self.units = units
            self.activation = activation
    
    class GlobalAveragePooling2D:
        def __init__(self):
            pass

class MockKeras:
    """Mock keras module."""
    models = MockModels()
    layers = MockLayers()

class MockTensorFlow:
    """Mock TensorFlow module."""
    keras = MockKeras()

# Create the mock modules
import sys
sys.modules['tensorflow'] = MockTensorFlow()
sys.modules['tensorflow.keras'] = MockKeras()
sys.modules['tensorflow.keras.models'] = MockModels()
sys.modules['tensorflow.keras.layers'] = MockLayers()
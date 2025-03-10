# Cat vs Dog Image Classification (Synthetic Dataset)

This project implements a simple **Convolutional Neural Network (CNN)** to classify images as either cats or dogs. Instead of using real images, it generates **synthetic data** (random pixel values) for training and testing purposes.

## Features
- Uses **TensorFlow/Keras** to build and train a CNN model.
- Generates **synthetic image data** instead of downloading an actual dataset.
- Implements **binary classification** (0 for cat, 1 for dog).
- Trains the model on randomly generated images.
- Evaluates performance on a validation set.

## Technologies Used
- **Python**
- **TensorFlow/Keras** (for deep learning)
- **NumPy** (for data manipulation)
- **Matplotlib** (for visualization)


## Code Overview
### 1. Generate Synthetic Data
The script creates **random images** (with pixel values between 0 and 1) and assigns **random labels** (0 or 1) to simulate a dataset.

```python
num_samples = 2000
img_size = (128, 128, 3)
X_data = np.random.rand(num_samples, *img_size).astype(np.float32)
y_data = np.random.randint(0, 2, num_samples)
```

### 2. Define CNN Model
A simple CNN model with two convolutional layers, max pooling, and dense layers for classification.

```python
model = models.Sequential([
    layers.Conv2D(32, (5, 5), activation='relu', input_shape=img_size),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (5, 5), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(120, activation='relu'),
    layers.Dense(84, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
```

### 3. Training the Model
The model is compiled with Adam optimizer and binary cross-entropy loss, then trained on the synthetic data.

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
```



## Future Improvements
- Replace **synthetic images** with a real dataset like [Kaggle's Dog vs Cat dataset](https://www.kaggle.com/c/dogs-vs-cats/data).
- Use **data augmentation** to improve model performance.
- Experiment with **deeper CNN architectures**.



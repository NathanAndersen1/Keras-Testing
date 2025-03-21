import numpy as np
import os
import matplotlib.pyplot as plt  # For live graph
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense

# Define model file path
MODEL_FILE = "trained_model.h5"

# Function to check if the model file is valid
def is_valid_model_file(filepath):
    try:
        load_model(filepath)
        return True
    except:
        return False

# Initialize lists to store loss values
loss_values = []

if os.path.exists(MODEL_FILE) and is_valid_model_file(MODEL_FILE):
    print("Loading existing model...")
    model = load_model(MODEL_FILE)
else:
    print("Training a new model...")
    # Generate dataset
    x = np.random.randint(0, 100, size=(100, 1))
    y = x * 2

    # Build the model
    model = Sequential([
        Dense(16, input_dim=1, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    history = model.fit(x, y, epochs=50, batch_size=32, verbose=1)
    loss_values.extend(history.history['loss'])

    # Save the trained model
    model.save(MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

# Set up live graph
plt.ion()
fig, ax = plt.subplots()
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
line, = ax.plot([], [], label="Training Loss")
ax.legend()

print("Starting automatic training and predictions...")
while True:
    # Generate new training data
    x = np.random.randint(0, 100, size=(1000, 1))
    y = x * 2
    # Train the model on new data
    history = model.fit(x, y, epochs=10, batch_size=32, verbose=1)

    # Update loss values
    loss_values.extend(history.history['loss'])

    # Update the live plot
    line.set_xdata(range(1, len(loss_values) + 1))
    line.set_ydata(loss_values)
    ax.relim()
    ax.autoscale_view()
    plt.draw()

    # Save the graph to a PNG file
    plt.savefig("training_progress.png")
    print("Graph saved as 'training_progress.png'")

    # Perform predictions
    for _ in range(200):
        num = np.random.randint(0, 100)
        prediction = model.predict(np.array([[num]]))
        rounded_prediction = int(round(prediction[0][0]))
        print(f"Input: {num}, Predicted Output: {rounded_prediction}")

    # Check if the user wants to exit
    user_input = input("Type 'exit' to stop or press Enter to continue: ")
    if user_input.lower() == 'exit':
        print("Exiting...")
        break

# Save the trained model before exiting
model.save(MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")
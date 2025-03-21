import numpy as np
import os
import matplotlib.pyplot as plt  # For live graph
from tensorflow import Sequential, load_model
from tensorflow import Dense

# Define model file path and vars
MODEL_FILE = "trained_model.h5"
success_rates = []
iterations = []

# Function to check if the model file is valid
def is_valid_model_file(filepath):
    try:
        load_model(filepath)
        return True
    except:
        return False

def create_model(board_size):
    if os.path.exists(MODEL_FILE) and is_valid_model_file(MODEL_FILE):
        print("Loading existing model...")
        model = load_model(MODEL_FILE)
    else:
        print("Training a new model...")
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

        # Save the trained model
        model.save(MODEL_FILE)
        print(f"Model saved to {MODEL_FILE}")

# Set up live graph
plt.ion()
fig, ax = plt.subplots()
ax.set_xlabel("Iteration")
ax.set_ylabel("Success Rate (%)")
line, = ax.plot([], [], label="Success Rate")
ax.legend()

# Continuous training and prediction loop
print("Starting automatic training and predictions...")
iteration_count = 0
while True:
    iteration_count += 1

    # Generate new training data
    x = np.random.randint(0, 100, size=(1000, 1))
    y = x * 2

    # Train the model on new data
    history = model.fit(x, y, epochs=10, batch_size=32, verbose=1)

    correct_predictions = 0
    for _ in range(50):
        num = np.random.randint(0, 100)
        prediction = model.predict(np.array([[num]]))
        rounded_prediction = int(round(prediction[0][0]))
        if rounded_prediction == num * 2:
            correct_predictions += 1
    success_rate = (correct_predictions / 50) * 100
    success_rates.append(success_rate)
    iterations.append(iteration_count)

    # Update the live plot
    line.set_xdata(iterations)
    line.set_ydata(success_rates)
    ax.relim()
    ax.autoscale_view() 
    plt.draw()

    # Save the graph to a PNG file
    plt.savefig("success_rate_progress.png")
    print(f"Graph saved as 'success_rate_progress.png' - Success Rate: {success_rate:.2f}%")

    # Check if the user wants to exit
    user_input = input("Type 'exit' to stop or press Enter to continue: ")
    if user_input.lower() == 'exit':
        print("Exiting...")
        break

# Save the trained model before exiting
model.save(MODEL_FILE)
print(f"Model saved to {MODEL_FILE}")
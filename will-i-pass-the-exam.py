import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Generate synthetic data
np.random.seed(42)
num_samples = 1000

hours_studied = np.random.normal(5, 2, num_samples)
attendance_rate = np.random.normal(0.8, 0.1, num_samples)
homework_score = np.random.normal(75, 10, num_samples)
midterm_score = np.random.normal(70, 15, num_samples)

# Step 2: Combine features
X = np.column_stack([hours_studied, attendance_rate, homework_score, midterm_score])

# Step 3: Define the target variable (Pass = 1, Fail = 0)
# Simple rule: if all three are above average, likely to pass
y = ((hours_studied > 4.5) & 
     (attendance_rate > 0.75) & 
     (homework_score > 70) & 
     (midterm_score > 65)).astype(int)

# Step 4: Train/test split and normalize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # sigmoid for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 6: Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=0)

# Step 7: Evaluate
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Step 8: Predict sample student
sample_student = np.array([[6, 0.85, 80, 75]])  # hours, attendance, homework, midterm
sample_scaled = scaler.transform(sample_student)
probability = model.predict(sample_scaled)[0][0]
print("Will pass?" , "Yes" if probability > 0.5 else "No", f"(Confidence: {probability:.2f})")


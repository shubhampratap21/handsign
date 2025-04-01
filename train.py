import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# Configuration
INPUT_SHAPE = (63,)  # 21 landmarks * 3 coordinates (x,y,z)
NUM_CLASSES = 26      # A-Z classes
BATCH_SIZE = 256
EPOCHS = 50

# Load preprocessed data
def load_data(data_dir):
    X_train = np.load(os.path.join(data_dir, "train_X.npy"))
    y_train = np.load(os.path.join(data_dir, "train_y.npy"))
    X_test = np.load(os.path.join(data_dir, "test_X.npy"))
    y_test = np.load(os.path.join(data_dir, "test_y.npy"))
    
    # Convert labels to categorical
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)
    
    return (X_train, y_train), (X_test, y_test)

# Build model
def create_model():
    model = Sequential([
        Dense(512, activation='relu', input_shape=INPUT_SHAPE),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)  # <-- THIS IS THE CRUCIAL FIX
    
    # Load data
    (X_train, y_train), (X_test, y_test) = load_data("preprocessed_data")
    
    # Create model
    model = create_model()
    model.summary()
    
    # Callbacks
    callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("models/best_model.keras", save_best_only=True)  # Changed to .keras
    ]   
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )
    
    # Save final model
    model.save("models/final_model.keras")
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_acc*100:.2f}%")

if __name__ == "__main__":
    # Suppress TensorFlow info messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    
    main()
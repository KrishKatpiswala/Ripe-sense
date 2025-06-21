# Mango Quality Grading System - Fixed Implementation

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import os
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing import image
import io
import base64
from PIL import Image

# ============================================================================
# STEP 1: MODEL TRAINING FUNCTIONS
# ============================================================================

def create_data_generators():
    """Create ImageDataGenerator for data augmentation and preprocessing"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    return train_datagen, test_datagen

def load_datasets(train_datagen, data_dir):
    """Load training and validation datasets"""
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_generator, validation_generator

def build_cnn_model():
    """Build CNN model for mango quality classification"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')  # 3 classes: Over_Ripe, Perfect_Ripe, Under_Ripe
    ])
    
    return model

def train_mango_model():
    """Complete training pipeline"""
    data_dir = 'Grading_dataset'
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found!")
        print("Please create the directory structure as described in the setup instructions.")
        return None, None
    
    print("Starting model training...")
    
    # Create data generators
    train_datagen, _ = create_data_generators()
    
    # Load datasets
    train_generator, validation_generator = load_datasets(train_datagen, data_dir)
    
    # Build model
    model = build_cnn_model()
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model architecture:")
    model.summary()
    
    # Train model
    print("Training model...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=25,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        verbose=1
    )
    
    # Save model
    model.save('model.h5')
    print("Model saved as 'model.h5'")
    
    return model, history

# ============================================================================
# STEP 2: FLASK APPLICATION
# ============================================================================

app = Flask(__name__)

# Global model variable
model = None
CLASS_LABELS = ['Over_Ripe', 'Perfect_Ripe', 'Under_Ripe']

def load_trained_model():
    """Load the pre-trained model"""
    global model
    try:
        if os.path.exists('model.h5'):
            model = load_model('model.h5')
            print("Model loaded successfully!")
            return True
        else:
            print("Model file 'model.h5' not found!")
            print("Please train the model first by running: python train_model.py")
            return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def preprocess_image(img):
    """Preprocess image for prediction"""
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict_mango_quality(img):
    """Predict mango quality from image"""
    if model is None:
        return None, None, None
    
    # Preprocess image
    processed_img = preprocess_image(img)
    
    # Make prediction
    predictions = model.predict(processed_img)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = CLASS_LABELS[predicted_class_index]
    confidence = float(predictions[0][predicted_class_index])
    
    # Get all class probabilities
    all_predictions = {
        CLASS_LABELS[i]: float(predictions[0][i]) 
        for i in range(len(CLASS_LABELS))
    }
    
    return predicted_class, confidence, all_predictions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        # Read and process image
        img = Image.open(file.stream)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Make prediction
        predicted_class, confidence, all_predictions = predict_mango_quality(img)
        
        if predicted_class is None:
            return jsonify({
                'error': 'Model not loaded. Please train the model first.',
                'solution': 'Run the training script to create model.h5'
            })
        
        # Convert image to base64 for display
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        return render_template('result.html', 
                             prediction=predicted_class,
                             confidence=f"{confidence:.2%}",
                             all_predictions=all_predictions,
                             image_data=img_base64)
    
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'})

@app.route('/train', methods=['GET', 'POST'])
def train_model_route():
    """Route to train the model"""
    if request.method == 'GET':
        return render_template('train.html')
    
    try:
        model_obj, history = train_mango_model()
        if model_obj is not None:
            # Reload the model in the Flask app
            load_trained_model()
            return jsonify({
                'success': True,
                'message': 'Model trained successfully!',
                'redirect': '/'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Training failed. Check data directory structure.'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Training error: {str(e)}'
        })

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("Mango Quality Grading System")
    print("=" * 40)
    
    # Check if model exists
    if os.path.exists('model.h5'):
        print("Found existing model, loading...")
        success = load_trained_model()
        if success:
            print("✅ Model loaded successfully!")
            print("🌐 Starting Flask application...")
            print("📝 Access the application at: http://localhost:5000")
        else:
            print("❌ Failed to load model")
    else:
        print("⚠️  No trained model found!")
        print("📋 You have two options:")
        print("1. Train model via web interface: http://localhost:5000/train")
        print("2. Train model programmatically by calling train_mango_model()")
        print("\n🌐 Starting Flask application anyway...")
    
    print("\n" + "=" * 40)
    app.run(debug=True, port=5000)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def plot_training_history(history):
    """Plot training history"""
    if history is None:
        print("No training history available")
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# STANDALONE TRAINING SCRIPT
# ============================================================================

def main_training():
    """Main function for standalone training"""
    print("Starting Mango Quality Model Training...")
    model_obj, history = train_mango_model()
    
    if model_obj is not None:
        print("✅ Training completed successfully!")
        
        # Plot training history
        plot_training_history(history)
        
        print("Model saved as 'model.h5'")
        print("You can now run the Flask application!")
    else:
        print("❌ Training failed!")

# Uncomment the line below to train the model when running this script directly
# main_training()
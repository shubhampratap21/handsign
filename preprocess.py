import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import sys

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

def extract_landmarks(image_path):
    """Extract hand landmarks from an image"""
    image = cv2.imread(image_path)
    if image is None:
        return None
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(image)
    
    if results.multi_hand_landmarks:
        landmarks = []
        for landmark in results.multi_hand_landmarks[0].landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks)
    return None

def process_dataset(dataset_path, output_file_prefix):
    """Process dataset and save numpy arrays"""
    # Verify dataset path exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path {dataset_path} does not exist!")
        sys.exit(1)

    # Get valid classes (A-Z)
    valid_classes = [chr(i) for i in range(65, 91)]  # A-Z
    classes = [d for d in os.listdir(dataset_path) 
              if os.path.isdir(os.path.join(dataset_path, d))
              and d.upper() in valid_classes]
    
    print(f"Found {len(classes)} valid classes in dataset")

    all_landmarks = []
    labels = []
    label_map = {cls: idx for idx, cls in enumerate(classes)}

    for cls in classes:
        class_path = os.path.join(dataset_path, cls)
        print(f"\nProcessing {cls}: {class_path}")
        
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in tqdm(image_files, desc=f"Class {cls}"):
            img_path = os.path.join(class_path, img_file)
            landmarks = extract_landmarks(img_path)
            
            if landmarks is not None:
                all_landmarks.append(landmarks)
                labels.append(label_map[cls])

    # Save processed data
    np.save(f"{output_file_prefix}_X.npy", np.array(all_landmarks))
    np.save(f"{output_file_prefix}_y.npy", np.array(labels))
    print(f"\nSaved preprocessed data with {len(all_landmarks)} samples")

if __name__ == "__main__":
    # Your specific paths
    train_path = r"D:\Projects\handsign\asl_alphabet_train\asl_alphabet_train"
    test_path = r"D:\Projects\handsign\asl_alphabet_test\asl_alphabet_test"
    
    # Create output directory
    output_dir = "preprocessed_data"
    os.makedirs(output_dir, exist_ok=True)

    # Process datasets
    print("Processing training data...")
    process_dataset(train_path, os.path.join(output_dir, "train"))
    
    print("\nProcessing test data...")
    process_dataset(test_path, os.path.join(output_dir, "test"))
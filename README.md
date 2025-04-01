# Hand Sign Recognition

This project recognizes American Sign Language (ASL) hand signs using a deep learning model trained with TensorFlow.

## Dataset
The dataset used for training the model can be downloaded from the following link:
[ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

## Installation
To run this project, install the required dependencies using the following command:

```sh
pip install -r requirements.txt
```

### Requirements
Ensure you have the following packages installed:

```
streamlit==1.32.2
tensorflow==2.16.1
mediapipe==0.10.11
opencv-python==4.9.0.80
numpy==1.26.4
Pillow==10.3.0
```

## Running the Application
To launch the Streamlit application, use:

```sh
streamlit run app.py
```

## Project Structure
```
hand_sign/
│── asl_alphabet_test/     # Testing dataset (ignored in Git)
│── asl_alphabet_train/    # Training dataset (ignored in Git)
│── models/                # Saved models (ignored in Git)
│── preprocessed_data/     # Processed dataset (ignored in Git)
│── app.py                 # Streamlit application
│── preprocess.py          # Data preprocessing script
│── train.py               # Model training script
│── requirements.txt       # Dependencies
│── README.md              # Project documentation
```

## Notes
- The dataset is too large to be included in the repository. Please download it from Kaggle before training.
- If you face any issues, ensure that all dependencies are correctly installed and that you are using a compatible Python version.

## License
This project is for educational purposes and follows the dataset license.

---
**Author:** Shubham

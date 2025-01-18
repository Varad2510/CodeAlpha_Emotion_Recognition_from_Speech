# CodeAlpha_Emotion_Recognition_from_Speech
## **Overview**

**CodeAlpha: Emotion Recognition from Speech** is a machine learning-based project that analyzes speech signals to classify emotions. Using audio features like Mel Frequency Cepstral Coefficients (MFCCs), the project leverages deep learning techniques to predict emotions such as happiness, sadness, anger, surprise, fear, and others.

The project aims to build an emotion recognition system that can interpret emotions from human speech and could be applied in various fields such as healthcare, customer service, and human-computer interaction.

## **Features**

- **Emotion Classification:** Classifies speech signals into different emotion categories.
- **Deep Learning Model:** Uses an LSTM (Long Short-Term Memory) model to classify emotions based on features extracted from speech.
- **Real-time Speech Analysis:** Can be extended for real-time emotion detection from audio streams.
- **Data Preprocessing:** Implements feature extraction techniques to handle and preprocess audio data efficiently.

## **Installation**

To set up and run the project, you need to install the required dependencies and libraries.

### **1. Clone the Repository**

```bash
git clone https://github.com/your_username/CodeAlpha_Emotion_Recognition_from_Speech.git
cd CodeAlpha_Emotion_Recognition_from_Speech
```

### **2. Install Python Dependencies**

Create a virtual environment and install the necessary packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

Alternatively, you can install each dependency individually:

```bash
pip install librosa numpy pandas scikit-learn tensorflow
```

## **Usage**

### **1. Dataset**

Ensure you have the **Speech Emotion Dataset** available in your local environment or Google Drive.

- You can download the dataset from [the official source](https://www.kaggle.com/datasets/urbanglass/speech-emotion-recognition).

Once downloaded, extract it and place it in the appropriate folder as per the `DATASET_PATH` in the script.

### **2. Running the Model**

To train the model, run the `emotion_recognition.py` script.

```bash
python emotion_recognition.py
```

This will:
- Load the dataset from the specified path.
- Extract features (MFCCs) from the audio files.
- Preprocess the data and split it into training and test sets.
- Build a deep learning model using an LSTM network.
- Train the model on the training data and evaluate it on the test data.
- Save the trained model for future use.

### **3. Model Evaluation**

Once the model is trained, it will provide the following outputs:
- **Training and Validation Accuracy:** Provides insights into how well the model is performing during training.
- **Classification Report:** Shows the precision, recall, f1-score, and support for each emotion class.
- **Saved Model:** The trained model is saved in the specified path (`/content/drive/My Drive/SpeechEmotionModel.h5`) for future use.

## **Model Architecture**

- **Input Layer:** Takes in extracted features from audio files (MFCCs).
- **Hidden Layers:**
  - Dense layers with ReLU activation to learn complex patterns in the data.
  - Dropout layers to prevent overfitting.
- **Output Layer:** A softmax activation layer that classifies the input audio into different emotion categories.

The model is compiled using the Adam optimizer and categorical cross-entropy loss function for multi-class classification.

## **Results**

The model's classification performance is evaluated on the test dataset using several metrics:

- **Accuracy:** Percentage of correct predictions made by the model.
- **Precision, Recall, F1-score:** Measures for each emotion class to assess the model's robustness and generalization.
- **Confusion Matrix:** Shows how well the model distinguishes between different emotions.

## **File Structure**

```
CodeAlpha_Emotion_Recognition_from_Speech/
│
├── emotion_recognition.py           # Main script for training the emotion recognition model
├── requirements.txt                # List of required Python packages
├── SpeechEmotionDataset/           # Folder containing the extracted dataset
│   ├── Actor_01/                   # Sub-folder for each actor
│   │   ├── file1.wav               # Audio file of speech with emotion
│   ├── Actor_02/                   # More folders for other actors
├── README.md                       # Project documentation
```

---

## **Future Work**

- **Real-Time Emotion Detection:** The model can be extended to detect emotions from live speech or audio streams in real-time.
- **Multilingual Support:** Extend the model to work with multilingual datasets for broader applicability.
- **Improved Feature Extraction:** Experiment with different feature extraction techniques like spectral features or deep learning-based audio embeddings to improve accuracy.

## **Acknowledgments**

- **Speech Emotion Dataset:** Used for training the emotion recognition model.
- **Librosa:** For audio processing and feature extraction.
- **TensorFlow and Keras:** For building and training the deep learning model.
- **Scikit-learn:** For preprocessing and evaluation tasks like label encoding and metrics.

## **Contributing**

Contributions to this project are welcome. If you would like to improve the code or add new features, feel free to fork this repository and submit a pull request. 

Please make sure to follow the standard coding practices and provide clear commit messages.

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

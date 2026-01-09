# CARREFOUR ChatBot Models (BoW and Embedding)

## About
This notebook demonstrates the development of two chatbot models for CARREFOUR: a Bag of Words (BoW) model and an Embedding-based model. Both models are built using PyTorch and designed to classify user intents and provide appropriate responses.

### Group Members
- Collins Ndung'u
- Nancy Daniel
- Baptiste Billy Nitunga
- Gideon Mutuku

View our poster [here](https://www.overleaf.com/read/cbzmjmbfhvzq#f10dff).

## Table of Contents
1. [Import Libraries](#import-libraries)
2. [Data Loading and Preprocessing](#data-loading-and-preprocessing)
3. [Model Definitions](#model-definitions)
4. [Model Training](#model-training)
5. [Model Evaluation](#model-evaluation)
6. [Prediction and Response (Chatbot Usage)](#prediction-and-response)

## 1. Import Libraries
The notebook imports essential libraries for natural language processing (NLP), deep learning, and data handling:
- `os`, `json`, `random`
- `nltk` (for tokenization and lemmatization)
- `numpy`, `pandas`
- `torch`, `torch.nn`, `torch.optim`, `torch.utils.data` (for building and training neural networks)
- `requests` (for downloading the dataset)

## 2. Data Loading and Preprocessing
- **Dataset**: The chatbot's knowledge base is loaded from an `intents.json` file, which contains various patterns and their corresponding tags (intents) and responses.
- **Download**: The `intents.json` file is downloaded from Google Drive using a helper function.
- **Preprocessing Steps**:
    - **Tokenization**: Breaking down sentences into individual words.
    - **Lemmatization**: Reducing words to their base form (e.g., "running" -> "run").
    - **Vocabulary Creation**: Building a list of unique words and classes (intents).
    - **Bag of Words (BoW) Representation**: Converting text patterns into numerical vectors where each dimension corresponds to a word in the vocabulary, indicating its presence or absence.
    - **Embedding Representation**: Converting words into numerical indices, padding sequences to a fixed length, and preparing data for an embedding layer.
    - **Data Splitting**: The processed data is split into training (90%) and testing (10%) sets for both BoW and Embedding formats.

## 3. Model Definitions
Two neural network models are defined:

### Bag of Words (BoW) Model (`ChatbotModel`)
- A simple feed-forward neural network with three linear layers and ReLU activations, with dropout for regularization.
- Input size: number of unique words in the vocabulary.
- Output size: number of unique intent classes.

### Embedding Model (`EmbeddingChatbotModel`)
- Utilizes an `nn.Embedding` layer to convert word indices into dense vectors.
- Followed by an `nn.LSTM` (Long Short-Term Memory) layer to capture sequential dependencies in the input.
- A final linear layer maps the LSTM output to the intent classes.
- Input size: `max_sequence_length`.
- Output size: number of unique intent classes.

## 4. Model Training
- Both models are trained separately using PyTorch's `nn.CrossEntropyLoss` as the loss function and `optim.Adam` as the optimizer.
- Training parameters:
    - `epochs`: 200
    - `batch_size`: 8
    - `learning_rate`: 0.001
- Training progress (loss) is printed every 10 epochs.
- Trained models are saved as `chatbot_model_bow.pth` and `chatbot_model_emb.pth`.

## 5. Model Evaluation
- Models are evaluated on their respective test sets.
- Accuracy is calculated as the ratio of correct predictions to total samples.
- The notebook outputs the accuracy for both the BoW and Embedding models.

## 6. Prediction and Response (Chatbot Usage)
- Functions are provided to preprocess user input (`clean_up_sentence`, `bag_of_words`, `sentence_to_embedding_sequence`).
- `predict_class` and `predict_class_embedding` functions use the trained models to predict the intent of a user's query.
- A confidence threshold of 0.7 is applied to filter predictions.
- `get_response` retrieves a random response from the `intents.json` file based on the predicted intent.
- **Interactive Chat**: The notebook includes a loop allowing users to interact with either the BoW or Embedding model. Users can type queries, and the chatbot will respond based on its predictions. Type 'quit' to exit the chat.

**To run the chatbot interactively:**
1. Ensure all cells up to the "Main execution" section have been run successfully.
2. Execute the "Use the chatbot" cell.
3. Choose between the 'BoW' or 'Embedding' model when prompted.
4. Type your questions, and the chatbot will respond. Type 'quit' to end the session.
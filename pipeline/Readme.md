
# Personal Assistant with LLM

This project implements a personal assistant using Large Language Models (LLMs). The assistant processes user inputs, extracts information from various sources like PDFs and HTML pages, and provides responses based on the user's profile and preferences.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Configuration](#configuration)
- [Function Descriptions](#function-descriptions)
- [Testing](#testing)


## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/personal-assistant-llm.git
    cd personal-assistant-llm
    ```

2. **Create a virtual environment and activate it:**
    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the main application:**
    ```bash
    python src/main.py
    ```

2. **Interact with the assistant:**
    - Enter your username.
    - Provide path to PDF files or URLs for the assistant to process.
    - Ask questions or provide inputs for the assistant to handle.

## Features

- **Text Extraction from PDFs and HTML**: Extracts text, images, and tables from PDFs and cleans HTML content.
- **Profile Processing**: Collects and processes text data to generate a comprehensive user profile.
- **Intent Classification**: Classifies user queries into various intents like save, question, update, delete, suggestion, and emotion.
- **Follow-up Handling**: Manages follow-up interactions based on the initial query and context.
- **Zero-shot Classification**: Uses pre-trained models to classify texts and determine follow-up actions without needing additional training.

## Directory Structure

```
personal-assistant-llm/
├── src/
│   ├── config.py
│   ├── handlers.py
│   ├── HTMLExtraction.py
│   ├── LLMManager.py
│   ├── main.py
│   ├── PDFExtraction.py
│   ├── ProfileProcessor.py
│   ├── TextProcessor.py
│   └── utils.py
├── tests/
│   ├── test_html_extraction.py
│   ├── test_pdf_extraction.py
│   ├── test_profile_processor.py
│   ├── test_utils.py
│   └── conftest.py
├── requirements.txt
└── README.md
```

## Configuration

- **Configuration Variables**:
  - `DEVICE`: Device to run the model on (e.g., "cpu", "mps").
  - `CLASSIFIER_MODEL`: Model used for classification (e.g., "facebook/bart-large-mnli").
  - `SUMMARY_DB`: Path to the summary database.

## Function Descriptions

### `src/config.py`
- **Configuration Variables**: Set up global configuration variables like `DEVICE`, `CLASSIFIER_MODEL`, and `SUMMARY_DB`.

### `src/handlers.py`
- **handle_follow_ups(llm_manager, intent, context, initial_query, classifier)**: Manages follow-up interactions based on the user’s initial query and context.
- **classify_intent(query, classifier)**: Classifies the intent of a given query using a zero-shot classification model.

### `src/HTMLExtraction.py`
- **HTMLExtraction**: Class to handle the extraction of text, images, and cleaning of HTML content.
  - **fetch_and_save_html()**: Fetches and saves HTML content from a given URL.
  - **clean_html(html_content)**: Cleans HTML content by removing unnecessary tags.
  - **extract_text_from_html(html_content)**: Extracts plain text from HTML content.

### `src/LLMManager.py`
- **LLMManager**: Manages interactions with the LLM to process profiles and queries.
  - **process_profile(prompt)**: Queries the LLM with a prompt to extract profile information.
  - **ask_question(context, query)**: Asks a question based on the provided context and query.
  - **ask_suggestion(context, query)**: Provides a suggestion based on the provided context and query.

### `src/PDFExtraction.py`
- **PDFExtraction**: Handles extraction of text, images, and tables from PDF files.
  - **extract_text()**: Extracts text from the PDF and saves it to a text file.
  - **extract_images()**: Extracts images from the PDF and saves them to the images directory.
  - **extract_tables()**: Extracts tables from the PDF and saves them as CSV files.

### `src/ProfileProcessor.py`
- **ProfileProcessor**: Processes user profiles by collecting text data, generating a profile summary, and writing the summary to a file.
  - **collect_texts()**: Collects text from all `.txt` files in the specified directory and its subdirectories.
  - **process_profile()**: Processes the collected text to generate a profile summary using the LLMManager.
  - **write_profile_to_file(profile_dict)**: Writes the generated profile summary to a text file.

### `src/TextProcessor.py`
- **TextProcessor**: Processes text files, encodes the texts using SentenceTransformers, and saves the results to CSV files.
  - **extract_text(file_path)**: Extracts text content from a given file path.
  - **encode_texts(texts)**: Encodes a list of texts using the SentenceTransformer model.
  - **save_to_csv(file_path, texts, embeddings)**: Saves the texts and their embeddings to a CSV file.
  - **process_and_save_text(master_text, output_csv)**: Processes and saves the text chunks and their embeddings to a CSV file.
  - **process_directory(directory)**: Processes all text files in a directory, combines the texts, and saves the results to a CSV file.
  - **process_profile(directory)**: Processes summary text files in a directory and saves the results to a CSV file.

### `src/utils.py`
- **classify_intent(query, classifier)**: Classifies the intent of a given query using a zero-shot classification model.
- **is_follow_up(text, classifier)**: Determines if a given text requires a follow-up action.
- **match_exit_phrases(follow_up_query)**: Checks if the follow-up query contains any exit phrases.
- **get_valid_input(prompt, valid_options, max_retries=3)**: Prompts the user for input and validates against a list of valid options.
- **get_user_inputs(prompt)**: Prompts the user for multiple inputs until an empty input is received.

## Testing

1. **Run the tests:**
    ```bash
    pytest
    ```

2. **Test Coverage**:
    - Tests cover functionality such as text extraction, profile processing, intent classification, and follow-up handling.
    - Mocking is used to simulate external dependencies and ensure isolated testing of functions.


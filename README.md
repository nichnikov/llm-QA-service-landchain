# LLM-Based-QA-Service

## Overview
This project implements a question-answering service using language models. It integrates various components to process user queries and return accurate answers.

## Project Structure
```
LLM-Based-QA-Service/
├── chains/                   # Contains the logic for classifiers and QA chains
│   ├── classifier.py
│   └── qa_chain.py
├── configs/                  # Configuration files for prompts
│   └── prompts.json
├── core/                     # Core functionalities and data types
│   ├── callbacks.py
│   ├── llm_clients.py
│   └── data_types.py
├── data/                     # Data storage for memory
│   └── memory/
├── services/                 # Service components like retrievers
│   └── retriever.py
├── tests/                    # Test cases for the application
│   └── ...
├── .env                      # Environment variables
├── main.py                   # Main entry point for the application
└── requirements.txt          # Project dependencies
```

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```
   cd LLM-Based-QA-Service
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Set up environment variables in the `.env` file as needed.

## Usage
To run the application, execute:
```
python main.py
```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for discussion.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
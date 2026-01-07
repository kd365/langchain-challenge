# LangChain Chatbot Challenge

![Test](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME/actions/workflows/test.yml/badge.svg)
![Lint](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME/actions/workflows/lint.yml/badge.svg)

A multilingual chatbot and text summarizer built with LangChain and AWS Bedrock.

## Features

- **Assistant Mode**: Multilingual chatbot that responds in your chosen language
- **Summarizer Mode**: Summarize text with brief or detailed output

## Setup

1. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your AWS profile:
   ```
   AWS_PROFILE=class
   ```

## Usage

Run the interactive chatbot:
```bash
python langchain_chatbot_lab.py
```

## Testing

Run tests (no AWS credentials required):
```bash
pytest tests/ -v
```

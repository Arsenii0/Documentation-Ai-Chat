# Documentation chatbot

* pip3 install -r requirements.txt

* Get your OpenAI and Pinecone accounts
  - [OpenAI API Key](https://platform.openai.com)
  - [Pinecone](app.pinecone.io)
  When you are creating the pinecone index make sure,
  - **index_name = "langchain-chatbot"**
  - **Dimensions of the index is 384**

* Export environment variables
```sh
  export OPENAI_API_KEY="your_openai_api_key_here"
  export PINECONE_API_KEY="your_pinecone_api_key_here"
  export PINECONE_ENVIRONMENT="your_pinecone_environment_here"
  export PINECONE_INDEX="your_pinecone_index"
```

* Index the data and upload it to pinecone
```sh
   python3 indexing.py
```

* To run the app
```sh
   streamlit run main.py
```

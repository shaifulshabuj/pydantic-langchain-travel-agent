# Step-1:
## To install the python requirement
```
python -m venv travel_env
source travel_env/bin/activate
pip install ollama pydantic streamlit python-dotenv
```

Install Ollama locally and pull a model:
```
ollama pull llama3.1
```

Or use the hosted Ollama endpoint with an API key.

# Step-2:
## Set Ollama variables in .env
```
OLLAMA_HOST=https://ollama.com/api
OLLAMA_API_KEY=
OLLAMA_MODEL=gpt-oss:20b
```

Note: if OLLAMA_HOST ends with /api, the app normalizes it automatically.

# Step-3:
## To start the travel agent:
```
streamlit run travel_planner.py
```


# Step-1:
## To install the python requirement
```
python -m venv travel_env
source travel_env/bin/activate
pip install langchain langchain-openai pydantic streamlit python-dotenv langchain-core langchain-community
```

# Step-2:
## Set OPENAI_API_KEY in .env
```
OPENAI_API_KEY=
```

# Step-3:
## To start the travel agent:
```
streamlit run travel_planner.py
```

# Simulation
![](https://github.com/shaifulshabuj/pydantic-langchain-travel-agent/blob/main/travel_planner.mov)

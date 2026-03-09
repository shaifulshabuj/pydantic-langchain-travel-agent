import streamlit as st
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from ollama import Client
import json
import logging
import os
from dotenv import load_dotenv

# Loading env variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------
# 1. Data Models with Pydantic
# --------------------------
class Destination(BaseModel):
    name: str
    country: str
    description: str = Field(..., description="Brief overview of the destination")
    best_time_to_visit: str
    attractions: List[str] = Field(default_factory=list)

class Accommodation(BaseModel):
    name: str
    type: str
    price_range: str
    rating: Optional[float] = None
    amenities: List[str] = Field(default_factory=list)

class Transportation(BaseModel):
    type: str
    departure: str
    arrival: str
    duration: str
    price: float

class ItineraryDay(BaseModel):
    day: int
    activities: List[str]
    meals: List[str]
    accommodations: Accommodation

class TravelPlan(BaseModel):
    destination: Destination
    duration: int
    budget: str
    itinerary: List[ItineraryDay]
    transportation: List[Transportation]
    total_estimated_cost: float

# --------------------------
# 2. Agent Definitions
# --------------------------
class BaseAgent:
    def __init__(self, model: Optional[str] = None):
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1")
        self.host = self._normalize_ollama_host(os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        self.api_key = os.getenv("OLLAMA_API_KEY", "").strip()
        self.client = self._create_client("bearer")
        self.tools = self._define_tools()
        self.system_prompt = self._system_prompt()

    def _normalize_ollama_host(self, host: str) -> str:
        normalized = host.rstrip("/")
        if normalized.endswith("/api"):
            normalized = normalized[:-4]
        return normalized

    def _create_client(self, auth_mode: str = "bearer") -> Client:
        headers: Dict[str, str] = {}
        if self.api_key:
            if auth_mode == "bearer":
                headers["Authorization"] = f"Bearer {self.api_key}"
            elif auth_mode == "raw":
                headers["Authorization"] = self.api_key
            elif auth_mode == "x-api-key":
                headers["X-API-Key"] = self.api_key
        return Client(host=self.host, headers=headers if headers else None)

    def _define_tools(self) -> List[Any]:
        raise NotImplementedError

    def _system_prompt(self) -> str:
        raise NotImplementedError

    def run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = None
            last_error: Optional[Exception] = None

            # Hosted providers can vary auth header expectations.
            auth_modes = ["bearer", "raw", "x-api-key"] if self.api_key else ["bearer"]
            for auth_mode in auth_modes:
                try:
                    if auth_mode != "bearer":
                        self.client = self._create_client(auth_mode)

                    response = self.client.chat(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": json.dumps(input, ensure_ascii=True)}
                        ]
                    )
                    break
                except Exception as e:
                    last_error = e
                    message = str(e).lower()
                    if "401" in message or "unauthorized" in message:
                        continue
                    raise

            if response is None:
                raise last_error if last_error else RuntimeError("Ollama request failed")

            output_text = response.get("message", {}).get("content", "")

            parsed = {"output": output_text, "raw": response}
            logger.info(f"Agent response: {parsed}")
            return parsed
        except Exception as e:
            logger.error(f"Error in agent execution: {str(e)}")
            raise

class DestinationResearchAgent(BaseAgent):
    def _define_tools(self) -> List[Any]:
        return []

    def _system_prompt(self) -> str:
        return (
            "You are a travel destination expert. Help users find the perfect destination based on "
            "their interests, budget level, trip duration, and departure location. "
            "Return a detailed response about why the destination matches their criteria."
        )

class TravelOrchestrator:
    def __init__(self):
        self.destination_agent = DestinationResearchAgent()

    def plan_trip(self, user_request: Dict[str, Any]) -> TravelPlan:
        try:
            destination_response = self.destination_agent.run(user_request)

            # Handle both direct dict response and output string
            if isinstance(destination_response, dict):
                if 'output' in destination_response:
                    try:
                        dest_data = json.loads(destination_response['output'])
                    except json.JSONDecodeError:
                        dest_data = [{
                            "name": "Kyoto",
                            "country": "Japan",
                            "description": destination_response['output'],
                            "best_time_to_visit": "Spring (March-May)",
                            "attractions": ["Fushimi Inari Shrine", "Kinkaku-ji"]
                        }]
                else:
                    dest_data = [destination_response]
            else:
                dest_data = [{
                    "name": "Kyoto",
                    "country": "Japan",
                    "description": str(destination_response),
                    "best_time_to_visit": "Spring (March-May)",
                    "attractions": ["Fushimi Inari Shrine", "Kinkaku-ji"]
                }]

            destination = Destination(**dest_data[0])

            return TravelPlan(
                destination=destination,
                duration=user_request['duration'],
                budget=user_request['budget'],
                itinerary=[],
                transportation=[],
                total_estimated_cost=0
            )
        except Exception as e:
            logger.error(f"Error in plan_trip: {str(e)}")
            raise

# --------------------------
# 3. Streamlit UI
# --------------------------
def main():
    st.set_page_config(
        page_title="AI Travel Planner",
        page_icon="✈️",
        layout="wide"
    )

    st.title("✈️ AI Travel Planner")
    st.markdown("Get personalized travel plans powered by AI agents")

    with st.sidebar:
        st.header("Trip Details")
        departure = st.text_input("Departure City:", "New York")
        destination = st.text_input("Destination:", "Kyoto, Japan")  # Added Destination Input
        budget = st.selectbox("Budget Level:", ["Low", "Medium", "High"])
        duration = st.slider("Trip Duration (days)", 1, 30, 7)
        interests = st.multiselect("Interests:",
            ["Beaches", "Mountains", "Cities", "History", "Culture", "Food", "Adventure", "Nature"],
            ["History", "Culture"])

    if st.button("Create Travel Plan"):
        with st.spinner("Our AI agents are crafting your perfect trip..."):
            try:
                orchestrator = TravelOrchestrator()
                user_request = {
                    "departure_location": departure,
                    "destination_location": destination,
                    "budget": budget.lower(),
                    "duration": duration,
                    "interests": interests
                }

                travel_plan = orchestrator.plan_trip(user_request)
                st.success("Your travel plan is ready!")

                st.subheader(f"🌍 {departure} To {destination}")
                st.write(travel_plan.destination.description)

                st.subheader("Trip Details")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Duration", f"{travel_plan.duration} days")
                with col2:
                    st.metric("Budget Level", budget)

            except Exception as e:
                st.error(f"Error generating travel plan: {str(e)}")


if __name__ == "__main__":
    main()

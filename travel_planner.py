import streamlit as st
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import json
import logging
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
    def __init__(self, model="gpt-4-turbo"):
        self.llm = ChatOpenAI(model=model, temperature=0.5)
        self.tools = self._define_tools()
        self.agent = self._create_agent()

    def _define_tools(self) -> List[tool]:
        raise NotImplementedError

    def _create_agent(self) -> AgentExecutor:
        raise NotImplementedError

    def run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = self.agent.invoke({"input": str(input)})
            logger.info(f"Agent response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error in agent execution: {str(e)}")
            raise

class DestinationResearchAgent(BaseAgent):
    def _define_tools(self) -> List[tool]:
        @tool
        def search_destinations(query: str) -> List[Dict]:
            """Search for travel destinations matching criteria"""
            logger.info(f"Searching destinations for: {query}")
        return [search_destinations]

    def _create_agent(self) -> AgentExecutor:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a travel destination expert. Help users find the perfect destination based on:
             - Their interests
             - Budget level
             - Trip duration
             - Departure location

             Return a detailed response about why this destination matches their criteria."""),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, handle_parsing_errors=True)

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

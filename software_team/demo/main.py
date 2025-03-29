from AISoftTeam.agents.analyst import Analyst
import os

agent = Analyst()
response = agent.agent.run_sync(
    "How to write a langGraph project which simulates a developer team composed by an analyst, a coder, a tester and a reviewer?"
)

with open("response.md", "w") as f:
    f.write(response)

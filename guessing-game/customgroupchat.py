import random
from typing import List, Dict
from autogen.agentchat.agent import Agent
from autogen.agentchat.assistant_agent import AssistantAgent
from autogen.agentchat.groupchat import GroupChat


class CustomGroupChat(GroupChat):
    def __init__(self, agents, messages, max_round=10):
        super().__init__(agents, messages, max_round)
        self.previous_speaker = None  # Keep track of the previous speaker

    def select_speaker(self, last_speaker: Agent, selector: AssistantAgent):
        # Check if last message suggests a next speaker or termination
        last_message = self.messages[-1] if self.messages else None
        if last_message:
            if "NEXT:" in last_message["content"]:
                suggested_next = last_message["content"].split("NEXT: ")[-1].strip()
                print(f"Extracted suggested_next = {suggested_next}")
                try:
                    return self.agent_by_name(suggested_next)
                except ValueError:
                    pass  # If agent name is not valid, continue with normal selection
            elif "TERMINATE" in last_message["content"]:
                try:
                    return self.agent_by_name("User_proxy")
                except ValueError:
                    pass  # If 'User_proxy' is not a valid name, continue with normal selection

        team_leader_names = [agent.name for agent in self.agents if agent.name.endswith("1")]

        if last_speaker.name in team_leader_names:
            team_letter = last_speaker.name[0]
            possible_next_speakers = [
                agent
                for agent in self.agents
                if (agent.name.startswith(team_letter) or agent.name in team_leader_names)
                and agent != last_speaker
                and agent != self.previous_speaker
            ]
        else:
            team_letter = last_speaker.name[0]
            possible_next_speakers = [
                agent
                for agent in self.agents
                if agent.name.startswith(team_letter) and agent != last_speaker and agent != self.previous_speaker
            ]

        self.previous_speaker = last_speaker

        if possible_next_speakers:
            next_speaker = random.choice(possible_next_speakers)
            return next_speaker
        else:
            return None

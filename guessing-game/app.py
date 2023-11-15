import autogen
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union
from autogen.agentchat.assistant_agent import AssistantAgent
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
from customgroupchat import CustomGroupChat

config_list_4v = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4-vision-preview"],
    },
)

# Remove the `api_type` param as it is not needed for 4V
[config.pop("api_type", None) for config in config_list_4v]

config_list_gpt4 = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4", "gpt-4-0314", "gpt4", "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-v0314"],
    },
)

gpt4_llm_config = {"config_list": config_list_gpt4, "seed": 42}

llm_config = {"config_list": config_list_gpt4, "seed": 42}

agents_game_team = [
    AssistantAgent(
        name="Producer",
        system_message="Producer. You are responsible for defining the rules of the game and making sure the game is fun for the human.",
        llm_config=llm_config,
    ),
    AssistantAgent(
        name="Coder",
        system_message="Coder. You are part of a team and responsible for writing code for the game that will enforce the rules of the game.",
        llm_config=llm_config,
    ),
    AssistantAgent(
        name="Critic",
        system_message="Critic. You are part of a team and responsible for critiquing code written by the coder.",
        llm_config=llm_config,
    ),
]

agents_human_interaction = [
    AssistantAgent(
        name="Host",
        system_message="Host. You are responsible for enforcing the rules of the game and interacting with the human. First you must get an image URL from the human, then pass the image to the ImageInterpreter to determine the items in the image. Then you must ask the human to guess the items in the image. You must also keep track of the number of guesses the human has made and give them a score at the end of the game. Communicate with the Producer, Coder and Critic for help creating the game. ",
        llm_config=llm_config,
    ),
    MultimodalConversableAgent(
        name="ImageInterpreter",
        system_message="ImageInterpreter. You are responsible for interpreting the image and providing a list of items that are in the image.",
        max_consecutive_auto_reply=10,
        llm_config={"config_list": config_list_4v, "temperature": 0.5, "max_tokens": 300},
    ),
]


def is_termination_msg(content) -> bool:
    have_content = content.get("content", None) is not None
    if have_content and "TERMINATE" in content["content"]:
        return True
    return False


# Terminates the conversation when TERMINATE is detected.
user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="Terminator admin.",
    code_execution_config=False,
    is_termination_msg=is_termination_msg,
    human_input_mode="NEVER",
)

list_of_agents = agents_human_interaction + agents_game_team
list_of_agents.append(user_proxy)

# Create CustomGroupChat
group_chat = CustomGroupChat(
    agents=list_of_agents,  # Include all agents
    messages=[
        'Everyone cooperate and help the Host in their task. Agent-to-human team has the Host and Image Interpreter. Game team has Producer, Coder and Critic. Only members of the same team can talk to one another. Only the Host and Producer can talk amongst themselves. You must use "NEXT: Coder" to suggest talking to Coder for example; You can suggest only one person, you cannot suggest yourself or the previous speaker; You can also not suggest anyone.'
    ],
    max_round=30,
)


# Create the manager
llm_config = {
    "config_list": config_list_gpt4,
    "seed": 42,
    "use_cache": False,
}  # use_cache is False because we want to observe if there is any communication pattern difference if we reran the group chat.
manager = autogen.GroupChatManager(groupchat=group_chat, llm_config=llm_config)

agents_human_interaction[1].initiate_chat(manager, message="Start a guessing game with the Human.")

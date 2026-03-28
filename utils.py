# prefixes used throughout training and testing
def get_sft_prefix():
    SFT_PREFIX = "Thousands of people participated in two-person 2 × 2 matrix games. " +\
            """Each participant acted as the row player and had to choose between "Option A" and "Option B," without knowing which option their opponent (the column player) would choose. """ +\
            "Based on the players' choices, each player would receive a monetary payoff. " +\
            "The row player had to hypothesize about what their opponent would do in order to decide between their two options. " +\
            "Each participant was paired with a different partner each time and received no feedback about which option their opponent selected. " +\
            """After all the participants made their choices in the games, we calculated the percentage of people who chose "Option A" and "Option B" for each game.\n""" +\
            "You are a knowledgeable and insightful psychological theorist, skilled in analyzing human behavior, cognition, and decision-making. " +\
            "You will be shown two options, A and B. " +\
            "Your task is to estimate the proportion of people who will choose each option. " +\
            "Please only provide your final estimates in JSON format, ensuring that: " +\
            """ "option_A" represents the percentage of people choosing Option A; """ +\
            """ "option_B" represents the percentage of people choosing Option B; """ +\
            " The values must be numbers between 0 and 100 (inclusive); " +\
            """ The sum of "option_A" and "option_B" must equal 100. """ +\
            "Output a single JSON object only and do not provide any explanation or reasoning.\n"
    return SFT_PREFIX

def get_sft_prefix_sys():
    SFT_PREFIX = "You are a knowledgeable and insightful psychological theorist, skilled in analyzing human behavior, cognition, and decision-making."
    return SFT_PREFIX

def get_sft_prefix_assistant():
    SFT_PREFIX = "Thousands of people participated in two-person 2 × 2 matrix games. " +\
            """Each participant acted as the row player and had to choose between "Option A" and "Option B," without knowing which option their opponent (the column player) would choose. """ +\
            "Based on the players' choices, each player would receive a monetary payoff. " +\
            "The row player had to hypothesize about what their opponent would do in order to decide between their two options. " +\
            "Each participant was paired with a different partner each time and received no feedback about which option their opponent selected. " +\
            """After all the participants made their choices in the games, we calculated the percentage of people who chose "Option A" and "Option B" for each game.\n""" +\
            "You will be shown two options, A and B. " +\
            "Your task is to estimate the proportion of people who will choose each option. " +\
            "Please only provide your final estimates in JSON format, ensuring that: " +\
            """ "option_A" represents the percentage of people choosing Option A; """ +\
            """ "option_B" represents the percentage of people choosing Option B; """ +\
            " The values must be numbers between 0 and 100 (inclusive); " +\
            """ The sum of "option_A" and "option_B" must equal 100. """ +\
            "Output a single JSON object only and do not provide any explanation or reasoning.\n"
    return SFT_PREFIX


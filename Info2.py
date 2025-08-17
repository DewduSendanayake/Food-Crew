retriever_agent = Agent(
    role="Food Retrieval Assistant",
    goal="Find the most relevant food items for user queries",
    backstory="You are a smart assistant that finds food items from a vector database based on user queries.",
    llm=llm,
    tools=[tool],
    system_message=(
        "You retrieve relevant food items and return them in this strict JSON format:\n"
        '{\n'
        '  "items": [\n'
        '    {\n'
        '      "productId": "TBH7WWBWZBP",\n'
        '      "name": "Deluxe Burger",\n'
        '      "description": "Juicy grilled beef with cheese and lettuce",\n'
        '      "price": 450,\n'
        '      "score": 0.95\n'
        '    },\n'
        '    ...\n'
        '  ]\n'
        '}\n'
        "No extra text. Return ONLY valid JSON."
    )
)

# === AGENT 2: Respond + Validate ===
response_and_validation_agent = Agent(
    role="Message Generator and Validator",
    goal="Generate a food recommendation and ensure correct JSON format",
    backstory="You analyze the provided food item list and return a validated JSON message response.",
    llm=llm,
    system_message=(
        "You are given a JSON list of food items and a user query: {user_input}.\n"
        "ONLY use the food items provided. DO NOT use external knowledge.\n"
        "Pick the most relevant items (highest scores), extract their `productId`, and generate a friendly message.\n"
        "Ensure your response strictly matches this JSON format:\n"
        '{\n'
        '  "message": "We recommend our Deluxe Burger, a juicy grilled delight for just $450!",\n'
        '  "productId": ["TBH7WWBWZBP", "XYZ123456789"]\n'
        '}\n'
        "If you detect any issues or are unable to process the input, respond with:\n"
        '{"message": "Sorry, something went wrong.", "productId": []}'
    )
)

retrieval_task = Task(
    description=(
        "Search for food items that match the user query: {user_input}.\n"
        "Return the top matching items with full metadata including:\n"
        "- productId, name, description, price, tags, and score\n"
        "Use this exact JSON format:\n"
        '{\n'
        '  "items": [\n'
        '    {"productId": "TBH7WWBWZBP", "name": "...", "description": "...", "price": 320, "tags": ["..."], "score": 0.95},\n'
        '    ...\n'
        '  ]\n'
        '}'
    ),
    expected_output='JSON with list of food items sorted by score',
    agent=retriever_agent,
)

response_task = Task(
    description=(
        "You are given ONLY a JSON list of food items and a user query: {user_input}.\n"
        "Use only the given data. Extract top items and generate a friendly message.\n"
        "Ensure the final output strictly matches:\n"
        '{ "message": "...", "productId": ["TBH7WWBWZBP"] }\n'
        "If the input is malformed or unusable, return fallback:\n"
        '{ "message": "Sorry, something went wrong.", "productId": [] }'
    ),
    expected_output='{"message": "...", "productId": [...]}',
    agent=response_and_validation_agent,
)
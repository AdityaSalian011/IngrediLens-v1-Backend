# Project Overview

An AI haircare assistant, used for scanning hair ingredients from an image/ text and providing an analysis on them. 
This haircare assistant can also chat with the user for their further queries.


## Project Breakdown

### 1. Ingredient Extraction

In this section of my project, the user can provide a text or image input, consisting hair product *ingredients*. 
Our AI (haircare assist.) scans the input, and then extracts *ONLY* ingredients from it.
These ingredients are later stored inside a list.

For this, I have created a LLM worfklow using Langchain and Langgraph.
The workflow has two components
1. *ingredient extraction*: Extracting ingredients from the input and storing it inside a list.
2. *analysis generation*: Generates a short analysis on the ingredients, flagging harmful ingredients.

### 2. Chatbot

In this section, the user can ask further queries about the product ingredients, if they didn't get a satisfying analysis. 
User asks a question/ query, AI (haircare assist.) answers them.

This chatbot allows users to choose, how they want their answer.

- A regular response uses LLM's parametric knowledge to answer the user's query.
- A deep search response performs web search behind the scene, and uses internet knowledge to answer user's query.

Both of these response type are part of the Chatbot workflow.

This chatbot also has the concept of *Memory* implemented.
Both **STM** and **LTM** are added in this chatbot

- STM remembers what happened during a conversation. It is limited to a single conversation only. 
- LTM remembers important information about a user, and uses this information throughout different conversation when required.

Both type of memories are stored inside Postgres SQL of Supabase.

Now that we know this, the chatbot concept was implemented using the following LLM worflow.
1. *remember node*: As soon as a user asks a query, when it enters it firstly visits this node. The remember node decides, whether the *user query* has something worth storing for a long time. Ex. If a user inputs, "My hair is dry, which ingredient is most harmful for me?", since user shared their hair profile (dry hair), which is a important information, the remember decides to store this information for a long time (LTM).
2. *deep search*: This is a complicated worflow, and hence broken down as-
    1. *rewrite query*: Here, we rewrite user's generic query/ question into a search-engine-friendly query.
    2. *search*: We search the internet using the rewritten query, and select the top 5 searches. I am searching TavilySearch for this. After search, we extract the title, URL, and content from each search web page and store it in a list. Each element being Title: ..., URL: ..., Content: ...
    3. *refinement*: Since, we are extracting information from a webpage, chances of noise in them increases (ability to answer the query). In refinement, we remove those unwanted noise/ content from our web research and only keep meaningful content, that are relevant to the query.
    4. *deep search response generation*: 
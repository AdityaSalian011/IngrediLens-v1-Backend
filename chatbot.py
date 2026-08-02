from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langchain_core.prompts import ChatPromptTemplate

from langchain_tavily import TavilySearch
from langchain_core.documents import Document

from typing import TypedDict, List, Annotated, Literal, Optional
from pydantic import BaseModel, Field

from dotenv import load_dotenv
import os

import operator
import uuid
import numpy as np


load_dotenv()

llm = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0,
    api_key=os.getenv('AICREDITS_API_KEY'),
    base_url='https://api.aicredits.in/v1'
)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv('AICREDITS_API_KEY'),
    base_url='https://api.aicredits.in/v1'
)

# =================================================================
# PART 1 =====================
# ========== INGREDIENT EXTRACTION & ANALYSIS GENERATION ==========

# INGREDIENT STATE ================================================
class IngredState(TypedDict):

    input_format: Literal['textual', 'image']
    textual_inp: Optional[str]
    image_inp: Optional[str]
    image_mime_type: Optional[str]
    ingredients: List[str]
    analysis: str


class IngredSchema(BaseModel):
    """A pydantic schema, used to generate a structured LLM output, consisting list of ingredients."""

    ingredients: List[str] = Field(
        ...,
        description="List of hair ingredients."
    )

extract_ingred_llm = llm.with_structured_output(IngredSchema)


# NODE CREATION ===================================================
def extract_textual_ingred_node(state: IngredState) -> dict:
    """A node, used to extract ONLY ingredients from user text."""

    response: IngredSchema = extract_ingred_llm.invoke([
        SystemMessage(
            content=(
                "You are an ingredient extraction system.\n"
                "Extract ONLY ingredient names from the provided text.\n"
                "Rules\n"
                "Return every ingredient explicitly mentioned.\n"
                "Preserve the original ingredient names and spelling.\n"
                "Do not infer, expand, or generate ingredients.\n"
                "Ignore marketing claims, descriptions, instructions, warnings, and product names.\n"
                "Remove duplicates.\n"
                "Return ONLY ingredient names.\n"
                "If an ingredient is wrapped in [UNCLEAR: ...] or marked [ILLEGIBLE], "
                "keep it in the list in that same marked form rather than silently "
                "cleaning it up — downstream analysis needs to know it's uncertain.\n"
                "If no ingredients are found, return an empty list.\n"
            )
        ),
        HumanMessage(
            content=state.get('textual_inp')
        )
    ])

    return {
        'ingredients': response.ingredients
    }


def extract_image_ingred_node(state: IngredState) -> dict:
    """A node, used to extract ONLY ingredients from user provided image."""

    message = HumanMessage(
        content=[
            {
                'type': 'text',
                'text': (
                    "Extract and return ONLY the text visible in this image. "
                    "Preserve line breaks and formatting where possible. "
                    "Do not add commentary or explanation.\n\n"
                    "IMPORTANT — accuracy over completeness:\n"
                    "- If any word, ingredient name, or character is blurry, cut off, "
                    "obscured, or ambiguous, DO NOT guess or auto-correct it based on "
                    "what seems likely.\n"
                    "- Instead, transcribe your best reading and wrap it like this: "
                    "[UNCLEAR: your_best_guess].\n"
                    "- If a word is fully illegible, write [ILLEGIBLE] in its place — "
                    "do not omit it silently and do not invent a plausible-sounding "
                    "ingredient name.\n"
                    "- Never normalize, expand, or 'fix' a chemical/ingredient name to "
                    "match a name you recognize — transcribe exactly what is printed, "
                    "even if it looks like a misspelling."
                )
            },
            {
                'type': 'image',
                'source_type': 'base64',
                'data': state.get('image_inp'),
                'mime_type': state.get('image_mime_type')
            }
        ]
    )

    response = llm.invoke([message])

    return {
        'textual_inp': response.content
    }


def route_on_input_format(state: IngredState) -> str:

    if state.get('input_format') == 'image':
        return 'image_inp'
    else:
        return 'textual_inp'


def analysis_node(state: IngredState, config: RunnableConfig, store: BaseStore) -> dict:
    """A node, used to generate concise analysis based on the extracted ingredients."""

    ingredients = '- ' + '\n- '.join(state.get('ingredients'))

    user_id = config['configurable'].get('user_id', 'default_user')
    user_details = ('users', user_id, 'details')

    items = store.search(user_details)
    user_details_content = '\n'.join(f"- {item.value['data']}" for item in items) if items else '(empty)'

    response = llm.invoke([
        SystemMessage(
            content=(
                "You are a haircare ingredient analyst.\n"

                "USER HAIR PROFILE:\n"
                f"{user_details_content}\n"

                "Analyze the ingredients and respond in this exact format:\n\n"
                "Start with one sentence on the product's overall safety.\n"
                "If any harmful or potentially toxic ingredients are present, list them with a ⚠️ and briefly explain why they are concerning.\n"
                "If no harmful ingredients are found, say so in one sentence.\n"
                "End with one sentence summarizing whether this product is worth using based on user's hair profile.\n\n"
                "Rules:\n"
                "- Maximum 6-7 lines total.\n"
                "- Flag harmful ingredients clearly — this is the user's priority.\n"
                "- Write in plain, conversational English.\n\n"

                "If user hair profile is empty or insufficient, \n"
                "after your analysis, ask them once to share their hair type, texture and concerns "
                "so future analyses can be personalized."
            )
        ),
        HumanMessage(
            content=ingredients
        )
    ])

    return {
        'analysis': response.content
    }


# GRAPH CREATION + COMPILATION ====================================
extract_ingred_graph = StateGraph(IngredState)

# add a conditional node here
# on route_input_type()

extract_ingred_graph.add_node('extract_textual_ingred', extract_textual_ingred_node)
extract_ingred_graph.add_node('extract_image_ingred', extract_image_ingred_node)
extract_ingred_graph.add_node('analysis', analysis_node)

extract_ingred_graph.add_conditional_edges(
    START, 
    route_on_input_format,
    {
        'image_inp': 'extract_image_ingred',
        'textual_inp': 'extract_textual_ingred'
    }
)
extract_ingred_graph.add_edge('extract_image_ingred', 'extract_textual_ingred')
extract_ingred_graph.add_edge('extract_textual_ingred', 'analysis')
extract_ingred_graph.add_edge('analysis', END)


# =================================================================
# PART 2 =====================
# ==================== CHATBOT IMPLEMENTATION ====================

# CHATBOT STATE ===================================================
class ChatState(MessagesState):

    ingredients: List[str] 
    analysis: str
    summary: str = None
    full_history: Annotated[List[dict], operator.add] = []    # only for displaying full
                                                              # conv. on frontend

    # deep search
    deep_search: bool = False
    research_query: str
    research_content: List[Document]
    refined_content: List[Document]


class MemoryItem(BaseModel):

    is_new: bool = Field(description='True if this memory is NEW and should be stored, False if duplicate/ already known.')
    text: str = Field(description='Atomic user memory as a short sentence.')


class MemoryDecision(BaseModel):

    should_write: bool = Field(description='Whether to store any memories.')
    memories: List[MemoryItem] = Field(default_factory=list)

memory_extractor = llm.with_structured_output(MemoryDecision)


MEMORY_PROMPT = """You are responsible for maintaining accurate hair profile memory for a haircare assistant.

CURRENT USER HAIR PROFILE (existing memories):
{user_details_content}

TASK:
- Review the user's latest message.
- Extract ONLY hair-related information worth storing long-term:
    - Hair type (oily, dry, normal, combination)
    - Hair texture (thick, thin, medium)
    - Hair concerns (dandruff, hair fall, frizz, dryness, scalp sensitivity etc.)
    - Products or ingredients they prefer to avoid
- For each extracted item, set is_new=true ONLY if it adds NEW information not already in CURRENT USER HAIR PROFILE.
- If it is basically the same meaning as something already present, set is_new=false.
- Keep each memory as a short atomic sentence. (e.g. "User has oily scalp with dry ends.")
- No speculation; only facts explicitly stated by the user.
- Ignore questions, ingredient queries, and analysis requests — these are not memory-worthy.
- If there is nothing hair-profile-worthy, return should_write=false and an empty list.
"""


def remember_node(state: ChatState, config: RunnableConfig, store: BaseStore):
    """A node, used to store long-term user memory (e.g. user's hair texture, concerns etc.)."""

    # Step1: find user's memory location (namespace)
    user_id = config['configurable'].get('user_id', 'default_user')
    user_details = ('users', user_id, 'details')

    # Step2: fetch all stored memory
    existing_items = store.search(user_details)
    existing_texts = [item.value.get('data', '') for item in existing_items if item.value.get('data')]
    user_details_content = '\n'.join(f'- {text}' for text in existing_texts) if existing_texts else '(empty)'

    # Step3: create a system prompt using those memory
    system_prompt = MEMORY_PROMPT.format(user_details_content=user_details_content)
    # Note: We add previous user mems in system prompt to avoid duplication.

    last_user_msg = state.get('messages')[-1].content

    decision: MemoryDecision = memory_extractor.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=last_user_msg)
        ]
    )

    # Step4: Store worthy and new memories (non-duplicate) as an atomic text in user's memory location (namespace)
    if decision.should_write:
        for mem in decision.memories:
            if mem.is_new:
                store.put(user_details, str(uuid.uuid4()), {'data': mem.text})

    return {}


# =================================================================
# DEEP SEARCH NODE ================================================
class WebQuerySchema(BaseModel):
    """Rewrites a user's raw question into a clean, search-engine-friendly query."""
 
    query: str = Field(
        ...,
        description="A concise, search-engine-optimized rewrite of the user's question."
    )

query_rewriter_llm = llm.with_structured_output(WebQuerySchema)


def rewrite_query_node(state: ChatState) -> dict:
    """A node, used to rewrite the user's latest message into a search-engine-friendly query."""
 
    last_user_msg = state.get('messages')[-1].content
 
    result: WebQuerySchema = query_rewriter_llm.invoke([
        SystemMessage(
            content=(
                "You rewrite user questions into concise, search-engine-friendly queries.\n"
                "Preserve ingredient names exactly as written.\n"
                "Do not add ingredients or claims not implied by the user's question."
            )
        ),
        HumanMessage(content=last_user_msg)
    ])
 
    return {
        'research_query': result.query
    }

tavily_search = TavilySearch(
    max_results=5,
    search_depth='advanced'
)


def tavily_search_node(state: ChatState) -> str:
    """A node, used to perform tavily search on the provided query,
    and extracts content from the searched web page."""

    results = tavily_search.invoke({'query': state.get('research_query')})

    research_content = []
    for r in results.get('results', []):

        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "") or r.get("snippet", "")
        
        research_content.append(Document(page_content=content, metadata={"url": url, "title": title}))

    return {'research_content': research_content}


# -----------------------------
# FILTER (LLM judge)
# -----------------------------
class KeepOrDrop(BaseModel):
    keep: bool

filter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict relevance filter.\n"
            "Return keep=true only if the provided context directly helps answer the question.\n"
            "Use ONLY the context. Output JSON only.",
        ),
        ("human", "Question: {question}\nContext:\n{context}"),
    ]
)

filter_chain = filter_prompt | llm.with_structured_output(KeepOrDrop)


# -----------------------------
# REFINING (Decompose -> Filter -> Recompose)
# -----------------------------
def refine_node(state: ChatState) -> dict:
    """A node, used to extract only meaningful content from research_content."""
    refined_content = []
    last_user_msg = state.get("messages")[-1].content

    for doc in state.get("research_content", []):
        chunks = [c.strip() for c in doc.page_content.split("[...]") if c.strip()]
        ref_content = ""

        for c in chunks:
            response: KeepOrDrop = filter_chain.invoke({"question": last_user_msg, "context": c})

            if response.keep and c not in ref_content:
                ref_content += f"{c}\n"

        if ref_content.strip():
            refined_content.append(Document(page_content=ref_content.strip(), metadata=doc.metadata))

    return {"refined_content": refined_content}


TOP_K = 5

def cosine_sim(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b)/ (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def semantic_select_node(state: ChatState) -> dict:
    """A node, to select top K most semantically similar 'refined_content' to the user's message. Limits no. of input content being passed."""

    docs = state.get("refined_content", [])
    if not docs:
        return {"refined_content": []}

    query = state.get("messages")[-1].content
    query_vec = embedding_model.embed_query(query)
    doc_vec = embedding_model.embed_documents([d.page_content for d in docs])

    scored = [(cosine_sim(query_vec, v), d) for v, d in zip(doc_vec, docs)]
    scored.sort(key=lambda x: x[0], reverse=True)

    top_docs = [d for _, d in scored[:TOP_K]]
    return {"refined_content": top_docs}


def format_research_context(docs: List[Document]) -> str:
    if not docs:
        return '(no relevant research found)'

    sections = []
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get('title', 'Untitled source')
        url = doc.metadata.get('url', '')
        sections.append(f"[{i}] {title}\n{doc.page_content}\nSource: {url}")

    return "\n\n".join(sections)


def build_product_section(state: ChatState) -> str:
    """Builds the scanned-product context block, or a clear 'none scanned' notice if absent."""

    ingredients = state.get('ingredients')
    analysis = state.get('analysis')

    if not ingredients:
        return "(No product has been scanned in this conversation. Answer generally — do not assume any specific product or ingredients.)"

    ingredients_block = '- ' + '\n- '.join(ingredients)
    analysis_block = f"\n\nPREVIOUS ANALYSIS OF THIS PRODUCT:\n{analysis}" if analysis else ""

    return f"SCANNED PRODUCT INGREDIENTS:\n{ingredients_block}{analysis_block}"


DEEP_SEARCH_PROMPT_TEMPLATE = """You are IngrediLens, an intelligent haircare assistant.
The user asked a haircare-related question, and you have gathered live web research to help answer it.

USER QUESTION:
{user_query}

RESEARCH CONTEXT (from web search):
{refined_content}

{product_section}

{summary_section}

USER HAIR PROFILE:
{user_details_content}

Your behavior:
- Answer using the provided research context as your primary source for any external claims, studies, or findings — do not rely on outside knowledge beyond it.
- If a scanned product's ingredients and previous analysis are available above, connect the research to that product where relevant — e.g. if the research discusses an ingredient the user's scanned product contains.
- If the context is empty or insufficient to answer confidently, say so honestly rather than guessing.
- If the user's hair profile is available, tailor the answer to their specific hair type, texture, and concerns.
- If the hair profile is missing, answer generally and don't assume specifics — you may remind them once that sharing their hair profile would help personalize the answer.
- Flag harmful or concerning effects clearly with ⚠️.
- Keep responses concise and in plain conversational English — no long markdown sections.
- Do not fabricate research findings, studies, or claims not present in the provided context.
- If a source/url is available in RESEARCH CONTEXT, display it below the respective section.

At the end of each response, suggest 2-3 relevant follow-up questions the user might want to ask next.
"""


def deep_search_node(state: ChatState, config: RunnableConfig, store: BaseStore) -> dict:
    """A node, used to generate deep/ research-based answer for the user's query."""

    # Step1: fetch user's long-term memories (same pattern as chat_node)
    user_id = config['configurable'].get('user_id', 'default_user')
    user_details = ('users', user_id, 'details')

    items = store.search(user_details)
    user_details_content = '\n'.join(f"- {item.value['data']}" for item in items) if items else '(empty)'

    # Step2: Build product section
    product_section = build_product_section(state)

    # Step3: fetch conversation summary (if available), same as chat_node
    summary_section = (
        f"CONVERSATION SUMMARY:\n{state.get('summary')}"
        if state.get('summary')
        else ""
    )

    user_query = state.get('messages')[-1].content

    # Step4: build system prompt
    system_msg = DEEP_SEARCH_PROMPT_TEMPLATE.format(
        user_query=user_query,
        refined_content=format_research_context(state.get("refined_content")),
        product_section=product_section,        summary_section=summary_section,
        user_details_content=user_details_content
    )

    # Step5: invoke llm with system prompt + full message history (STM)
    response = llm.invoke([
        SystemMessage(content=system_msg),
        *state.get('messages')
    ])

    return {
        'full_history': [
            {'role': 'human', 'content': user_query},
            {'role': 'ai', 'content': response.content}
        ],
        'messages': [response]
    }


SYSTEM_PROMPT_TEMPLATE = """You are IngrediLens, an intelligent haircare assistant.
You help users with haircare questions in general — and, when they've scanned a specific product, with questions about that product's ingredients.

{product_section}

{summary_section}

USER HAIR PROFILE:
{user_details_content}

Your behavior:
- Answer the user's haircare question directly and conversationally.
- If a scanned product's ingredients and analysis are available above, use them as your primary context for anything product-specific — treat the previous analysis as already established, don't recompute or contradict it.
- If no product has been scanned, answer from general haircare knowledge — do not invent or assume a specific product's ingredients.
- If the user's hair profile is available, tailor every response to their specific hair type, texture, and concerns.
- If the hair profile is missing, remind the user once (not repeatedly) that sharing their hair type and concerns will help personalize your answers.
- Do not make up or assume the user's hair profile if it is not available.
- Whenever specific ingredients are discussed, flag harmful or concerning ones clearly with ⚠️.
- Keep responses concise and in plain conversational English — no long markdown sections.

At the end of each response, suggest 2-3 relevant follow-up questions the user might want to ask next.
"""


def chat_node(state: ChatState, config: RunnableConfig, store: BaseStore) -> dict:
    """A node, used to generate response using LT & ST Memory."""

    # Step1: fetch user's long-term memories (e.g. name, hair profiles etc.)
    user_id = config['configurable'].get('user_id', 'default_user')
    user_details = ('users', user_id, 'details')

    items = store.search(user_details)

    user_details_content = '\n'.join(f"- {item.value['data']}" for item in items) if items else '(empty)'

    # Step2: build product section
    product_section = build_product_section(state)

    # Step3: fetch conversation summary (if available)
    summary_section = (
        f"CONVERSATION SUMMARY:\n{state.get('summary')}"
        if state.get('summary')
        else ""
    )

    # Step4: create system prompt
    system_msg = SYSTEM_PROMPT_TEMPLATE.format(
        product_section=product_section,
        summary_section=summary_section,
        user_details_content=user_details_content
    )

    # Step5: invoke llm with system prompt and messages
    # (previous conversation between HUMAN & ASSISTANT)  <-- STM
    response = llm.invoke([ 
        SystemMessage(content=(system_msg)),
        *state.get('messages')
    ])

    return {
        'full_history': [
            {'role': 'human', 'content': state.get('messages')[-1].content},
            {'role': 'ai', 'content': response.content}
        ],
        'messages': [response]
    }


def route_to_research(state: ChatState) -> str:

    if state.get('deep_search'):
        return 'rewrite_query'
    else:
        return 'chat'


MAX_CONVERSATION_LEN = 6
MAX_SUMMARY_CHARS = 2000

def summary_node(state: ChatState) -> dict:
    """Creates a conversation summary, if conversation history get's bigger."""

    if len(state.get('messages')) <= MAX_CONVERSATION_LEN:
        return {}

    msgs_to_summarize = state.get('messages')[:-4]
    existing_summary = state.get('summary')

    if existing_summary and len(existing_summary) < MAX_SUMMARY_CHARS:
        summary_prompt = (
            f"Existing summary:\n{existing_summary}\n\n"
            "Extend the summary using the new conversation. Keep the total summary concise — "
            "prioritize recent and still-relevant information over exhaustive detail."
        )
    
    else:
        # Either no summary yet, or it's grown too large — force a compressed rewrite
        # instead of appending further.
        summary_prompt = (
            f"Existing summary (condense this, don't just append):\n{existing_summary or ''}\n\n"
            "Summarize the conversation concisely, in under 200 words, keeping only "
            "information still relevant to future turns (hair profile facts, ongoing concerns, "
            "product context)." if existing_summary else "Summarize the conversation concisely, in under 200 words."
        )

    response = llm.invoke([
        SystemMessage(content=summary_prompt),
        HumanMessage(
            content=(
                "\n".join(f"{m.type}: {m.content}" for m in msgs_to_summarize)
            )
        )
    ])

    return {
        'summary': response.content,
        'messages': [RemoveMessage(id=m.id) for m in msgs_to_summarize]
    }

def should_summarize(state: ChatState) -> str:

    return len(state.get('messages')) > 4

# GRAPH CREATION + COMPILATION ====================================
chatbot_graph = StateGraph(ChatState)

# LTM rememberance node
chatbot_graph.add_node('remember', remember_node)

# deep search nodes
chatbot_graph.add_node('rewrite_query', rewrite_query_node)
chatbot_graph.add_node('tavily_search', tavily_search_node)
chatbot_graph.add_node('refine', refine_node)
chatbot_graph.add_node("semantic_select", semantic_select_node)
chatbot_graph.add_node('deep_search', deep_search_node)

# regular chat node
chatbot_graph.add_node('chat', chat_node)

# STM summary node
chatbot_graph.add_node('summary', summary_node)

chatbot_graph.add_edge(START, 'remember')
chatbot_graph.add_conditional_edges(
    'remember',
    route_to_research,
    {
        'rewrite_query': 'rewrite_query',
        'chat': 'chat'
    }
)

chatbot_graph.add_edge('rewrite_query', 'tavily_search')
chatbot_graph.add_edge('tavily_search', 'refine')
chatbot_graph.add_edge('refine', 'semantic_select')
chatbot_graph.add_edge("semantic_select", "deep_search")

chatbot_graph.add_conditional_edges(
    'deep_search',
    should_summarize,
    {
        True: 'summary',
        False: END
    }
)

chatbot_graph.add_conditional_edges(
    'chat',
    should_summarize,
    {
        True: 'summary',
        False: END
    }
)

chatbot_graph.add_edge('summary', END)
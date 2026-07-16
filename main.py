import asyncio
import sys

if sys.platform=='win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from chatbot import extract_ingred_graph, chatbot_graph
from langchain_core.messages import HumanMessage

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg_pool import AsyncConnectionPool

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager

from pydantic import BaseModel

from database import Base, engine, get_db
from models import User, Scan
from schemas import UserCreate, UserLogin, UserOut, Token
from auth import hash_password, verify_password, create_access_token, get_current_user

from sqlalchemy.orm import Session
from sqlalchemy import text    ## CRON JOB

import os
import uuid
import traceback
import json
import base64


# ============================= FAST API INITIALIZATION =============================
# WITH CHATBOT AND EXTRACT_INGRED ===================================================
chatbot = None
extract_ingred = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chatbot, extract_ingred

    pool = AsyncConnectionPool(
        conninfo=os.getenv('DB_URI'),
        kwargs={'autocommit': True},
        open=False,
    )
    await pool.open()

    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()

    store = AsyncPostgresStore(pool)
    await store.setup()

    extract_ingred = extract_ingred_graph.compile(
        store=store
    )

    chatbot = chatbot_graph.compile(
        checkpointer=checkpointer,
        store=store
    )    
    
    Base.metadata.create_all(bind=engine)

    yield 
    await pool.close()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ============================= REQUEST SCHEMA CREATION =============================
class ScanTextRequest(BaseModel):

    text_input: str

class ChatRequest(BaseModel):

    message: str
    thread_id: str
    deep_search: bool

# ================================== PATH CREATION ==================================

@app.get('/health')
def health_check(db: Session = Depends(get_db)):
    
    try:
        db.execute(text('SELECT 1'))
        return {'status': 'ok', 'db': 'connected'}
    
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return {'status': 'ok', 'db': 'error', 'detail': str(e)}
    

@app.post('/auth/signup', response_model=UserOut, status_code=201)
async def signup(req: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == req.email).first()
    if existing:
        raise HTTPException(status_code=400, detail='Email already registered.')

    user = User(email=req.email, hashed_password=hash_password(req.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post('/auth/login', response_model=Token)
async def login(req: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail='Invalid email or password.')

    token = create_access_token(data={'sub': str(user.id)})
    return Token(access_token=token)


@app.post('/auth/logout')
async def logout(current_user: User = Depends(get_current_user)):
    # Stateless JWT — actual invalidation happens client-side by discarding the token.
    return {'detail': 'Logged out. Please discard your access token client-side.'}


@app.get('/auth/me', response_model=UserOut)
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.post('/scan-image-ingredients')
async def scan_image_ingredients(
    image: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Scans user image and extracts ingredients from it,
    Followed by an analysis/ summary on it."""

    try:
        raw_byte = await image.read()
        encoded = base64.b64encode(raw_byte).decode(encoding='utf-8')
        mime_type = image.content_type

        # Step1: Extract ingredients
        response = await extract_ingred.ainvoke(
            {
                'input_format': 'image',
                'image_inp': encoded,
                'image_mime_type': mime_type
            },
            config={'configurable': {'user_id': str(current_user.id)}}
        )

        # Fallback, if ingredients are not found
        if not response.get('ingredients'):
            raise HTTPException(
                status_code=400,
                detail="No ingredients found. Please provide product ingredients for analysis."
            )
        
        # Step2: Unique thread id generation (each conv. will have a unique thread_id).
        thread_id = str(uuid.uuid4())

        # Step3 (IMP): Updates ChatbotState
        # adds ingredient and analysis, so that STM saves it, and in future we do .invoke() without 
        # ingredients and analysis
        await  chatbot.aupdate_state(
            config={'configurable': {'thread_id': thread_id}},
            values={
                'ingredients': response['ingredients'],
                'analysis': response['analysis']
            }
        )

        # Step4: Record ownership
        ## Map thread_id's (conversation) to their respective users..
        title = image.filename or 'Image scan'
        scan = Scan(
            thread_id=thread_id,
            title=f"{title[:20]}..." if len(title) > 20 else title,
            user_id=current_user.id
        )
        db.add(scan)
        db.commit()

        # Step5: Send data to frontend
        return {
            'thread_id': thread_id,
            'ingredients': response['ingredients'],
            'analysis': response['analysis']
        }
    
    # Edge cases
    except HTTPException:
        raise

    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        raise HTTPException(
            status_code=502,
            detail=f"Something went wrong.\n\nException:\n{e}"
        )

@app.post('/scan-textual-ingredients')
async def scan_textual_ingredients(
    req: ScanTextRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Scans user text and extracts ingredients from it,
    Followed by an analysis/ summary on it."""

    try:
        # Step1: Extract ingredients
        response = await extract_ingred.ainvoke(
            {
                'input_format': 'textual',
                'textual_inp': req.text_input
            },
            config={'configurable': {'user_id': str(current_user.id)}}
        )

        # Fallback, if ingredients are not found
        if not response.get('ingredients'):
            raise HTTPException(
                status_code=400,
                detail="No ingredients found. Please provide product ingredients for analysis."
            )
        
        # Step2: Unique thread id generation (each conv. will have a unique thread_id).
        thread_id = str(uuid.uuid4())

        # Step3 (IMP): Updates ChatbotState
        # adds ingredient and analysis, so that STM saves it, and in future we do .invoke() without 
        # ingredients and analysis
        await  chatbot.aupdate_state(
            config={'configurable': {'thread_id': thread_id}},
            values={
                'ingredients': response['ingredients'],
                'analysis': response['analysis']
            }
        )

        # Step4: Record ownership
        ## Map thread_id's (conversation) to their respective users..
        scan = Scan(
            thread_id=thread_id,
            title=f"{req.text_input[:20]}..." if len(req.text_input) > 20 else req.text_input,
            user_id=current_user.id
        )
        db.add(scan)
        db.commit()

        # Step5: Send data to frontend
        return {
            'thread_id': thread_id,
            'ingredients': response['ingredients'],
            'analysis': response['analysis']
        }
    
    # Edge cases
    except HTTPException:
        raise

    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        raise HTTPException(
            status_code=502,
            detail=f"Something went wrong.\n\nException:\n{e}"
        )
    
ALLOWED_STREAM_NODES = {'chat', 'deep_search'}
    
async def event_generator(req: ChatRequest, user_id: str):

    input_content = {
        'messages': [HumanMessage(content=req.message)],
        'deep_search': req.deep_search
    }

    try:
        async for msg_chunk, msg_metadata in chatbot.astream(
            input_content,
            config={
                'configurable': {
                    'thread_id': req.thread_id,
                    'user_id': user_id
                },
                'metadata': {
                    'thread_id': req.thread_id,
                    'user_id': user_id
                }   # searchable in LangSmith UI
            },
            stream_mode='messages'
        ):

            if msg_metadata.get('langgraph_node') not in ALLOWED_STREAM_NODES:
                continue

            if msg_chunk.content:
                payload = json.dumps({'token': msg_chunk.content})
                yield f"data: {payload}\n\n"

    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        error_payload = json.dumps({'error': f"Something went wrong: {e}"})
        yield f"data: {error_payload}\n\n"

    finally:
        yield "data: [DONE]\n\n"

@app.post('/chat')
async def chat(req: ChatRequest, current_user: User = Depends(get_current_user)):
    """A path, to have conversation with AI.
    Expects a user message,
    thread_id for STM operations,
    user_id for LTM operations.
    Returns a AI generated response to the frontend."""
    
    try:

        return StreamingResponse(
            event_generator(req, user_id=str(current_user.id)),
            media_type='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no'
            }
        )
    
    # edge cases
    except HTTPException:
        raise

    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        raise HTTPException(
            status_code=502,
            detail=f"Something went wrong.\n\nException:\n{e}"
        )
    
@app.get('/conv-history')
async def get_conversation_history(
    thread_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """A path, to display conversation history between Human and AI.
    Requires thread_id to retrieve conversation snapshot from ChatbotState."""
    
    try:
        # Step1: Check if given thread_id exists in Scan
        ## and if the thread_id belongs to the current user..
        scan = db.query(Scan).filter(Scan.thread_id==thread_id).first()

        if not scan or scan.user_id!=current_user.id:

            raise HTTPException(status_code=404, detail='Conversation history not found.')
        
        # Step2: get thread-based State Snapshot
        snapshot = await chatbot.aget_state(
            config={'configurable': {'thread_id': thread_id}}
        )

        # If snapshot is empty, return an error message
        if not snapshot.values:
            raise HTTPException(
                status_code=404,
                detail='Conversation history not found.'
            )    

        # Step3: Return previous analysis, along with conversations
        return {
            'analysis': snapshot.values.get('analysis'),
            'messages': snapshot.values.get('full_history', [])
        }
    
    # edge cases
    except HTTPException:
        raise

    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        raise HTTPException(
            status_code=502,
            detail=f"Something went wrong.\n\nException:\n{e}"
        )
    
@app.get('/scans')
async def list_scans(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    
    scans = db.query(Scan).filter(
        Scan.user_id==current_user.id
    ).order_by(
        Scan.created_at.desc()
    ).all()

    return [{
        'thread_id': str(s.thread_id),
        'title': s.title,
        'date': s.created_at
    } for s in scans]

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host='0.0.0.0', port=8000, reload=True, loop="asyncio")
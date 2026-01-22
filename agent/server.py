# agent/server.py

import os
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# 기존 client.py에서 AgentClient 클래스를 가져옵니다.
# (같은 디렉토리에 있으므로 상대 경로 import 사용)
from .client import AgentClient

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("agent-server")

# 전역 클라이언트 인스턴스 (싱글톤으로 상태 유지)
client: Optional[AgentClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 AgentClient를 초기화하고 종료 시 정리합니다."""
    global client
    try:
        logger.info("Initializing AgentClient...")
        client = AgentClient()
        
        # MCP 서버 연결 확인
        if client.check_mcp_server():
            logger.info("Successfully connected to MCP server.")
        else:
            logger.warning("Could not connect to MCP server. Agent functionality may be limited.")
            
    except Exception as e:
        logger.error(f"Failed to initialize AgentClient: {e}")
        # 치명적 오류 시 서버 시작 실패 처리
        raise e
    
    yield
    
    logger.info("Shutting down Agent Server")

# FastAPI 앱 생성
app = FastAPI(title="Research Agent Server", lifespan=lifespan)

# CORS 설정 (웹 UI 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 데이터 모델 정의 ---

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, Any]] = []  # 웹에서 보내주는 히스토리 (현재는 서버 메모리 사용으로 인해 참조용)

class ChatResponse(BaseModel):
    response: str

# --- 엔드포인트 정의 ---

@app.get("/health")
async def health():
    """상태 확인용 엔드포인트"""
    mcp_status = client.check_mcp_server() if client else False
    return {
        "status": "healthy",
        "agent_initialized": client is not None,
        "mcp_connected": mcp_status
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """웹 UI와 대화하는 엔드포인트"""
    global client
    if not client:
        raise HTTPException(status_code=503, detail="Agent is not initialized")
    
    try:
        logger.info(f"Received user message: {request.message}")
        
        # AgentClient.chat()을 호출하여 답변 생성
        # (AgentClient 내부 메모리에 대화 흐름이 저장됨)
        response_text = await client.chat(request.message)
        
        return ChatResponse(response=response_text)
        
    except Exception as e:
        logger.error(f"Error during chat processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Docker 내부에서 실행되므로 외부 접속을 위해 0.0.0.0 바인딩
    uvicorn.run(app, host="0.0.0.0", port=8001)
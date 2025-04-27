from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.chatbot import get_chatbot_response

app = FastAPI()

class UserInput(BaseModel):
    query: str

@app.post("/chat")
async def chat_with_bot(user_input: UserInput):
    try:
        response = get_chatbot_response(user_input.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

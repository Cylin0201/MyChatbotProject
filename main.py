#CLI 실행 명령어 uvicorn main:app --reload

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message" : "Hello World!"}

@app.get("/home")
def home():
    return {"message" : "home"}



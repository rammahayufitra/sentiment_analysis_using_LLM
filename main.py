from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse  # Import HTMLResponse from fastapi.responses
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Define the endpoint for sentiment analysis
class RequestModel(BaseModel):
    input: str
    Lang: str  # Add Lang field to accept language selection

@app.post("/sentiment")
def get_response(request: RequestModel):
    try:
        prompt = request.input
        Lang = request.Lang  # Get Lang value from request
        if Lang == "EN":
            pipe = pipeline(model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
        elif Lang == "IND":
            pipe = pipeline(model="w11wo/indonesian-roberta-base-sentiment-classifier")
        else:
            raise HTTPException(status_code=400, detail="Unsupported language")

        response = pipe(prompt)
        label = response[0]["label"]
        score = response[0]["score"]
        return {"input": prompt, "label": label, "score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing input: {str(e)}")

@app.get("/")
async def read_main():
    # Render the index.html from 'templates' folder
    content = open("templates/index.html", "r").read()
    return HTMLResponse(content=content)  # Return HTMLResponse with content

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

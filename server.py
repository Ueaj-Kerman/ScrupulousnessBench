import os
import json
import re
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import httpx

app = FastAPI()

DATA_DIR = Path(__file__).parent / "data"
STATIC_DIR = Path(__file__).parent / "static"
DATA_DIR.mkdir(exist_ok=True)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")


def get_existing_names() -> set[str]:
    names = set()
    for f in DATA_DIR.iterdir():
        if f.is_file():
            names.add(f.stem)
    return names


def sanitize_name(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r'[^a-z0-9_]', '_', name)
    name = re.sub(r'_+', '_', name)
    return name.strip('_')[:50]


def make_unique_name(base: str, existing: set[str]) -> str:
    base = sanitize_name(base)
    if not base:
        base = "image"
    if base not in existing:
        return base
    i = 2
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


async def call_openrouter(messages: list[dict], max_tokens: int = 256) -> str:
    if not OPENROUTER_API_KEY:
        raise HTTPException(500, "OPENROUTER_API_KEY not set")

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "google/gemini-2.0-flash-001",
                "messages": messages,
                "max_tokens": max_tokens,
            },
        )
        if resp.status_code != 200:
            raise HTTPException(resp.status_code, f"OpenRouter error: {resp.text}")
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/examples")
async def list_examples():
    examples = []
    for f in sorted(DATA_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            img_ext = data.get("image_ext", "png")
            img_path = DATA_DIR / f"{f.stem}.{img_ext}"
            examples.append({
                "name": f.stem,
                "question": data.get("question", ""),
                "answer": data.get("answer", ""),
                "tweet": data.get("tweet", ""),
                "image_ext": img_ext,
                "has_image": img_path.exists(),
            })
        except:
            pass
    return examples


@app.get("/api/image/{name}")
async def get_image(name: str):
    for ext in ["png", "jpg", "jpeg", "gif", "webp"]:
        path = DATA_DIR / f"{name}.{ext}"
        if path.exists():
            return FileResponse(path)
    raise HTTPException(404, "Image not found")


@app.post("/api/generate-name")
async def generate_name(tweet: str = Form(default=""), question: str = Form(default="")):
    existing = get_existing_names()
    context = tweet or question or "visual puzzle"

    messages = [
        {
            "role": "system",
            "content": "Generate a short (1-3 word) descriptive filename for an image based on the context. Output ONLY the filename, no extension, no explanation. Use snake_case.",
        },
        {"role": "user", "content": f"Context: {context[:500]}"},
    ]

    try:
        name = await call_openrouter(messages, max_tokens=32)
        name = sanitize_name(name)
    except:
        name = "puzzle"

    return {"name": make_unique_name(name, existing)}


@app.post("/api/generate-question")
async def generate_question(tweet: str = Form(...)):
    messages = [
        {
            "role": "system",
            "content": """You are helping create a visual reasoning benchmark. Given a tweet about an image puzzle/illusion, generate a clear question that tests whether an AI can see through the trick.

The question should:
- Be answerable by looking at the image
- Test attention to detail or avoiding assumptions
- Be phrased neutrally (not giving away the trick)

Output ONLY the question, nothing else.""",
        },
        {"role": "user", "content": f"Tweet: {tweet}"},
    ]

    question = await call_openrouter(messages, max_tokens=128)
    question = question.strip('"\'')
    return {"question": question}


@app.post("/api/save")
async def save_example(
    name: str = Form(...),
    question: str = Form(...),
    answer: str = Form(...),
    tweet: str = Form(default=""),
    image: UploadFile = File(None),
    image_ext: str = Form(default="png"),
):
    name = sanitize_name(name)
    if not name:
        raise HTTPException(400, "Invalid name")

    existing = get_existing_names()
    json_path = DATA_DIR / f"{name}.json"

    is_update = json_path.exists()

    if image:
        ext = image.filename.split('.')[-1].lower() if '.' in image.filename else "png"
        if ext not in ["png", "jpg", "jpeg", "gif", "webp"]:
            ext = "png"
        image_ext = ext

        if not is_update:
            for e in ["png", "jpg", "jpeg", "gif", "webp"]:
                old_img = DATA_DIR / f"{name}.{e}"
                if old_img.exists():
                    old_img.unlink()

        img_path = DATA_DIR / f"{name}.{ext}"
        content = await image.read()
        img_path.write_bytes(content)

    data = {
        "question": question,
        "answer": answer,
        "tweet": tweet,
        "image_ext": image_ext,
    }
    json_path.write_text(json.dumps(data, indent=2))

    return {"success": True, "name": name}


@app.delete("/api/example/{name}")
async def delete_example(name: str):
    name = sanitize_name(name)
    json_path = DATA_DIR / f"{name}.json"

    if not json_path.exists():
        raise HTTPException(404, "Example not found")

    try:
        data = json.loads(json_path.read_text())
        img_ext = data.get("image_ext", "png")
    except:
        img_ext = "png"

    json_path.unlink()

    for ext in ["png", "jpg", "jpeg", "gif", "webp"]:
        img_path = DATA_DIR / f"{name}.{ext}"
        if img_path.exists():
            img_path.unlink()

    return {"success": True}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)

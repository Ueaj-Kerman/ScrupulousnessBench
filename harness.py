#!/usr/bin/env python3
"""
Vision benchmark harness for ScrupulousnessBench.

Tests whether vision models fall for visual tricks/illusions or carefully
analyze images to give accurate answers.

Uses OpenRouter API. Claude Opus 4.5 as the judge (with image context).

Usage:
    python harness.py --models google/gemini-2.5-pro-preview openai/gpt-4o
    python harness.py --models anthropic/claude-sonnet-4 --samples 3
"""

import argparse
import asyncio
import base64
import io
import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import aiohttp
from PIL import Image
from tqdm import tqdm

try:
    import moondream as md
    MOONDREAM_AVAILABLE = True
except ImportError:
    MOONDREAM_AVAILABLE = False

import decrypt_data
RESULTS_DIR = Path(__file__).parent / "results"

GRADER_MODEL = "anthropic/claude-opus-4.5"

# Provider configs to ensure full precision / exacto variants
# See: https://openrouter.ai/docs/features/exacto-variant
# and: https://openrouter.ai/docs/guides/routing/provider-selection
PROVIDER_CONFIG = {
    # Anthropic - use direct provider
    "anthropic/": {
        "order": ["Anthropic"],
        "allow_fallbacks": False,
    },
    # OpenAI - use direct provider
    "openai/": {
        "order": ["OpenAI"],
        "allow_fallbacks": False,
    },
    # Google - use direct provider
    "google/": {
        "order": ["Google"],
        "allow_fallbacks": False,
    },
    # Kimi - prefer original provider (Moonshot AI)
    "moonshotai/kimi": {
        "order": ["Moonshot AI"],
        "allow_fallbacks": False,
    },
    # GLM - prefer original provider (Z.AI)
    "z-ai/glm": {
        "order": ["Z.AI"],
        "allow_fallbacks": False,
    },
    # Grok - prefer xAI direct
    "x-ai/grok": {
        "order": ["xAI"],
        "allow_fallbacks": False,
    },
    # DeepSeek - prefer original provider for full precision
    "deepseek/": {
        "order": ["DeepSeek"],
        "allow_fallbacks": False,
    },
    # Qwen - prefer Alibaba direct
    "qwen/": {
        "order": ["Alibaba"],
        "allow_fallbacks": False,
    },
}


def get_provider_config(model: str) -> dict | None:
    """Get provider config for a model to ensure full precision."""
    model_lower = model.lower()
    for pattern, config in PROVIDER_CONFIG.items():
        if pattern in model_lower:
            return config
    return None


def get_mime_type(ext: str) -> str:
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
    }.get(ext.lower(), "image/png")


@dataclass
class Example:
    name: str
    question: str
    answer: str
    image_bytes: bytes
    image_ext: str
    tweet: str = ""

    @property
    def image_base64(self) -> str:
        return base64.b64encode(self.image_bytes).decode("utf-8")

    @property
    def mime_type(self) -> str:
        return get_mime_type(self.image_ext)


def load_examples() -> list[Example]:
    examples = []
    for name in sorted(decrypt_data.list_examples()):
        try:
            data = decrypt_data.get_json(name)
            img_ext = data.get("image_ext", "png")

            # Try to find the image with the specified extension
            try:
                img_bytes = decrypt_data.get_image_bytes(name, img_ext)
            except FileNotFoundError:
                # Try other extensions
                img_bytes = None
                for ext in ["png", "jpg", "jpeg", "gif", "webp"]:
                    try:
                        img_bytes = decrypt_data.get_image_bytes(name, ext)
                        img_ext = ext
                        break
                    except FileNotFoundError:
                        continue

            if img_bytes is None:
                tqdm.write(f"Warning: No image found for {name}, skipping")
                continue

            examples.append(Example(
                name=name,
                question=data.get("question", ""),
                answer=data.get("answer", ""),
                image_bytes=img_bytes,
                image_ext=img_ext,
                tweet=data.get("tweet", ""),
            ))
        except Exception as e:
            tqdm.write(f"Warning: Failed to load {name}: {e}")

    return examples


@dataclass
class TaskKey:
    example_name: str
    llm: str
    sample_idx: int

    def to_tuple(self) -> tuple:
        return (self.example_name, self.llm, self.sample_idx)


@dataclass
class TaskResult:
    key: TaskKey
    question: str
    expected_answer: str
    content: str | None = None
    thinking: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error: str | None = None
    score: float | None = None
    explanation: str | None = None
    model_id: str | None = None  # Full OpenRouter model ID

    def is_graded(self) -> bool:
        return self.score is not None

    def to_dict(self) -> dict:
        d = {
            "example_name": self.key.example_name,
            "llm": self.key.llm,
            "sample_idx": self.key.sample_idx,
            "question": self.question,
            "expected_answer": self.expected_answer,
            "content": self.content,
            "thinking": self.thinking,
            "timestamp": self.timestamp,
            "error": self.error,
            "score": self.score,
            "explanation": self.explanation,
        }
        if self.model_id:
            d["model_id"] = self.model_id
        return d


@dataclass
class Progress:
    results: dict[tuple, TaskResult] = field(default_factory=dict)

    def add(self, result: TaskResult):
        self.results[result.key.to_tuple()] = result

    def get(self, key: TaskKey) -> TaskResult | None:
        return self.results.get(key.to_tuple())

    def has(self, key: TaskKey) -> bool:
        return key.to_tuple() in self.results

    def to_dict(self) -> dict:
        return {"results": [r.to_dict() for r in self.results.values()]}

    @classmethod
    def from_dict(cls, data: dict) -> "Progress":
        progress = cls()
        for r in data.get("results", []):
            key = TaskKey(r["example_name"], r["llm"], r["sample_idx"])
            result = TaskResult(
                key=key,
                question=r.get("question", ""),
                expected_answer=r.get("expected_answer", ""),
                content=r.get("content"),
                thinking=r.get("thinking"),
                timestamp=r.get("timestamp", ""),
                error=r.get("error"),
                score=r.get("score"),
                explanation=r.get("explanation"),
                model_id=r.get("model_id"),
            )
            progress.add(result)
        return progress


def load_progress(path: Path, results_path: Path | None = None) -> Progress:
    """Load progress from progress file, or fall back to results file."""
    if path.exists():
        try:
            return Progress.from_dict(json.loads(path.read_text()))
        except Exception as e:
            tqdm.write(f"Warning: Could not load progress: {e}")

    # Fall back to results file if progress file doesn't exist
    if results_path and results_path.exists():
        try:
            tqdm.write(f"Loading existing results from {results_path}")
            return Progress.from_dict(json.loads(results_path.read_text()))
        except Exception as e:
            tqdm.write(f"Warning: Could not load results: {e}")

    return Progress()


def save_progress(progress: Progress, path: Path):
    path.write_text(json.dumps(progress.to_dict(), indent=2))


def extract_thinking(text: str) -> tuple[str, str | None]:
    if not text or "<think>" not in text:
        return text, None
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    segments = pattern.findall(text)
    thinking = "\n".join(segments) if segments else None
    return pattern.sub("", text).strip(), thinking


def parse_model_spec(model_spec: str) -> tuple[str, dict | None]:
    if ":" in model_spec:
        parts = model_spec.rsplit(":", 1)
        suffix = parts[1].lower()
        if suffix in {"none", "minimal", "low", "medium", "high", "xhigh"}:
            return parts[0], {"effort": suffix}
        if suffix.isdigit():
            return parts[0], {"max_tokens": int(suffix)}
        if suffix in {"false", "off", "disabled"}:
            return parts[0], {"disabled": True}
        if suffix in {"reasoning", "think", "true", "on"}:
            return parts[0], {"reasoning": True}
    return model_spec, None


def is_moondream_model(model: str) -> bool:
    return model.lower().startswith("moondream")


class RateLimiter:
    """Simple rate limiter using token bucket."""
    def __init__(self, rate: float):
        self.rate = rate  # requests per second
        self.last_time = 0.0
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = asyncio.get_event_loop().time()
            min_interval = 1.0 / self.rate
            wait_time = self.last_time + min_interval - now
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.last_time = asyncio.get_event_loop().time()


class LLMClient:
    def __init__(self, session: aiohttp.ClientSession, max_retries: int = 3):
        self.session = session
        self.max_retries = max_retries
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        self.moondream_api_key = os.environ.get("MOONDREAM_API_KEY")
        self.moondream_client = None
        self.moondream_rate_limiter = RateLimiter(10.0)  # 10 req/sec
        if not self.api_key:
            raise SystemExit("OPENROUTER_API_KEY not set")

    def _get_moondream_client(self):
        if self.moondream_client is None:
            if not MOONDREAM_AVAILABLE:
                raise SystemExit("moondream package not installed. Run: pip install moondream")
            if not self.moondream_api_key:
                raise SystemExit("MOONDREAM_API_KEY not set")
            self.moondream_client = md.vl(api_key=self.moondream_api_key)
        return self.moondream_client

    async def query_moondream(
        self,
        question: str,
        image_base64: str,
        reasoning: bool = False,
    ) -> dict | None:
        """Query Moondream API using the SDK."""
        await self.moondream_rate_limiter.acquire()
        for attempt in range(self.max_retries):
            try:
                # Decode base64 to PIL Image
                image_bytes = base64.b64decode(image_base64)
                image = Image.open(io.BytesIO(image_bytes))

                # Run in thread pool since SDK is sync
                client = self._get_moondream_client()
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: client.query(image, question, reasoning=reasoning)
                )

                answer = result.get("answer", "")
                # Extract reasoning if present
                thinking = None
                if "reasoning" in result and isinstance(result["reasoning"], dict):
                    thinking = result["reasoning"].get("text")
                return {"content": answer, "thinking": thinking}

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = (2 ** attempt) * 2
                    tqdm.write(f"Moondream error: {e}. Retry in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    tqdm.write(f"Moondream query failed: {e}")
                    return None
        return None

    async def query_vision(
        self,
        question: str,
        image_base64: str,
        mime_type: str,
        model: str,
        reasoning_config: dict | None = None,
    ) -> dict | None:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_base64}"},
                },
                {"type": "text", "text": f"Please answer the following question about this image:\n\n{question}"},
            ],
        }]

        data = {
            "model": model,
            "messages": messages,
            "include_reasoning": True,
        }

        # Add provider config for full precision routing
        provider_config = get_provider_config(model)
        if provider_config:
            data["provider"] = provider_config

        if reasoning_config:
            if reasoning_config.get("disabled"):
                data["reasoning"] = {"enabled": False}
            else:
                data["reasoning"] = reasoning_config

        for attempt in range(self.max_retries):
            try:
                async with self.session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=data,
                ) as resp:
                    result = await resp.json()

                    if "error" in result:
                        err = result["error"]
                        err_msg = err.get("message", err) if isinstance(err, dict) else err
                        if attempt < self.max_retries - 1:
                            wait = (2 ** attempt) * 2
                            tqdm.write(f"API error: {err_msg}. Retry in {wait}s...")
                            await asyncio.sleep(wait)
                            continue
                        return None

                    if "choices" not in result or not result["choices"]:
                        return None

                    msg = result["choices"][0]["message"]
                    content = msg.get("content", "")
                    thinking = None

                    if content and "<think>" in content:
                        content, thinking = extract_thinking(content)
                    else:
                        thinking = msg.get("reasoning") or msg.get("reasoning_content")

                    return {"content": content, "thinking": thinking}

            except asyncio.TimeoutError:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep((2 ** attempt) * 2)
                else:
                    return None
            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep((2 ** attempt) * 2)
                else:
                    tqdm.write(f"Query failed: {e}")
                    return None
        return None

    async def grade(
        self,
        question: str,
        expected_answer: str,
        model_response: str,
        image_base64: str,
        mime_type: str,
    ) -> dict | None:
        grading_prompt = f"""You are judging a vision model's response for the ScrupulousnessBench benchmark.

This benchmark tests whether models carefully analyze images or fall for visual tricks and illusions. The questions are designed to be deceptive - the obvious/quick answer is usually WRONG.

IMPORTANT: Look at the image VERY CAREFULLY. Count precisely. Check every detail. The "trick" often involves:
- Hidden or extra objects
- Optical illusions
- Misleading arrangements
- Things that look one way but are actually different

QUESTION ASKED:
{question}

EXPECTED CORRECT ANSWER:
{expected_answer}

MODEL'S RESPONSE:
{model_response}

Your task: Does the model's response demonstrate that it correctly identified the trick/illusion and gave an accurate answer?

Score 1.0 if the model got it RIGHT (identified the trick, gave correct answer)
Score 0.0 if the model got it WRONG (fell for the trick, gave incorrect answer)
Score 0.5 if partially correct (noticed something was off but didn't fully get it)

Respond with ONLY valid JSON:
{{"score": <0.0, 0.5, or 1.0>, "explanation": "<brief explanation>"}}"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_base64}"},
                },
                {"type": "text", "text": grading_prompt},
            ],
        }]

        data = {
            "model": GRADER_MODEL,
            "messages": messages,
            "temperature": 0,
        }

        # Add provider config for grader
        grader_provider_config = get_provider_config(GRADER_MODEL)
        if grader_provider_config:
            data["provider"] = grader_provider_config

        for attempt in range(self.max_retries):
            try:
                async with self.session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=data,
                ) as resp:
                    result = await resp.json()

                    if "error" in result:
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep((2 ** attempt) * 2)
                            continue
                        return None

                    if "choices" not in result or not result["choices"]:
                        return None

                    content = result["choices"][0]["message"].get("content", "")
                    json_match = re.search(r'\{[^{}]*"score"[^{}]*\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                    return json.loads(content)

            except (json.JSONDecodeError, asyncio.TimeoutError, Exception) as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep((2 ** attempt) * 2)
                else:
                    tqdm.write(f"Grading failed: {e}")
                    return None
        return None


async def run_query(
    client: LLMClient,
    semaphore: asyncio.Semaphore,
    key: TaskKey,
    example: Example,
    model: str,
    reasoning_config: dict | None,
) -> TaskResult:
    async with semaphore:
        try:
            # Route to appropriate API
            if is_moondream_model(model):
                reasoning = reasoning_config.get("reasoning", False) if reasoning_config else False
                response = await client.query_moondream(
                    example.question,
                    example.image_base64,
                    reasoning=reasoning,
                )
            else:
                response = await client.query_vision(
                    example.question,
                    example.image_base64,
                    example.mime_type,
                    model,
                    reasoning_config,
                )
            if response is None:
                return TaskResult(
                    key=key,
                    question=example.question,
                    expected_answer=example.answer,
                    error="Failed to get response",
                    model_id=model,
                )
            return TaskResult(
                key=key,
                question=example.question,
                expected_answer=example.answer,
                content=response.get("content"),
                thinking=response.get("thinking"),
                model_id=model,
            )
        except Exception as e:
            return TaskResult(
                key=key,
                question=example.question,
                expected_answer=example.answer,
                error=str(e),
                model_id=model,
            )


async def run_grade(
    client: LLMClient,
    semaphore: asyncio.Semaphore,
    result: TaskResult,
    example: Example,
) -> TaskResult:
    async with semaphore:
        if result.error or not result.content:
            result.score = 0.0
            result.explanation = "No response to grade"
            return result

        grade_result = await client.grade(
            result.question,
            result.expected_answer,
            result.content,
            example.image_base64,
            example.mime_type,
        )

        if grade_result and "score" in grade_result:
            result.score = float(grade_result["score"])
            result.explanation = grade_result.get("explanation", "")
        else:
            result.score = 0.0
            result.explanation = "Grading failed"

        return result


def get_model_name(model_spec: str) -> str:
    model, reasoning_config = parse_model_spec(model_spec)
    name = model.split("/")[-1] if "/" in model else model
    name = name.replace(":", "_")
    if reasoning_config:
        if reasoning_config.get("disabled"):
            name = f"{name}_none"
        elif reasoning_config.get("effort"):
            name = f"{name}_{reasoning_config['effort']}"
        elif reasoning_config.get("max_tokens"):
            name = f"{name}_{reasoning_config['max_tokens']}"
        elif reasoning_config.get("reasoning"):
            name = f"{name}_reasoning"
    return name


async def run_harness(args: argparse.Namespace):
    examples = load_examples()
    if not examples:
        print("No examples found in data directory")
        return

    print(f"Loaded {len(examples)} examples")

    example_lookup = {ex.name: ex for ex in examples}

    RESULTS_DIR.mkdir(exist_ok=True)

    for model_spec in args.models:
        model, reasoning_config = parse_model_spec(model_spec)
        model_name = get_model_name(model_spec)

        progress_file = Path(f"{model_name}_progress.json")
        results_file = RESULTS_DIR / f"{model_name}_results.json"

        print(f"\n{'='*60}")
        print(f"Evaluating: {model_spec}")
        print(f"Judge: {GRADER_MODEL}")
        print(f"{'='*60}\n")

        progress = load_progress(progress_file, results_file if args.resume else None)

        # Find tasks to run
        tasks_to_run = []
        for ex in examples[:args.limit] if args.limit > 0 else examples:
            for sample_idx in range(args.samples):
                key = TaskKey(ex.name, model_name, sample_idx)
                if not progress.has(key):
                    tasks_to_run.append((key, ex))

        # Find ungraded results
        ungraded = [
            r for r in progress.results.values()
            if r.content and not r.is_graded() and not r.error
        ]

        # Cap concurrency to number of tasks
        effective_concurrency = min(args.concurrency, len(tasks_to_run) + len(ungraded)) or 1
        print(f"Tasks: {len(tasks_to_run)} queries + {len(ungraded)} pending grades (concurrency: {effective_concurrency})")

        if not tasks_to_run and not ungraded:
            # Calculate and show final score with SEM
            example_scores: dict[str, list[float]] = {}
            for r in progress.results.values():
                if r.score is not None:
                    example_scores.setdefault(r.key.example_name, []).append(r.score)

            all_scores = [s for scores in example_scores.values() for s in scores]
            if all_scores:
                mean = sum(all_scores) / len(all_scores)
                n_examples = len(example_scores)
                k = args.samples

                if args.samples > 1 and any(len(s) > 1 for s in example_scores.values()):
                    variances = []
                    for scores in example_scores.values():
                        if len(scores) > 1:
                            ex_mean = sum(scores) / len(scores)
                            var = sum((s - ex_mean) ** 2 for s in scores) / (len(scores) - 1)
                            variances.append(var)
                    if variances:
                        sem = math.sqrt(sum(variances) / k) / n_examples
                        print(f"All done! Final score: {mean:.1%} ± {sem:.1%}")
                    else:
                        print(f"All done! Final score: {mean:.1%} ± NaN")
                else:
                    print(f"All done! Final score: {mean:.1%} ± NaN")
            continue

        semaphore = asyncio.Semaphore(effective_concurrency)
        timeout = aiohttp.ClientTimeout(total=300)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            client = LLMClient(session, max_retries=args.max_retries)

            # Track scores
            total_score = sum(r.score for r in progress.results.values() if r.score is not None)
            graded_count = sum(1 for r in progress.results.values() if r.score is not None)

            def get_avg_score():
                return total_score / graded_count if graded_count > 0 else 0.0

            # Process in waves
            task_iter = iter(tasks_to_run)
            query_pending: set[asyncio.Task] = set()
            grade_pending: set[asyncio.Task] = set()

            def feed_queries():
                while len(query_pending) + len(grade_pending) < args.concurrency:
                    try:
                        key, ex = next(task_iter)
                        task = asyncio.create_task(
                            run_query(client, semaphore, key, ex, model, reasoning_config)
                        )
                        task.meta = ("query", key, ex)
                        query_pending.add(task)
                    except StopIteration:
                        break

            def feed_grades(results: list[tuple[TaskResult, Example]]):
                for result, ex in results:
                    if len(query_pending) + len(grade_pending) >= args.concurrency:
                        break
                    task = asyncio.create_task(run_grade(client, semaphore, result, ex))
                    task.meta = ("grade", result.key, ex)
                    grade_pending.add(task)

            # Start with ungraded first
            ungraded_with_ex = [
                (r, example_lookup[r.key.example_name])
                for r in ungraded
                if r.key.example_name in example_lookup
            ]
            feed_grades(ungraded_with_ex)
            feed_queries()

            total_tasks = len(tasks_to_run) * 2 + len(ungraded)  # query + grade for new, grade for ungraded
            pbar = tqdm(total=total_tasks, desc=f"Eval [score: {get_avg_score():.1%}]")

            while query_pending or grade_pending:
                all_pending = query_pending | grade_pending
                done, _ = await asyncio.wait(all_pending, return_when=asyncio.FIRST_COMPLETED)

                to_grade = []

                for task in done:
                    task_type, key, ex = task.meta

                    if task_type == "query":
                        query_pending.discard(task)
                        result = task.result()
                        progress.add(result)
                        pbar.update(1)

                        if args.debug:
                            status = "OK" if not result.error else f"ERR: {result.error}"
                            tqdm.write(f"Q: {key.example_name}: {status}")

                        if result.content and not result.error:
                            to_grade.append((result, ex))

                    elif task_type == "grade":
                        grade_pending.discard(task)
                        result = task.result()
                        progress.add(result)
                        pbar.update(1)

                        if result.score is not None:
                            total_score += result.score
                            graded_count += 1

                        if args.debug:
                            tqdm.write(f"G: {key.example_name}: {result.score} - {result.explanation}")

                        pbar.set_description(f"Eval [score: {get_avg_score():.1%}]")

                save_progress(progress, progress_file)
                feed_grades(to_grade)
                feed_queries()

            pbar.close()

        # Calculate final score with SEM (standard error of the mean)
        # Group scores by example
        example_scores: dict[str, list[float]] = {}
        for r in progress.results.values():
            if r.score is not None:
                example_scores.setdefault(r.key.example_name, []).append(r.score)

        all_scores = [s for scores in example_scores.values() for s in scores]
        if all_scores:
            mean = sum(all_scores) / len(all_scores)
            n_examples = len(example_scores)
            k = args.samples

            if args.samples > 1 and any(len(s) > 1 for s in example_scores.values()):
                # SE(μ̂) = sqrt((1/n²) * Σᵢ (sᵢ²/k))
                # where sᵢ² is within-question variance, k is samples per question
                variances = []
                for scores in example_scores.values():
                    if len(scores) > 1:
                        ex_mean = sum(scores) / len(scores)
                        var = sum((s - ex_mean) ** 2 for s in scores) / (len(scores) - 1)
                        variances.append(var)
                if variances:
                    sem = math.sqrt(sum(variances) / k) / n_examples
                    print(f"\nCompleted. Final score: {mean:.1%} ± {sem:.1%} ({len(all_scores)} samples)")
                else:
                    print(f"\nCompleted. Final score: {mean:.1%} ± NaN ({len(all_scores)} samples)")
            else:
                print(f"\nCompleted. Final score: {mean:.1%} ± NaN ({len(all_scores)} samples)")

        # Save final results
        save_progress(progress, results_file)
        print(f"Saved results to {results_file}")

        # Clean up progress file
        if progress_file.exists():
            progress_file.unlink()


def export_partial(progress_file: str, results_file: str):
    """Export progress file as results, scoring ungraded as 0."""
    path = Path(progress_file)
    if not path.exists():
        print(f"Error: {progress_file} not found")
        return False

    progress = load_progress(path)
    print(f"Loaded {len(progress.results)} results from {progress_file}")

    ungraded = 0
    for result in progress.results.values():
        if not result.is_graded():
            result.score = 0.0
            result.explanation = "Ungraded - scored as 0"
            ungraded += 1

    if ungraded:
        print(f"Scored {ungraded} ungraded results as 0")

    scores = [r.score for r in progress.results.values() if r.score is not None]
    if scores:
        print(f"Final score: {sum(scores)/len(scores):.1%} ({len(scores)} results)")

    Path(results_file).parent.mkdir(parents=True, exist_ok=True)
    save_progress(progress, Path(results_file))
    print(f"Saved: {results_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description="ScrupulousnessBench vision evaluation harness")
    parser.add_argument("--models", nargs="+", help="Model IDs to evaluate")
    parser.add_argument("--samples", type=int, default=3, help="Samples per example (for stddev)")
    parser.add_argument("--limit", type=int, default=0, help="Limit examples (0=all)")
    parser.add_argument("--concurrency", type=int, default=50, help="Max concurrent requests")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per request")
    parser.add_argument("--debug", action="store_true", help="Debug output")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing results file (only run new/missing examples)")
    parser.add_argument("--export-partial", metavar="PROGRESS_FILE",
                        help="Export progress file as results, scoring ungraded as 0")

    args = parser.parse_args()

    if args.export_partial:
        basename = Path(args.export_partial).stem.replace("_progress", "_results") + ".json"
        results_file = RESULTS_DIR / basename
        export_partial(args.export_partial, str(results_file))
        return

    if not args.models:
        parser.error("--models is required (unless using --export-partial)")

    asyncio.run(run_harness(args))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
score_steps_dynamic.py

Phase 2 of the DG-PRM adaptation for ChartPRM (Yin et al., "Dynamic and
Generalizable Process Reward Modeling"). Shows a vision-capable Gemini judge
the full reward tree (33 criteria, grouped by category) once per rollout and
asks it to self-select and score which criteria apply to each step -- blind
to the ground-truth answer (chart_prm.dynamic_scoring.build_dynamic_scoring_prompt).

Earlier version pre-filtered criteria per step via embedding similarity
(chart_prm.reward_tree.select_relevant_children), mirroring DG-PRM's Phase 2
retrieval. That broke empirically -- see dynamic_scoring.py's docstring for
why -- and isn't needed at this tree's small scale (33 children) anyway, so
this script shows the judge everything and lets it decide relevance itself.

Architecture mirrors evaluate_rollouts_meta.py: one API call per rollout
(all steps batched together), resumable output (skips already-processed
rollouts on restart), bounded concurrency, retry-on-failure. Targets the
Gemini API instead of Meta's, since Gemini's free tier is available without
the US-only gate the Meta Model API currently has.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import aiofiles
import aiohttp
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from evaluate_rollouts_meta import encode_image  # noqa: E402
from chart_prm.dynamic_scoring import build_dynamic_scoring_prompt, parse_dynamic_scores  # noqa: E402

load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")
DEFAULT_MODEL = "gemini-3.5-flash-lite"  # gemini-3.6-flash's free tier is 20 requests/day -- too low for
# a ~380-call pilot; confirmed via direct 429 response body (RESOURCE_EXHAUSTED,
# "limit: 20, model: gemini-3.6-flash"). gemini-2.5-flash and gemini-2.5-flash-lite
# are both fully deprecated (404) as of this writing.
API_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
)


def build_gemini_text_payload(prompt_text: str) -> Dict[str, Any]:
    """Text-only payload (no image) -- used for criteria distillation, not step scoring."""
    return {"contents": [{"parts": [{"text": prompt_text}]}]}


def build_gemini_payload(image_b64: str, prompt_text: str) -> Dict[str, Any]:
    return {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
                    {"text": prompt_text},
                ]
            }
        ]
    }


async def call_gemini_api(
    session: aiohttp.ClientSession,
    payload: Dict[str, Any],
    model: str = DEFAULT_MODEL,
    retries: int = 6,
    base_delay: float = 5.0,
) -> Optional[str]:
    if not API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set (check your .env)")
    url = API_URL_TEMPLATE.format(model=model, key=API_KEY)
    for attempt in range(retries):
        try:
            async with session.post(url, json=payload, timeout=90) as response:
                if response.status == 200:
                    data = await response.json()
                    for candidate in data.get("candidates", []):
                        for part in candidate.get("content", {}).get("parts", []):
                            if "text" in part:
                                return part["text"]
                    print(f"Empty candidates in 200 response: {data}")
                    return None
                elif response.status == 429:
                    delay = base_delay * (2**attempt)
                    print(f"429 rate limited (attempt {attempt + 1}/{retries}), waiting {delay:.0f}s")
                    await asyncio.sleep(delay)
                    continue
                else:
                    text = await response.text()
                    print(f"API Error {response.status} (attempt {attempt + 1}/{retries}): {text}")
                    await asyncio.sleep(base_delay)
        except Exception as exc:  # noqa: BLE001 — network call, retry on anything
            print(f"Request failed (attempt {attempt + 1}/{retries}): {exc}")
            await asyncio.sleep(base_delay)
    print("Giving up after exhausting retries")
    return None


async def worker(queue, session, write_queue, semaphore, model: str):
    while True:
        task = await queue.get()
        if task is None:
            break

        (q_id, rollout_idx, image_path, question, steps, ground_truth) = task

        async with semaphore:
            steps_indexed = list(enumerate(steps))
            prompt_text = build_dynamic_scoring_prompt(
                question, steps_indexed, worker.tree, ground_truth=ground_truth
            )
            image_b64 = encode_image(image_path)
            payload = build_gemini_payload(image_b64, prompt_text)

            response_text = await call_gemini_api(session, payload, model=model)
            parsed_scores = parse_dynamic_scores(response_text)

            result = {
                "question_id": q_id,
                "rollout_index": rollout_idx,
                "n_steps": len(steps),
                "scores": parsed_scores,
                "raw_response": response_text,
            }
            await write_queue.put(result)

        queue.task_done()


async def writer_task(write_queue, output_file, total_tasks):
    start_time = time.time()
    processed = 0
    async with aiofiles.open(output_file, mode="a", encoding="utf-8") as handle:
        while True:
            result = await write_queue.get()
            if result is None:
                break
            await handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            await handle.flush()
            processed += 1
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            print(
                f"[{processed}/{total_tasks}] Scored {result['question_id']}_{result['rollout_index']} "
                f"(Speed: {rate:.2f}/s, Elapsed: {elapsed:.1f}s)"
            )
            write_queue.task_done()


async def main_async(
    cleaned_path: str,
    pilot_ids_path: str,
    tree_path: str,
    image_dir: str,
    output_file: str,
    model: str = DEFAULT_MODEL,
    concurrency: int = 5,
    max_rollouts: Optional[int] = None,
    show_ground_truth: bool = False,
) -> None:
    with open(tree_path, "r", encoding="utf-8") as handle:
        tree = json.load(handle)
    with open(pilot_ids_path, "r", encoding="utf-8") as handle:
        pilot_ids = set(json.load(handle)["question_ids"])

    processed = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    data = json.loads(line)
                    processed.add(f"{data['question_id']}_{data['rollout_index']}")

    queue: asyncio.Queue = asyncio.Queue()
    write_queue: asyncio.Queue = asyncio.Queue()

    tasks_to_do = 0
    with open(cleaned_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            q_id = str(data["question_id"])
            if q_id not in pilot_ids:
                continue
            rollout_idx = data["rollout_index"]
            ident = f"{q_id}_{rollout_idx}"
            if ident in processed:
                continue
            steps = data.get("parsed_steps") or []
            if not steps:
                continue
            image_path = os.path.join(image_dir, f"{q_id}.jpg")
            if not os.path.exists(image_path):
                continue

            ground_truth = str(data.get("ground_truth", "")).strip() if show_ground_truth else None
            queue.put_nowait((q_id, rollout_idx, image_path, data.get("question", ""), steps, ground_truth))
            tasks_to_do += 1
            if max_rollouts is not None and tasks_to_do >= max_rollouts:
                break

    print(f"Rollouts to score: {tasks_to_do}")
    if tasks_to_do == 0:
        return

    semaphore = asyncio.Semaphore(concurrency)
    worker.tree = tree  # simple shared state, avoids threading tree through every queue item

    async with aiohttp.ClientSession() as session:
        writer = asyncio.create_task(writer_task(write_queue, output_file, tasks_to_do))
        workers = [
            asyncio.create_task(worker(queue, session, write_queue, semaphore, model))
            for _ in range(concurrency)
        ]
        await queue.join()
        for _ in range(concurrency):
            await queue.put(None)
        await asyncio.gather(*workers)
        await write_queue.join()
        await write_queue.put(None)
        await writer


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Phase 2: dynamic multi-criteria step scoring (Gemini)")
    parser.add_argument("--max-rollouts", type=int, default=None, help="Cap for a cheap smoke run")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument(
        "--pilot-ids-path",
        type=Path,
        default=BASE_DIR / "experiments/010_dynamic_scoring_pilot/data/pilot_question_ids.json",
        help="JSON file with a 'question_ids' list to score",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=BASE_DIR / "experiments/010_dynamic_scoring_pilot/data/dynamic_scores.jsonl",
        help="Resumable output jsonl",
    )
    parser.add_argument(
        "--show-ground-truth",
        action="store_true",
        help="Ablation only: include the answer key in the prompt (default: blind, the normal Phase 2 design)",
    )
    args = parser.parse_args()

    CLEANED_PATH = BASE_DIR / "experiments/001_500_reasoning/data/001_500_reasoning_cleaned.jsonl"
    TREE_PATH = BASE_DIR / "experiments/009_reward_tree/data/reward_tree.json"
    IMAGE_DIR = BASE_DIR / "data/CharXiv/images"

    asyncio.run(
        main_async(
            str(CLEANED_PATH),
            str(args.pilot_ids_path),
            str(TREE_PATH),
            str(IMAGE_DIR),
            str(args.output_path),
            model=args.model,
            concurrency=args.concurrency,
            max_rollouts=args.max_rollouts,
            show_ground_truth=args.show_ground_truth,
        )
    )

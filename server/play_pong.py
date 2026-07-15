"""FastAPI WebSocket server for human-vs-JEPA Pong (PettingZoo pong_v3).

Unlike server/infer.py -- where the browser owns a synthetic Pong and the server
only perceives -- ALE cannot run in a browser, so here the server owns the env
and the client is a renderer plus a keyboard. Frames flow server -> client;
actions flow client -> server.

All env/model/policy logic lives in server/pong_core.py so it can be driven
headlessly; this module only moves bytes.

Usage:
    python -m server.play_pong --checkpoint checkpoints/lepong_atari_statehead.pt --port 8793
"""
import argparse
import logging
import pathlib

import asyncio
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from server.pong_core import (AGENT_HUMAN, AGENT_JEPA, NOOP, PongSession, png_b64)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pong")

_cfg = argparse.Namespace()

app = FastAPI(title="lepong-pong", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health():
    return {"status": "ok", "game": "pong_v3",
            "jepa_agent": AGENT_JEPA, "human_agent": AGENT_HUMAN,
            "policy": "align-then-move on state_head(predict_next(pixels)).ball_y"}


@app.get("/", response_class=HTMLResponse)
async def index():
    p = pathlib.Path(__file__).parent.parent / "client" / "pong_play.html"
    if not p.exists():
        return HTMLResponse(f"<h1>Client not found at {p}</h1>", status_code=404)
    return HTMLResponse(p.read_text())


@app.websocket("/ws-pong")
async def pong_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("Client connected")

    try:
        session = PongSession(_cfg.checkpoint, torch.device(_cfg.device),
                              frameskip=_cfg.frameskip, img_size=_cfg.img_size,
                              seed=_cfg.seed)
    except Exception as e:
        logger.error("Model load failed: %s", e, exc_info=True)
        await ws.send_json({"error": str(e)})
        await ws.close(code=1011)
        return

    human_action = NOOP

    async def read_actions():
        """Drain client keypresses without blocking the game loop."""
        nonlocal human_action
        while True:
            msg = await ws.receive_json()
            a = int(msg.get("action", NOOP))
            human_action = a if 0 <= a < 6 else NOOP

    reader = asyncio.create_task(read_actions())
    dt = _cfg.frameskip / 60.0            # ALE runs at 60 Hz

    try:
        while True:
            # ALE is synchronous CPU work; keep it off the event loop so the
            # reader task can keep draining keypresses while it runs.
            tel = await asyncio.to_thread(session.step, human_action)
            raw = tel.pop("raw")
            tel["frame_png"] = png_b64(raw, scale=_cfg.display_scale)
            await ws.send_json(tel)
            await asyncio.sleep(dt)

    except WebSocketDisconnect:
        logger.info("Client disconnected after %d ticks", session.tick)
    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)
    finally:
        reader.cancel()
        session.close()


def main():
    global _cfg
    pa = argparse.ArgumentParser()
    pa.add_argument("--checkpoint", required=True)
    pa.add_argument("--port", type=int, default=8793)
    pa.add_argument("--host", default="0.0.0.0")
    pa.add_argument("--frameskip", type=int, default=4,
                    help="Must match the collector's stride (default 4).")
    pa.add_argument("--img-size", type=int, default=128)
    pa.add_argument("--display-scale", type=int, default=2)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    _cfg = pa.parse_args()

    import uvicorn
    logger.info("Pong server on %s:%d  (JEPA=%s left, human=%s right)",
                _cfg.host, _cfg.port, AGENT_JEPA, AGENT_HUMAN)
    uvicorn.run(app, host=_cfg.host, port=_cfg.port, log_level="info")


if __name__ == "__main__":
    main()

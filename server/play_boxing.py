"""FastAPI WebSocket server for human-vs-JEPA Boxing (PettingZoo boxing_v2).

The server owns the ALE env and the client is a renderer plus a keyboard, the
same split server/play_tennis.py uses. All env/model/policy logic lives in
server/boxing_core.py; this module only moves bytes.

JEPA drives `first_0` (white, left) off pixels alone; the human takes `second_0`
via BoxingSession.step's `opponent_override`, replacing the scripted seat.

Usage:
    python -m server.play_boxing --port 8794
    # --own ram swaps JEPA's own (x,y) to proprioception; see BoxingSession.
"""
import argparse
import asyncio
import logging
import pathlib

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from server.boxing_core import (AGENT_JEPA, AGENT_OPPONENT, NOOP, BoxingSession,
                                png_b64)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("boxing")

_cfg = argparse.Namespace()

app = FastAPI(title="lepong-boxing", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health():
    return {"status": "ok", "game": "boxing_v2",
            "jepa_agent": AGENT_JEPA, "human_agent": AGENT_OPPONENT,
            "policy": "chase-then-punch on state_head(predict_next(pixels))"}


@app.get("/", response_class=HTMLResponse)
async def index():
    p = pathlib.Path(__file__).parent.parent / "client" / "boxing.html"
    if not p.exists():
        return HTMLResponse(f"<h1>Client not found at {p}</h1>", status_code=404)
    return HTMLResponse(p.read_text())


@app.websocket("/ws-boxing")
async def boxing_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("Client connected")

    try:
        session = BoxingSession(_cfg.checkpoint, torch.device(_cfg.device),
                                frameskip=_cfg.frameskip, img_size=_cfg.img_size,
                                seed=_cfg.seed, predict_k=_cfg.predict_k,
                                own_source=_cfg.own)
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
            human_action = a if 0 <= a < 18 else NOOP
            if "predict_k" in msg:
                try:
                    session.predict_k = max(1, int(msg["predict_k"]))
                except (TypeError, ValueError):
                    pass

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
    pa.add_argument("--checkpoint", default="checkpoints/boxing_R1.pt")
    pa.add_argument("--port", type=int, default=8794)
    pa.add_argument("--host", default="0.0.0.0")
    pa.add_argument("--frameskip", type=int, default=4,
                    help="Must match the collector's stride (default 4).")
    pa.add_argument("--predict-k", type=int, default=1,
                    help="World-model rollout depth before reading the boxers. "
                         "1 = one-step prediction re-anchored to pixels every "
                         "tick; >1 predicts k steps ahead (error compounds).")
    pa.add_argument("--own", choices=["pred", "ram"], default="pred",
                    help="source of JEPA's own (x,y): state head or RAM")
    pa.add_argument("--img-size", type=int, default=128)
    pa.add_argument("--display-scale", type=int, default=2)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    _cfg = pa.parse_args()

    import uvicorn
    logger.info("Boxing server on %s:%d  (JEPA=%s white/left, human=%s black/right)",
                _cfg.host, _cfg.port, AGENT_JEPA, AGENT_OPPONENT)
    uvicorn.run(app, host=_cfg.host, port=_cfg.port, log_level="info")


if __name__ == "__main__":
    main()

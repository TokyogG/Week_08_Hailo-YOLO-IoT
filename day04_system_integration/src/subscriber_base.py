#!/usr/bin/env python3
from __future__ import annotations

import argparse
import signal
import time
from dataclasses import dataclass
from typing import Callable, Optional

import paho.mqtt.client as mqtt

from .utils import safe_json_loads


@dataclass
class SubConfig:
    host: str = "localhost"
    port: int = 1883
    keepalive: int = 30
    client_id: str = "desktop-subscriber"
    topics: tuple[str, ...] = ("#",)
    qos: int = 0
    verbose: bool = False


class MqttSubscriber:
    """
    Robust-ish MQTT subscriber:
    - connects
    - subscribes to topics
    - calls on_message(topic, payload_dict, raw_bytes)
    - auto-reconnect via paho loop
    """

    def __init__(
        self,
        cfg: SubConfig,
        on_message: Callable[[str, Optional[dict], bytes], None],
    ):
        self.cfg = cfg
        self._on_message_user = on_message
        self._connected = False
        self._stop = False

        self.client = mqtt.Client(client_id=cfg.client_id, clean_session=True)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

        # Reconnect backoff
        self.client.reconnect_delay_set(min_delay=1, max_delay=10)

        signal.signal(signal.SIGINT, self._handle_sigint)

    def _handle_sigint(self, signum, frame):
        print("\n[SYS] SIGINT received — stopping subscriber...")
        self._stop = True

    def _on_connect(self, client, userdata, flags, rc):
        self._connected = (rc == 0)
        if self.cfg.verbose:
            print(f"[MQTT] connected rc={rc}")

        if self._connected:
            for t in self.cfg.topics:
                client.subscribe(t, qos=self.cfg.qos)
                if self.cfg.verbose:
                    print(f"[MQTT] subscribed topic={t} qos={self.cfg.qos}")

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        if self.cfg.verbose:
            print(f"[MQTT] disconnected rc={rc}")

    def _on_message(self, client, userdata, msg):
        raw = msg.payload
        obj = safe_json_loads(raw)
        try:
            self._on_message_user(msg.topic, obj, raw)
        except Exception as e:
            print(f"[SUB] handler error topic={msg.topic}: {e}")

    def run_forever(self) -> int:
        self.client.connect(self.cfg.host, self.cfg.port, self.cfg.keepalive)
        self.client.loop_start()

        # wait for connect (brief)
        t0 = time.time()
        while not self._connected and (time.time() - t0) < 2.0:
            time.sleep(0.02)

        if not self._connected:
            print(f"[MQTT] WARNING: not connected yet to {self.cfg.host}:{self.cfg.port} (will retry)")

        try:
            while not self._stop:
                time.sleep(0.1)
        finally:
            try:
                self.client.loop_stop()
            except Exception:
                pass
            try:
                self.client.disconnect()
            except Exception:
                pass

        return 0


def build_common_argparser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--qos", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    return ap
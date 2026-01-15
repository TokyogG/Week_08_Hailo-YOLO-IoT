#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Optional

import paho.mqtt.client as mqtt


@dataclass
class MqttConfig:
    host: str = "192.168.0.85"
    port: int = 1883
    keepalive: int = 30
    client_id: str = "pi-hailo-yolo"
    username: Optional[str] = None
    password: Optional[str] = None
    tls: bool = False  # keep false for LAN unless you set it up
    verbose: bool = False  # if True, prints publish success


class MqttPublisher:
    def __init__(self, cfg: MqttConfig):
        self.cfg = cfg
        self.client = mqtt.Client(client_id=cfg.client_id, clean_session=True)

        if cfg.username:
            self.client.username_pw_set(cfg.username, cfg.password)

        self._connected = False
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags, rc):
        self._connected = (rc == 0)
        if self.cfg.verbose:
            print(f"[MQTT] on_connect rc={rc} connected={self._connected}")

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        if self.cfg.verbose:
            print(f"[MQTT] on_disconnect rc={rc}")

    def connect(self) -> None:
        # non-blocking network thread
        self.client.connect(self.cfg.host, self.cfg.port, self.cfg.keepalive)
        self.client.loop_start()

        # small wait for connect (don’t hang forever)
        t0 = time.time()
        while not self._connected and (time.time() - t0) < 2.0:
            time.sleep(0.02)

        if not self._connected:
            raise RuntimeError(f"MQTT connect failed: {self.cfg.host}:{self.cfg.port}")

        print(f"[MQTT] connected to {self.cfg.host}:{self.cfg.port}")

    def close(self) -> None:
        try:
            self.client.loop_stop()
        except Exception:
            pass
        try:
            self.client.disconnect()
        except Exception:
            pass

    def publish_json(self, topic: str, payload: dict[str, Any], qos: int = 0, retain: bool = False) -> bool:
        # Guard: are we even connected?
        if not self._connected:
            print(f"[MQTT] WARNING: publish attempted while NOT connected (topic={topic})")
            return False

        try:
            data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        except Exception as e:
            print(f"[MQTT] JSON encode failed for topic={topic}: {e}")
            return False

        info = self.client.publish(topic, data, qos=qos, retain=retain)

        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            print(f"[MQTT] publish failed rc={info.rc} topic={topic}")
            return False

        if self.cfg.verbose:
            print(f"[MQTT] published topic={topic}")

        return True

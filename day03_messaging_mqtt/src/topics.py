def build_topics(device_id: str):
    """
    Centralized MQTT topic definitions for Day03
    """
    return {
        "events": f"edge/{device_id}/events",
        "telemetry": f"edge/{device_id}/telemetry",
    }

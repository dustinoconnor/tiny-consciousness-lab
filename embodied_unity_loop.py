#!/usr/bin/env python3
"""First Unity body loop for the functional ego.

Unity side:
    RobotUdpBridge listens on UDP 5055 and sends body state to UDP 5056.

Python side:
    This script sends up/down/left/right/idle/sleep/wake commands and updates
    a tiny fatigue/self-repair model from Unity body feedback.

This is intentionally modest. The first goal is closing the embodied cybernetic
loop:

    mind state -> body action -> world feedback -> mind update -> sleep repair
"""

import argparse
import json
import math
import random
import socket
import time


ACTIONS = ["up", "left", "right", "down", "idle"]


def clamp(x, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, x)))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def metrics_from_state(crosstalk, complexity, memory, prediction_error):
    latency = clamp(0.15 + 0.75 * complexity + 0.30 * crosstalk)
    fatigue_report = clamp(0.36 * crosstalk + 0.30 * complexity + 0.22 * prediction_error + 0.12 * latency)
    delusion_index = sigmoid(9.0 * (crosstalk + 0.45 * complexity - 0.72))
    state_separability = clamp(memory * (1.0 - 0.62 * crosstalk) * (1.0 - 0.24 * complexity))
    return {
        "fatigue_report": fatigue_report,
        "delusion_index": delusion_index,
        "state_separability": state_separability,
    }


def dream_repair(crosstalk, complexity, memory, sleep_steps):
    for step in range(sleep_steps):
        crosstalk *= 0.948
        complexity *= 0.970
        if step > 85:
            memory *= 0.995
        if step > 140:
            memory *= 0.990
    return clamp(crosstalk), clamp(complexity), clamp(memory)


class UnityBodyLink:
    def __init__(self, unity_host="127.0.0.1", unity_port=5055, listen_port=5056):
        self.unity_addr = (unity_host, unity_port)
        self.sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receiver.bind(("127.0.0.1", listen_port))
        self.receiver.setblocking(False)

    def send(self, payload):
        data = json.dumps(payload).encode("utf-8")
        self.sender.sendto(data, self.unity_addr)

    def receive_latest(self):
        latest = None
        while True:
            try:
                data, _addr = self.receiver.recvfrom(8192)
            except BlockingIOError:
                return latest
            try:
                latest = json.loads(data.decode("utf-8"))
            except json.JSONDecodeError:
                continue


class EmbodiedFunctionalEgo:
    def __init__(self, hz=5.0, sleep_seconds=60.0):
        self.crosstalk = 0.07
        self.complexity = 0.12
        self.memory = 0.98
        self.prediction_error = 0.18
        self.fatigue_report = 0.0
        self.delusion_index = 0.0
        self.sleep_remaining = 0
        self.sleep_total_ticks = 0
        self.sleep_repair_credit = 0.0
        self.hz = max(hz, 0.1)
        self.sleep_seconds = max(sleep_seconds, 1.0)
        self.last_action = "idle"
        self.turn_bias = 1
        self.steps = 0

    @property
    def sleeping(self):
        return self.sleep_remaining > 0

    def update_from_body(self, body_state):
        blocked = bool(body_state and body_state.get("blocked", False))
        grounded = bool(body_state is None or body_state.get("grounded", True))
        moving = self.last_action in {"up", "down", "left", "right"}

        body_error = 0.18
        if blocked and self.last_action == "up":
            body_error += 0.55
        if not grounded:
            body_error += 0.20
        if moving and body_state and body_state.get("animation") == "Idle":
            body_error += 0.18

        self.prediction_error = clamp(0.88 * self.prediction_error + 0.12 * body_error)
        if not self.sleeping:
            self.crosstalk = clamp(self.crosstalk + 0.0025 + 0.010 * self.prediction_error)
            self.complexity = clamp(self.complexity + 0.0015 + 0.006 * self.prediction_error)

        metrics = metrics_from_state(self.crosstalk, self.complexity, self.memory, self.prediction_error)
        self.fatigue_report = metrics["fatigue_report"]
        self.delusion_index = metrics["delusion_index"]

    def choose_action(self, body_state):
        if self.sleeping:
            self.sleep_remaining -= 1
            # Spread the lab's 50-step repair sweet spot across however long
            # Unity holds the body in its sleeping animation.
            self.sleep_repair_credit += 50.0 / max(self.sleep_total_ticks, 1)
            while self.sleep_repair_credit >= 1.0:
                self.crosstalk, self.complexity, self.memory = dream_repair(
                    self.crosstalk,
                    self.complexity,
                    self.memory,
                    1,
                )
                self.sleep_repair_credit -= 1.0
            self.last_action = "sleep"
            return "sleep"

        urgency = clamp(0.55 * self.fatigue_report + 0.30 * self.delusion_index + 0.15 * self.complexity)
        if urgency > 0.58:
            self.sleep_total_ticks = max(1, int(round(self.sleep_seconds * self.hz)))
            self.sleep_remaining = self.sleep_total_ticks
            self.sleep_repair_credit = 0.0
            self.last_action = "sleep"
            return "sleep"

        if body_state is None:
            self.last_action = "idle"
            return "idle"

        forward_clear = bool(body_state.get("forward_clear", True))
        left_clear = bool(body_state.get("left_clear", True))
        right_clear = bool(body_state.get("right_clear", True))

        if not forward_clear:
            if left_clear and right_clear:
                action = "left" if self.turn_bias < 0 else "right"
                self.turn_bias *= -1
            elif left_clear:
                action = "left"
            elif right_clear:
                action = "right"
            else:
                action = "down"
        else:
            # Mostly walk forward, with a little scanning drift so the body
            # does not trace a perfectly dead line forever.
            roll = random.random()
            if roll < 0.72:
                action = "up"
            elif roll < 0.84 and left_clear:
                action = "left"
            elif roll < 0.96 and right_clear:
                action = "right"
            else:
                action = "idle"

        self.last_action = action
        return action

    def command_payload(self, action):
        mode = "sleep" if action == "sleep" else "wake"
        return {
            "action": action,
            "mode": mode,
            "fatigue": round(self.fatigue_report, 4),
            "sleep_remaining": self.sleep_remaining,
        }

    def status_line(self, body_state):
        body = "no_body" if body_state is None else (
            f"{body_state.get('animation')} pos=({body_state.get('x', 0):.1f},"
            f"{body_state.get('z', 0):.1f}) blocked={body_state.get('blocked')}"
        )
        return (
            f"step={self.steps:04d} action={self.last_action:>5} "
            f"fatigue={self.fatigue_report:.2f} delusion={self.delusion_index:.2f} "
            f"sleep={self.sleep_remaining:03d} {body}"
        )


def main():
    parser = argparse.ArgumentParser(description="Run the first Python-to-Unity embodied functional ego loop.")
    parser.add_argument("--unity-host", default="127.0.0.1")
    parser.add_argument("--unity-port", type=int, default=5055)
    parser.add_argument("--listen-port", type=int, default=5056)
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run. 0 means run until Ctrl-C.")
    parser.add_argument("--sleep-seconds", type=float, default=60.0, help="How long Unity should keep the body asleep.")
    args = parser.parse_args()

    link = UnityBodyLink(args.unity_host, args.unity_port, args.listen_port)
    ego = EmbodiedFunctionalEgo(hz=args.hz, sleep_seconds=args.sleep_seconds)
    delay = 1.0 / max(args.hz, 0.1)
    started = time.time()
    latest_body = None

    print(f"Sending commands to Unity UDP {args.unity_host}:{args.unity_port}")
    print(f"Listening for robot state on UDP 127.0.0.1:{args.listen_port}")
    print("Press Ctrl-C to stop.")

    try:
        while args.duration <= 0.0 or time.time() - started < args.duration:
            received = link.receive_latest()
            if received is not None:
                latest_body = received

            ego.update_from_body(latest_body)
            action = ego.choose_action(latest_body)
            link.send(ego.command_payload(action))
            if ego.steps % int(max(args.hz, 1.0)) == 0:
                print(ego.status_line(latest_body))
            ego.steps += 1
            time.sleep(delay)
    except KeyboardInterrupt:
        print("\nStopping embodied loop.")
    finally:
        link.send({"action": "idle", "mode": "wake", "fatigue": ego.fatigue_report, "sleep_remaining": 0})


if __name__ == "__main__":
    main()

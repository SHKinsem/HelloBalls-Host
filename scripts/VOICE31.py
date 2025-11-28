# voice_bot_mac.py
# Offline trigger-word + command recognizer for macOS (VOSK + sounddevice; replies via `say`)
# Requires a trigger word ("Tenni") before every command to avoid false positives.

import argparse
import json
import queue
import sys
import time
import shutil
import subprocess
import platform
from pathlib import Path
from difflib import get_close_matches


import sounddevice as sd
from vosk import Model, KaldiRecognizer

# ---------------- CONFIG DEFAULTS ----------------
DEFAULT_MODEL_PATH = "/home/sunrise/Downloads/VOICE/vosk-model-small-en-us-0.15"

# These will be detected dynamically, but defaults are here as a fallback
SAMPLE_RATE = 16000
BLOCK_SECONDS = 0.20            # ~0.2 s chunks
ACTIVE_EXTENSION_ON_SPEECH = 2.0

# ---------------- COMMANDS ----------------
COMMAND_ALIASES = {
    "fetch ball": [
        "fetch ball", "fetch the ball", "fetch",
    ],
    "ball please": [
        "ball please", "ball", "serve a ball", "give me a ball",
    ],
    "stop": [
        "stop",
    ],
    "switch": [
        "switch", "switch target", "change target",
    ],
}

ACK_OK = "Got it."

# ---------------- TRIGGER WORD(S) ----------------
REQUIRE_TRIGGER = True
TRIGGER_ALIASES = [
    "Tan Ni",          # e.g., "Tenni, fetch ball"
]

# ---------------- DEVICE SELECTION ----------------
INPUT_DEVICE_INDEX = 1 # <-- set this to your mic name or int index, or None
OUTPUT_DEVICE_INDEX = None
PREFERRED_DEVICE_NAMES = ("USBAudio1.0", "USB Audio", "USBAudio","USB Composite Device Mono")

# ---------------- ACTION HANDLERS ----------------
def on_fetch_ball():
    print("[ACTION] Fetching the ball")

def on_ball_please():
    print("[ACTION] Serving one ball")

def on_switch():
    print("[ACTION] Switching target")

def on_stop():
    print("[ACTION] Stopping")

INTENT_MAP = {
    "fetch ball": on_fetch_ball,
    "ball please": on_ball_please,
    "switch": on_switch,
    "stop": on_stop,
}

# ---------------- UTILITIES ----------------
audio_q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        pass
    audio_q.put(bytes(indata))

def normalized_text(vosk_result_json: str) -> str:
    try:
        data = json.loads(vosk_result_json)
        return data.get("text", "").strip().lower()
    except Exception:
        return ""

def build_grammar(phrases):
    items = [p.lower() for p in phrases]
    return json.dumps(items, ensure_ascii=False)

def speak(text: str):
    """Cross-platform TTS: macOS uses `say`, Linux uses `espeak`."""
    system = platform.system().lower()
    if system == "darwin" and shutil.which("say"):
        subprocess.run(["say", "-r", "200", text], check=False)
    elif system == "linux" and shutil.which("espeak"):
        subprocess.run(["espeak", text], check=False)
    else:
        print(f"[SAY] {text}")

def mac_permissions_hint():
    print(
        "[HINT] If you get silence/no input, grant mic permission: "
        "System Settings → Privacy & Security → Microphone → allow Terminal/iTerm/IDE."
    )

def drain_queue(max_chunks=50):
    try:
        for _ in range(max_chunks):
            audio_q.get_nowait()
    except queue.Empty:
        pass

def resolve_input_device():
    if INPUT_DEVICE_INDEX is not None:
        return INPUT_DEVICE_INDEX
    try:
        devs = sd.query_devices()
        for want in PREFERRED_DEVICE_NAMES:
            for idx, d in enumerate(devs):
                if d["max_input_channels"] > 0 and want.lower() in d["name"].lower():
                    return idx
    except Exception:
        pass
    return None

def detect_samplerate(resolved_input_device=None):
    try:
        info = sd.query_devices(resolved_input_device, 'input')
        sr = info.get('default_samplerate', None)
        if sr:
            return int(round(sr))
    except Exception:
        pass
    try:
        info = sd.query_devices(None, 'input')
        sr = info.get('default_samplerate', None)
        if sr:
            return int(round(sr))
    except Exception:
        pass
    return 16000

def compute_blocksize(sr: int) -> int:
    bs = int(max(512, round(sr * BLOCK_SECONDS)))
    if bs % 2:
        bs += 1
    return bs

# ---- Build phrase pools
ALL_COMMAND_PHRASES = set()
CANONICAL_FOR_PHRASE = {}
for canon, alias_list in COMMAND_ALIASES.items():
    for a in alias_list:
        a_l = a.lower()
        ALL_COMMAND_PHRASES.add(a_l)
        CANONICAL_FOR_PHRASE[a_l] = canon

ALL_TRIGGER_PHRASES = set(t.lower() for t in TRIGGER_ALIASES)

TRIGGERED_CMD_PHRASES = set()
CANONICAL_FOR_TRIGGERED = {}

def _combine_trigger_aliases():
    seps = [" ", ", ", ": "]
    for trig in ALL_TRIGGER_PHRASES:
        for cmd_alias in ALL_COMMAND_PHRASES:
            for sep in seps:
                full = f"{trig}{sep}{cmd_alias}"
                TRIGGERED_CMD_PHRASES.add(full)
                CANONICAL_FOR_TRIGGERED[full] = CANONICAL_FOR_PHRASE[cmd_alias]

_combine_trigger_aliases()

def strip_trigger_prefix(text: str):
    t = text.strip().lower()
    for trig in sorted(ALL_TRIGGER_PHRASES, key=len, reverse=True):
        if t.startswith(trig + " "):
            return trig, t[len(trig) + 1:].strip()
        if t.startswith(trig + ","):
            return trig, t[len(trig) + 1:].strip(" ,:")
        if t.startswith(trig + ":"):
            return trig, t[len(trig) + 1:].strip(" ,:")
    return None, t

def fuzzy_to_canonical(text: str, cutoff=0.88):
    if text in CANONICAL_FOR_PHRASE:
        return CANONICAL_FOR_PHRASE[text], 1.0
    matches = get_close_matches(text, list(CANONICAL_FOR_PHRASE.keys()), n=1, cutoff=cutoff)
    if matches:
        best = matches[0]
        return CANONICAL_FOR_PHRASE[best], 0.90
    matches = get_close_matches(text, list(COMMAND_ALIASES.keys()), n=1, cutoff=cutoff)
    if matches:
        return matches[0], 0.88
    return None, 0.0

# ---------------- ARGUMENTS ----------------
def parse_args():
    ap = argparse.ArgumentParser(description="Offline trigger-word + command recognizer (macOS)")
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to unzipped VOSK model directory")
    ap.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    return ap.parse_args()

def list_devices_and_exit():
    print("\n[AUDIO DEVICES]")
    try:
        devs = sd.query_devices()
        for idx, d in enumerate(devs):
            print(f"{idx:>3}: {d['name']}  (in:{d['max_input_channels']}, out:{d['max_output_channels']}, default SR:{d.get('default_samplerate','?')})")
    except Exception as e:
        print(f"[ERROR] Could not list devices: {e}", file=sys.stderr)
    sys.exit(0)

# ---------------- MAIN ----------------
def main():
    args = parse_args()
    if args.list_devices:
        list_devices_and_exit()

    model_dir = Path(args.model)
    if not model_dir.exists():
        print(f"[ERROR] Model not found at: {model_dir}", file=sys.stderr)
        sys.exit(1)

    in_dev = resolve_input_device()
    if in_dev is not None:
        print(f"[INFO] Using input device: {in_dev}")

    sr = detect_samplerate(in_dev)
    bs = compute_blocksize(sr)
    print(f"[INFO] Input sample rate: {sr} Hz | Blocksize: {bs} frames")

    sd.default.channels = 1
    sd.default.samplerate = sr
    try:
        sd.default.latency = ("low", "low")
    except Exception:
        pass
    if in_dev is not None or OUTPUT_DEVICE_INDEX is not None:
        sd.default.device = (in_dev, OUTPUT_DEVICE_INDEX)

    print("[INFO] Loading VOSK model…")
    model = Model(str(model_dir))

    cmd_grammar = build_grammar(sorted(TRIGGERED_CMD_PHRASES))
    rec_cmd = KaldiRecognizer(model, sr, cmd_grammar)

    mac_permissions_hint()
    print(f"[INFO] Ready. Trigger word(s): {sorted(ALL_TRIGGER_PHRASES)}")
    print("[INFO] Say: 'Tenni, <command>'")
    print("[INFO] Commands:", sorted(list(COMMAND_ALIASES.keys())))

    try:
        with sd.RawInputStream(callback=audio_callback, dtype="int16",
                               blocksize=bs, samplerate=sr, channels=1,
                               device=in_dev):
            while True:
                data = audio_q.get()

                if rec_cmd.AcceptWaveform(data):
                    heard_full = normalized_text(rec_cmd.Result())
                    if heard_full:
                        print(f"[HEARD] Raw: {heard_full}")

                        if heard_full in CANONICAL_FOR_TRIGGERED:
                            canonical = CANONICAL_FOR_TRIGGERED[heard_full]
                            confidence = 1.0
                        else:
                            trig, remainder = strip_trigger_prefix(heard_full)
                            if REQUIRE_TRIGGER and not trig:
                                print("[INFO] Ignored: missing trigger.")
                                rec_cmd.Reset()
                                continue
                            canonical, confidence = fuzzy_to_canonical(remainder, cutoff=0.88)

                        if canonical:
                            print(f"[MAP] → {canonical} (conf≈{confidence:.2f})")
                            speak(ACK_OK)
                            try:
                                INTENT_MAP[canonical]()
                            except Exception as e:
                                print(f"[ERROR] While executing action: {e}", file=sys.stderr)
                            rec_cmd.Reset()
                            drain_queue()
                        else:
                            print(f"[WARN] Unrecognized after trigger: {heard_full}")
                    else:
                        pass
                else:
                    partial = normalized_text(rec_cmd.PartialResult())
                    if partial:
                        trig, _ = strip_trigger_prefix(partial)
                        if trig:
                            pass  # keep listening
    except Exception as e:
        print(f"[ERROR] Could not open audio stream: {e}", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Exiting.")

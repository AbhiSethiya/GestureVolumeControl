import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os
import glob
import math
import ctypes

# ─────────────────────────────────────────────────────────────────────────────
# Windows System Volume Control (pycaw)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from pycaw.pycaw import AudioUtilities

    _speakers = AudioUtilities.GetSpeakers()
    system_volume = _speakers.EndpointVolume
    PYCAW_AVAILABLE = True
    print("[OK] pycaw loaded — system volume control enabled")
except Exception as e:
    system_volume = None
    PYCAW_AVAILABLE = False
    print(f"[WARN] pycaw not available: {e} — volume control will be visual only")



def get_system_volume_percent():
    """Get current Windows system volume as 0-100 integer."""
    if PYCAW_AVAILABLE and system_volume:
        try:
            scalar = system_volume.GetMasterVolumeLevelScalar()
            return int(round(scalar * 100))
        except Exception:
            return 50
    return 50


def set_system_volume_percent(vol_percent):
    """Set Windows system volume (0-100)."""
    if PYCAW_AVAILABLE and system_volume:
        try:
            scalar = max(0.0, min(1.0, vol_percent / 100.0))
            system_volume.SetMasterVolumeLevelScalar(scalar, None)
        except Exception:
            pass


def get_system_mute():
    """Get current Windows mute state."""
    if PYCAW_AVAILABLE and system_volume:
        try:
            return bool(system_volume.GetMute())
        except Exception:
            return False
    return False


def set_system_mute(muted):
    """Set Windows mute state."""
    if PYCAW_AVAILABLE and system_volume:
        try:
            system_volume.SetMute(int(muted), None)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe Hand Landmarker setup (Tasks API)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = 'models/hand_landmarker.task'

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
hand_options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.4,
    running_mode=vision.RunningMode.VIDEO
)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

# ─────────────────────────────────────────────────────────────────────────────
# Hand Landmark Indices (same as legacy mp.solutions.hands)
# ─────────────────────────────────────────────────────────────────────────────
WRIST = 0
THUMB_CMC = 1; THUMB_MCP = 2; THUMB_IP = 3; THUMB_TIP = 4
INDEX_MCP = 5; INDEX_PIP = 6; INDEX_DIP = 7; INDEX_TIP = 8
MIDDLE_MCP = 9; MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP = 13; RING_PIP = 14; RING_DIP = 15; RING_TIP = 16
PINKY_MCP = 17; PINKY_PIP = 18; PINKY_DIP = 19; PINKY_TIP = 20

FINGER_TIPS = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIPS = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]

# Hand connections for drawing
HAND_CONNECTIONS = [
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    (WRIST, MIDDLE_MCP), (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    (WRIST, RING_MCP), (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    (WRIST, PINKY_MCP), (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
    (INDEX_MCP, MIDDLE_MCP), (MIDDLE_MCP, RING_MCP), (RING_MCP, PINKY_MCP),
]

# ─────────────────────────────────────────────────────────────────────────────
# Modern Color Palette (BGR format for OpenCV)
# ─────────────────────────────────────────────────────────────────────────────
# Base darks
BG_DARK         = (18, 14, 12)       # near-black
BG_PANEL        = (28, 24, 20)       # dark panel
BG_CARD         = (38, 33, 28)       # card background
BG_CARD_HOVER   = (48, 43, 38)       # lighter card
BORDER_SUBTLE   = (55, 50, 45)       # subtle borders

# Text
TEXT_WHITE      = (245, 245, 245)
TEXT_LIGHT      = (200, 200, 205)
TEXT_MID        = (145, 145, 150)
TEXT_DIM        = (95, 95, 100)

# Neon Accents (BGR)
NEON_CYAN       = (209, 255, 0)      # #00FFD1 in BGR
NEON_CYAN_DIM   = (140, 180, 0)      # dimmer cyan
NEON_PURPLE     = (247, 85, 168)     # #A855F7 in BGR
NEON_GREEN      = (100, 230, 80)     # bright green
NEON_RED        = (70, 70, 240)      # vibrant red
NEON_AMBER      = (50, 190, 255)     # warm amber
NEON_BLUE       = (255, 170, 60)     # electric blue

# Gesture-specific colors (BGR) — primary, glow
GESTURE_COLORS = {
    "PLAY/PAUSE":   (NEON_GREEN,  (120, 255, 100)),
    "VOL UP":       (NEON_CYAN,   (220, 255, 50)),
    "VOL DOWN":     (NEON_BLUE,   (255, 190, 100)),
    "CIRCLE":       (NEON_CYAN,   (220, 255, 50)),
    "MUTE":         (NEON_RED,    (90, 90, 255)),
    "UNMUTE":       (NEON_GREEN,  (120, 255, 100)),
    "NEXT TRACK":   (NEON_AMBER,  (80, 210, 255)),
    "PREV TRACK":   (NEON_PURPLE, (255, 120, 200)),
    "NONE":         (TEXT_DIM,    TEXT_MID),
}

GESTURE_ICONS = {
    "PLAY/PAUSE":  "||  /  >",
    "VOL UP":      "VOL +",
    "VOL DOWN":    "VOL -",
    "CIRCLE":      "CIRCLE",
    "MUTE":        "MUTE",
    "UNMUTE":      "UNMUTE",
    "NEXT TRACK":  "NEXT >>|",
    "PREV TRACK":  "|<< PREV",
    "NONE":        "---",
}

# ─────────────────────────────────────────────────────────────────────────────
# Layout constants — larger, more spacious
# ─────────────────────────────────────────────────────────────────────────────
WEBCAM_W = 640
WEBCAM_H = 480
PANEL_W  = 420
LEFT_W   = WEBCAM_W
TOTAL_W  = LEFT_W + PANEL_W
TOTAL_H  = 680
STATUS_BAR_H = 40
WINDOW_H = TOTAL_H + STATUS_BAR_H

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def draw_filled_rounded_rect(img, pt1, pt2, color, radius=12, alpha=1.0):
    """Draw a filled rectangle with rounded corners, optional alpha blending."""
    x1, y1 = pt1
    x2, y2 = pt2
    h_img, w_img = img.shape[:2]

    # Clamp coordinates
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w_img, x2); y2 = min(h_img, y2)
    if x2 <= x1 or y2 <= y1:
        return

    radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

    if alpha < 1.0:
        overlay = img[y1:y2, x1:x2].copy()
        mask_h, mask_w = y2 - y1, x2 - x1
        rect_img = np.full((mask_h, mask_w, 3), color, dtype=np.uint8)
        mask = np.zeros((mask_h, mask_w), dtype=np.uint8)

        if radius < 1:
            mask[:] = 255
        else:
            cv2.rectangle(mask, (radius, 0), (mask_w - radius, mask_h), 255, -1)
            cv2.rectangle(mask, (0, radius), (mask_w, mask_h - radius), 255, -1)
            cv2.circle(mask, (radius, radius), radius, 255, -1)
            cv2.circle(mask, (mask_w - radius - 1, radius), radius, 255, -1)
            cv2.circle(mask, (radius, mask_h - radius - 1), radius, 255, -1)
            cv2.circle(mask, (mask_w - radius - 1, mask_h - radius - 1), radius, 255, -1)

        mask_3ch = mask[:, :, None] > 0
        blended = cv2.addWeighted(rect_img, alpha, overlay, 1 - alpha, 0)
        np.copyto(img[y1:y2, x1:x2], blended, where=mask_3ch)
    else:
        if radius < 1:
            cv2.rectangle(img, pt1, pt2, color, -1)
            return
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius - 1, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius - 1, y2 - radius - 1), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius - 1), radius, color, -1)


def draw_rounded_rect_border(img, pt1, pt2, color, thickness=1, radius=12):
    """Draw a rounded rectangle border (outline only)."""
    x1, y1 = pt1
    x2, y2 = pt2
    radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if radius < 1:
        cv2.rectangle(img, pt1, pt2, color, thickness)
        return
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)


def draw_gradient_bar(img, x, y, w, h, color_start, color_end, radius=6):
    """Draw a horizontal gradient bar with rounded ends."""
    if w < 2 or h < 2:
        return
    h_img, w_img = img.shape[:2]
    if y + h > h_img or x + w > w_img:
        w = min(w, w_img - x)
        h = min(h, h_img - y)
    if w < 2 or h < 2:
        return

    bar = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(w):
        ratio = i / max(w - 1, 1)
        c = tuple(int(color_start[j] + ratio * (color_end[j] - color_start[j])) for j in range(3))
        bar[:, i] = c

    mask = np.zeros((h, w), dtype=np.uint8)
    r = min(radius, h // 2, w // 2)
    if r > 0:
        cv2.rectangle(mask, (r, 0), (w - r, h), 255, -1)
        cv2.rectangle(mask, (0, r), (w, h - r), 255, -1)
        cv2.circle(mask, (r, r), r, 255, -1)
        cv2.circle(mask, (w - r - 1, r), r, 255, -1)
        cv2.circle(mask, (r, h - r - 1), r, 255, -1)
        cv2.circle(mask, (w - r - 1, h - r - 1), r, 255, -1)
    else:
        mask[:] = 255

    roi = img[y:y+h, x:x+w]
    np.copyto(roi, bar, where=(mask[:, :, None] > 0))


def lerp_color(c1, c2, t):
    """Linearly interpolate between two BGR colors."""
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + t * (c2[i] - c1[i])) for i in range(3))


def put_text_with_bg(img, text, org, font, scale, color, thickness,
                     bg_color=None, padding=6, radius=8):
    """Draw text with an optional rounded background."""
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    if bg_color is not None:
        draw_filled_rounded_rect(
            img,
            (x - padding, y - th - padding),
            (x + tw + padding, y + baseline + padding),
            bg_color,
            radius
        )
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)


def draw_glow_text(img, text, org, font, scale, color, thickness, glow_color=None, glow_size=2):
    """Draw text with a subtle glow effect behind it."""
    if glow_color is None:
        glow_color = tuple(max(0, c // 3) for c in color)
    x, y = org
    # Draw glow layers
    for offset in range(glow_size, 0, -1):
        alpha_color = lerp_color(BG_DARK, glow_color, 0.3 / offset)
        cv2.putText(img, text, (x, y), font, scale, alpha_color, thickness + offset * 2, cv2.LINE_AA)
    # Main text
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)


def draw_pill_badge(img, text, center_x, center_y, color, text_color=None, scale=0.38, padding_x=14, padding_y=6):
    """Draw a pill-shaped badge with text."""
    if text_color is None:
        text_color = TEXT_WHITE
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)
    x1 = center_x - tw // 2 - padding_x
    y1 = center_y - th // 2 - padding_y
    x2 = center_x + tw // 2 + padding_x
    y2 = center_y + th // 2 + padding_y + baseline
    radius = (y2 - y1) // 2
    draw_filled_rounded_rect(img, (x1, y1), (x2, y2), color, radius=radius)
    text_x = center_x - tw // 2
    text_y = center_y + th // 2
    cv2.putText(img, text, (text_x, text_y), font, scale, text_color, 1, cv2.LINE_AA)


def draw_speaker_icon(img, cx, cy, size, color, muted=False):
    """Draw a simple speaker icon."""
    body_w = size // 3
    body_h = size // 2
    cv2.rectangle(img, (cx - size // 2, cy - body_h // 2),
                  (cx - size // 2 + body_w, cy + body_h // 2), color, -1)
    pts = np.array([
        [cx - size // 2 + body_w, cy - body_h // 2],
        [cx - size // 2 + body_w, cy + body_h // 2],
        [cx + size // 4, cy + size // 2],
        [cx + size // 4, cy - size // 2],
    ], np.int32)
    cv2.fillPoly(img, [pts], color)

    if muted:
        cv2.line(img, (cx - size // 2 - 2, cy - size // 2 - 2),
                 (cx + size // 2 + 2, cy + size // 2 + 2), NEON_RED, 2, cv2.LINE_AA)
        cv2.line(img, (cx + size // 2 + 2, cy - size // 2 - 2),
                 (cx - size // 2 - 2, cy + size // 2 + 2), NEON_RED, 2, cv2.LINE_AA)
    else:
        # Sound waves
        for i in range(1, 3):
            wave_r = size // 2 + i * 5
            cv2.ellipse(img, (cx + size // 4, cy), (wave_r, wave_r),
                        0, -40, 40, color, 1, cv2.LINE_AA)


def draw_play_icon(img, cx, cy, size, color):
    """Draw a play triangle icon."""
    pts = np.array([
        [cx - size // 2, cy - size // 2],
        [cx - size // 2, cy + size // 2],
        [cx + size // 2, cy],
    ], np.int32)
    cv2.fillPoly(img, [pts], color)


def draw_pause_icon(img, cx, cy, size, color):
    """Draw a pause icon (two bars)."""
    bar_w = max(3, size // 4)
    gap = max(3, size // 4)
    cv2.rectangle(img, (cx - gap - bar_w, cy - size // 2),
                  (cx - gap, cy + size // 2), color, -1)
    cv2.rectangle(img, (cx + gap, cy - size // 2),
                  (cx + gap + bar_w, cy + size // 2), color, -1)


def draw_hand_landmarks_styled(img, landmarks, w, h):
    """Draw hand landmarks with a styled neon look on a given image region."""
    # Draw connections with neon glow
    for start_idx, end_idx in HAND_CONNECTIONS:
        start = landmarks[start_idx]
        end = landmarks[end_idx]
        x1, y1 = int(start.x * w), int(start.y * h)
        x2, y2 = int(end.x * w), int(end.y * h)
        # Glow layer
        cv2.line(img, (x1, y1), (x2, y2), NEON_CYAN_DIM, 4, cv2.LINE_AA)
        # Main line
        cv2.line(img, (x1, y1), (x2, y2), NEON_CYAN, 2, cv2.LINE_AA)

    # Draw landmark dots
    for idx, lm in enumerate(landmarks):
        cx, cy = int(lm.x * w), int(lm.y * h)
        if idx in [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]:
            cv2.circle(img, (cx, cy), 8, NEON_CYAN_DIM, -1)
            cv2.circle(img, (cx, cy), 5, NEON_GREEN, -1)
            cv2.circle(img, (cx, cy), 8, NEON_CYAN, 1, cv2.LINE_AA)
        else:
            cv2.circle(img, (cx, cy), 4, NEON_CYAN_DIM, -1)
            cv2.circle(img, (cx, cy), 3, NEON_CYAN, -1)


# ─────────────────────────────────────────────────────────────────────────────
# Gesture Recognition
# ─────────────────────────────────────────────────────────────────────────────

def get_fingers_up(landmarks, handedness_label="Right"):
    """
    Determine which fingers are extended.
    landmarks: list of NormalizedLandmark from the Tasks API.
    handedness_label: "Left" or "Right" from MediaPipe handedness result.
        NOTE: MediaPipe labels refer to the *actual* hand (not the mirrored
        webcam view), so "Right" means the user's right hand.
    Returns a list of 5 booleans: [thumb, index, middle, ring, pinky].
    """
    fingers = []

    # Thumb: compare tip x vs IP joint x.
    # After cv2.flip(frame, 1), the image is mirrored.  MediaPipe still
    # reports handedness of the *real* hand.  In the mirrored image:
    #   - A real Right hand appears on the right side, thumb points LEFT
    #     (toward smaller x) when extended.
    #   - A real Left hand appears on the left side, thumb points RIGHT
    #     (toward larger x) when extended.
    # This is independent of whether the palm or back of the hand faces
    # the camera, because MediaPipe's handedness already accounts for that.
    thumb_tip = landmarks[THUMB_TIP]
    thumb_ip = landmarks[THUMB_IP]

    if handedness_label == "Right":
        # Real right hand (shown on right side of mirrored feed).
        # Extended thumb tip has smaller x than IP joint.
        fingers.append(thumb_tip.x < thumb_ip.x)
    else:
        # Real left hand (shown on left side of mirrored feed).
        # Extended thumb tip has larger x than IP joint.
        fingers.append(thumb_tip.x > thumb_ip.x)

    # Other four fingers: tip y < PIP y means extended (y decreases upward).
    # This works regardless of palm vs back-of-hand orientation.
    for tip_id, pip_id in zip(FINGER_TIPS, FINGER_PIPS):
        fingers.append(landmarks[tip_id].y < landmarks[pip_id].y)

    return fingers


# ─── Circular motion detection state ────────────────────────────────────────
# We track the index fingertip position over time and compute the cumulative
# angle swept around the centroid of all tracked points.  When the total
# angle exceeds a threshold (~270°), a circle is detected.  The sign of the
# cumulative angle gives the direction (CW / CCW).
#
# This centroid-angle approach is far more robust than cross-products of
# consecutive displacement vectors because it:
#   - Works with circles of any size or speed
#   - Is tolerant of irregular / wobbly shapes
#   - Doesn't depend on tiny per-frame magnitudes

CIRCLE_HISTORY_LEN = 60          # number of recent positions to keep
CIRCLE_MIN_POINTS = 8            # minimum points needed to evaluate
CIRCLE_ANGLE_THRESHOLD = 2.5     # ~140° in radians (half-circle is enough)
CIRCLE_MIN_RADIUS = 0.015        # reject jitter: min avg distance from centroid
CIRCLE_CONSISTENCY = 0.50        # at least 50% of angle steps same sign

circle_positions = []        # list of (x, y) normalized coords
circle_last_direction = None # "CW" or "CCW" or None
circle_cooldown_time = 0.0   # timestamp of last circle-triggered action
CIRCLE_COOLDOWN = 0.5        # seconds between circle volume steps
circle_grace_frames = 0      # frames since last CIRCLE gesture (grace period)
CIRCLE_GRACE_MAX = 8         # keep positions alive for this many non-CIRCLE frames


def detect_circle_direction(positions):
    """
    Analyze a sequence of (x, y) positions to detect circular motion using
    cumulative angle around the centroid.

    Returns "CW" (clockwise in screen coords / from user perspective → vol up),
            "CCW" (counter-clockwise → vol down),
            or None if no clear circle detected.
    """
    n = len(positions)
    if n < CIRCLE_MIN_POINTS:
        return None

    # Compute centroid of all tracked points
    cx = sum(p[0] for p in positions) / n
    cy = sum(p[1] for p in positions) / n

    # Check minimum radius — reject finger jitter / stationary finger
    avg_radius = sum(math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2)
                     for p in positions) / n
    if avg_radius < CIRCLE_MIN_RADIUS:
        return None

    # Compute angles from centroid to each point, then accumulate
    angles = [math.atan2(p[1] - cy, p[0] - cx) for p in positions]

    cumulative_angle = 0.0
    positive_steps = 0
    negative_steps = 0

    for i in range(1, len(angles)):
        delta = angles[i] - angles[i - 1]
        # Wrap to [-π, π]
        if delta > math.pi:
            delta -= 2 * math.pi
        elif delta < -math.pi:
            delta += 2 * math.pi
        cumulative_angle += delta
        if delta > 0:
            positive_steps += 1
        elif delta < 0:
            negative_steps += 1

    total_steps = positive_steps + negative_steps
    if total_steps == 0:
        return None

    # Check consistency: most angle steps should agree on direction
    dominant_ratio = max(positive_steps, negative_steps) / total_steps
    if dominant_ratio < CIRCLE_CONSISTENCY:
        return None

    abs_angle = abs(cumulative_angle)
    if abs_angle < CIRCLE_ANGLE_THRESHOLD:
        return None

    # In screen coordinates (Y-down):
    #   positive cumulative angle = counter-clockwise in math coords
    #                              = clockwise on screen (CW)
    #   negative = CCW on screen
    if cumulative_angle > 0:
        return "CW"
    else:
        return "CCW"


def classify_gesture(fingers_up, landmarks):
    """
    Classify hand gesture based on which fingers are up.
    Returns gesture name string.

    Gestures:
      Index only (1 finger)        → CIRCLE  (track for volume circle)
      2 fingers (I+M, no thumb)    → MUTE
      3 fingers (I+M+R, no thumb)  → UNMUTE
      Peace Sign (I+M) with thumb  → PLAY/PAUSE  (reserved — see 2-finger mute)
      Thumb Up (thumb only)        → NEXT TRACK
      Thumb Down (thumb only)      → PREV TRACK
    """
    thumb, index, middle, ring, pinky = fingers_up
    total_up = sum(fingers_up)

    # ── 2 fingers: index + middle (no thumb) → MUTE
    if index and middle and not ring and not pinky and not thumb:
        return "MUTE"

    # ── 3 fingers: index + middle + ring (no thumb, no pinky) → UNMUTE
    if index and middle and ring and not pinky and not thumb:
        return "UNMUTE"

    # ── Index finger only (possibly with thumb) → CIRCLE tracking
    if index and not middle and not ring and not pinky:
        return "CIRCLE"

    # ── Thumb only → NEXT or PREV based on thumb vertical direction
    if thumb and not index and not middle and not ring and not pinky:
        thumb_tip_y = landmarks[THUMB_TIP].y
        wrist_y = landmarks[WRIST].y
        if thumb_tip_y < wrist_y:
            return "NEXT TRACK"
        else:
            return "PREV TRACK"

    # ── Open palm (5 fingers) → PLAY/PAUSE
    if total_up == 5:
        return "PLAY/PAUSE"

    return "NONE"


# ─────────────────────────────────────────────────────────────────────────────
# Reliable Media Key Simulation (SendInput API)
# ─────────────────────────────────────────────────────────────────────────────

# Virtual key codes for media keys
VK_MEDIA_PLAY_PAUSE = 0xB3
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1

# SendInput constants
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP       = 0x0002
INPUT_KEYBOARD        = 1


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_ushort),
        ("wParamH", ctypes.c_ushort),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("union", INPUT_UNION),
    ]


def send_media_key(vk_code):
    """Send a media key press/release using the SendInput API.

    SendInput is more reliable than the legacy keybd_event for media keys
    because it inserts input into the same queue the OS uses, and correctly
    handles extended virtual-key codes used by media buttons.
    """
    extra = ctypes.pointer(ctypes.c_ulong(0))
    # Key down
    inp_down = INPUT(
        type=INPUT_KEYBOARD,
        union=INPUT_UNION(ki=KEYBDINPUT(
            wVk=vk_code, wScan=0,
            dwFlags=KEYEVENTF_EXTENDEDKEY,
            time=0, dwExtraInfo=extra
        ))
    )
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp_down), ctypes.sizeof(INPUT))
    time.sleep(0.05)
    # Key up
    inp_up = INPUT(
        type=INPUT_KEYBOARD,
        union=INPUT_UNION(ki=KEYBDINPUT(
            wVk=vk_code, wScan=0,
            dwFlags=KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP,
            time=0, dwExtraInfo=extra
        ))
    )
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp_up), ctypes.sizeof(INPUT))


# ─────────────────────────────────────────────────────────────────────────────
# Media Player State
# ─────────────────────────────────────────────────────────────────────────────

class SystemMediaController:
    """Controls system media instead of local files."""

    def __init__(self, media_dir="sample"):
        self.volume = get_system_volume_percent()
        self.is_muted = get_system_mute()
        self.is_playing = False
        self.duration_sec = 0.0
        self.current_time_sec = 0.0
        self.video_cap = None  # compatibility
        self.playlist = []  # compatibility

    def get_track_name(self):
        """Get current track filename."""
        return "System Media"

    def read_frame(self):
        """No video frame for system media."""
        return None

    def toggle_play_pause(self):
        """Toggle between play and pause using SendInput."""
        send_media_key(VK_MEDIA_PLAY_PAUSE)
        self.is_playing = not self.is_playing
        return self.is_playing

    def set_volume(self, vol_percent):
        """Set volume to an absolute level (0-100). Controls REAL system volume."""
        self.volume = max(0, min(100, int(vol_percent)))
        if self.is_muted and self.volume > 0:
            self.is_muted = False
            set_system_mute(False)
        set_system_volume_percent(self.volume)

    def change_volume(self, delta):
        """Change volume by delta. Clamps to 0-100. Controls REAL system volume."""
        self.set_volume(self.volume + delta)

    def toggle_mute(self):
        """Toggle mute state. Controls REAL system mute."""
        self.is_muted = not self.is_muted
        set_system_mute(self.is_muted)

    def next_track(self):
        """Skip to next track using SendInput."""
        send_media_key(VK_MEDIA_NEXT_TRACK)

    def prev_track(self):
        """Skip to previous track using SendInput."""
        send_media_key(VK_MEDIA_PREV_TRACK)

    def get_progress(self):
        """No progress available for system media."""
        return 0.0

    def format_time(self, seconds):
        """Format seconds to MM:SS."""
        return "--:--"


# ─────────────────────────────────────────────────────────────────────────────
# Initialize
# ─────────────────────────────────────────────────────────────────────────────
WINDOW_NAME = 'Gesture Media Player'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, TOTAL_W, WINDOW_H)

# Webcam
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_W)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_H)

# Media player
player = SystemMediaController()

# Gesture state
current_gesture = "NONE"
last_gesture = "NONE"
gesture_cooldown = 1.0        # seconds between gesture triggers
last_gesture_time = 0.0
gesture_hold_frames = 0       # how many frames the same gesture has been held
GESTURE_CONFIRM_FRAMES = 4   # require N consecutive frames to confirm gesture

# Volume step per circle detection
VOLUME_STEP = 5  # percent per circle action

# Action history log (last 8 actions)
action_log = []
MAX_LOG_SIZE = 8

# FPS tracking
fps_timer = time.time()
fps_count = 0
display_fps = 0.0

# Smooth volume display
smooth_volume = float(player.volume)
SMOOTH_FACTOR = 0.15

# Animation pulse for active gesture
gesture_pulse = 0.0

# Frame timestamp for MediaPipe VIDEO mode (needs monotonic ms timestamps)
frame_timestamp_ms = 0


def log_action(gesture_name):
    """Add an action to the log with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    action_log.insert(0, (timestamp, gesture_name))
    if len(action_log) > MAX_LOG_SIZE:
        action_log.pop()


def execute_gesture(gesture, player_obj):
    """Execute the media player action for a recognized gesture."""
    if gesture == "PLAY/PAUSE":
        state = player_obj.toggle_play_pause()
        log_action("PLAY" if state else "PAUSE")
    elif gesture == "MUTE":
        if not player_obj.is_muted:
            player_obj.toggle_mute()
            log_action("MUTED")
    elif gesture == "UNMUTE":
        if player_obj.is_muted:
            player_obj.toggle_mute()
            log_action("UNMUTED")
    elif gesture == "NEXT TRACK":
        player_obj.next_track()
        log_action("NEXT TRACK")
    elif gesture == "PREV TRACK":
        player_obj.prev_track()
        log_action("PREV TRACK")


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

while True:
    try:
        ret, webcam_frame = webcam.read()
        if not ret or webcam_frame is None:
            break

        webcam_frame = cv2.resize(webcam_frame, (WEBCAM_W, WEBCAM_H))
        webcam_frame = cv2.flip(webcam_frame, 1)  # Mirror for natural interaction

        # ── MediaPipe hand detection (Tasks API) ─────────────────────────
        rgb_frame = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        frame_timestamp_ms += 33  # ~30fps monotonic timestamp
        results = hand_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        detected_gesture = "NONE"
        hand_detected = False

        if results.hand_landmarks and len(results.hand_landmarks) > 0:
            hand_detected = True
            landmarks = results.hand_landmarks[0]  # First hand

            # Get handedness label from MediaPipe ("Left" or "Right")
            handedness_label = "Right"  # default fallback
            if results.handedness and len(results.handedness) > 0:
                top_category = results.handedness[0][0]
                handedness_label = top_category.category_name  # "Left" or "Right"

            # Draw styled hand landmarks on webcam frame
            draw_hand_landmarks_styled(webcam_frame, landmarks, WEBCAM_W, WEBCAM_H)

            # Classify gesture
            fingers_up = get_fingers_up(landmarks, handedness_label)
            detected_gesture = classify_gesture(fingers_up, landmarks)

            # Display finger state near wrist
            wrist_lm = landmarks[WRIST]
            wx, wy = int(wrist_lm.x * WEBCAM_W), int(wrist_lm.y * WEBCAM_H)
            finger_names = ["T", "I", "M", "R", "P"]
            finger_str = " ".join(
                f"{n}" for n, up in zip(finger_names, fingers_up) if up
            )
            if finger_str:
                put_text_with_bg(
                    webcam_frame, finger_str,
                    (wx - 30, wy + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, NEON_GREEN, 1,
                    bg_color=(20, 20, 20), padding=6, radius=8
                )

        # ── Gesture confirmation & cooldown ──────────────────────────────
        if detected_gesture == "CIRCLE" and hand_detected:
            # ── Track index fingertip for circular motion ─────────────
            idx_tip = landmarks[INDEX_TIP]
            circle_positions.append((idx_tip.x, idx_tip.y))
            circle_grace_frames = 0  # reset grace counter
            # Keep history bounded
            if len(circle_positions) > CIRCLE_HISTORY_LEN:
                circle_positions.pop(0)

            # Check for circle direction
            direction = detect_circle_direction(circle_positions)
            now = time.time()

            if direction is not None and now - circle_cooldown_time >= CIRCLE_COOLDOWN:
                if direction == "CW":
                    # Clockwise (circle right) → Volume Up
                    player.change_volume(VOLUME_STEP)
                    current_gesture = "VOL UP"
                    log_action(f"VOL UP -> {player.volume}%")
                else:
                    # Counter-clockwise (circle left) → Volume Down
                    player.change_volume(-VOLUME_STEP)
                    current_gesture = "VOL DOWN"
                    log_action(f"VOL DOWN -> {player.volume}%")
                circle_cooldown_time = now
                last_gesture_time = now
                # Clear some history to prevent repeat triggers
                circle_positions = circle_positions[-5:]
            else:
                current_gesture = "CIRCLE"

            last_gesture = "CIRCLE"
            gesture_hold_frames = 0

        elif detected_gesture != "NONE":
            # Don't immediately clear circle positions — allow brief
            # flickers (e.g. a second finger briefly detected as up)
            circle_grace_frames += 1
            if circle_grace_frames > CIRCLE_GRACE_MAX:
                circle_positions.clear()

            if detected_gesture == last_gesture:
                gesture_hold_frames += 1
            else:
                gesture_hold_frames = 1
                last_gesture = detected_gesture

            now = time.time()
            if (gesture_hold_frames >= GESTURE_CONFIRM_FRAMES and
                    now - last_gesture_time >= gesture_cooldown):
                current_gesture = detected_gesture
                execute_gesture(detected_gesture, player)
                last_gesture_time = now
                gesture_hold_frames = 0
        else:
            gesture_hold_frames = 0
            last_gesture = "NONE"
            # Grace period: don't immediately clear circle positions
            # when hand briefly loses detection or gesture flickers
            circle_grace_frames += 1
            if circle_grace_frames > CIRCLE_GRACE_MAX:
                circle_positions.clear()

        # Fade gesture display after cooldown
        time_since_gesture = time.time() - last_gesture_time
        if time_since_gesture > 2.0:
            current_gesture = "NONE"

        # Pulse animation for gesture badge
        gesture_pulse = (gesture_pulse + 0.08) % (2 * math.pi)
        pulse_alpha = 0.7 + 0.3 * math.sin(gesture_pulse)

        # ── Gesture badge on webcam feed ─────────────────────────────────
        if current_gesture != "NONE":
            g_color = GESTURE_COLORS.get(current_gesture, (TEXT_DIM, TEXT_MID))[0]
            g_glow = GESTURE_COLORS.get(current_gesture, (TEXT_DIM, TEXT_MID))[1]
            badge_text = f"  {GESTURE_ICONS[current_gesture]}  "
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(badge_text, font, 0.7, 2)
            bx = WEBCAM_W // 2 - tw // 2
            by = 45
            # Glow background
            draw_filled_rounded_rect(webcam_frame,
                                     (bx - 16, by - th - 14),
                                     (bx + tw + 16, by + 14),
                                     g_color, radius=14, alpha=0.85)
            cv2.putText(webcam_frame, badge_text, (bx, by),
                        font, 0.7, TEXT_WHITE, 2, cv2.LINE_AA)

            # ── Vertical volume bar on webcam (when controlling volume) ──
            if current_gesture in ("VOL UP", "VOL DOWN", "CIRCLE"):
                vbar_x = WEBCAM_W - 50       # right side of webcam
                vbar_y = 70                   # top margin
                vbar_h = WEBCAM_H - 140       # bar height
                vbar_w = 24                   # bar width
                vol_pct = player.volume / 100.0
                fill_h = int(vol_pct * vbar_h)

                # Track background
                draw_filled_rounded_rect(
                    webcam_frame,
                    (vbar_x, vbar_y),
                    (vbar_x + vbar_w, vbar_y + vbar_h),
                    (30, 28, 25), radius=vbar_w // 2, alpha=0.6
                )
                # Filled portion (from bottom up)
                if fill_h > 2:
                    fill_top = vbar_y + vbar_h - fill_h
                    draw_gradient_bar(
                        webcam_frame,
                        vbar_x, fill_top, vbar_w, fill_h,
                        NEON_CYAN_DIM, NEON_CYAN,
                        radius=vbar_w // 2
                    )
                # Volume percentage label
                vol_label = f"{player.volume}%"
                (vlw, vlh), _ = cv2.getTextSize(
                    vol_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                put_text_with_bg(
                    webcam_frame, vol_label,
                    (vbar_x + vbar_w // 2 - vlw // 2,
                     vbar_y + vbar_h + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, NEON_CYAN, 1,
                    bg_color=(20, 18, 15), padding=6, radius=8
                )

                # Draw circle trail on webcam for visual feedback
                if len(circle_positions) > 2:
                    pts = [(int(p[0] * WEBCAM_W), int(p[1] * WEBCAM_H))
                           for p in circle_positions]
                    for j in range(1, len(pts)):
                        alpha_line = j / len(pts)
                        line_color = lerp_color(NEON_CYAN_DIM, NEON_CYAN, alpha_line)
                        cv2.line(webcam_frame, pts[j - 1], pts[j],
                                 line_color, 2, cv2.LINE_AA)
        else:
            if not hand_detected:
                msg = "Show hand to control"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(msg, font, 0.5, 1)
                bx = WEBCAM_W // 2 - tw // 2
                put_text_with_bg(
                    webcam_frame, msg,
                    (bx, 42),
                    font, 0.5, TEXT_MID, 1,
                    bg_color=(30, 28, 25), padding=10, radius=10
                )

        # ── Read media frame ─────────────────────────────────────────────
        media_frame = player.read_frame()

        # ── Build canvas ─────────────────────────────────────────────────
        canvas = np.full((WINDOW_H, TOTAL_W, 3), BG_DARK, dtype=np.uint8)

        # ── LEFT SIDE ────────────────────────────────────────────────────
        # Video area (top portion)
        video_h = 380
        webcam_area_h = TOTAL_H - video_h - 28  # 28px for progress area

        if media_frame is not None:
            media_resized = cv2.resize(media_frame, (LEFT_W, video_h))
            # Slight cinematic grading
            effective_vol = 0 if player.is_muted else player.volume
            brightness = 0.7 + 0.3 * (effective_vol / 100.0)
            media_resized = cv2.convertScaleAbs(media_resized, alpha=brightness, beta=5)
            canvas[0:video_h, 0:LEFT_W] = media_resized

            # Track name overlay (bottom-left of video)
            track_name = player.get_track_name()
            if len(track_name) > 40:
                track_name = track_name[:37] + "..."
            put_text_with_bg(
                canvas, f" {track_name} ",
                (16, video_h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT_WHITE, 1,
                bg_color=(15, 12, 10), padding=8, radius=8
            )

            # Play/pause icon overlay (center of video, only when paused)
            if not player.is_playing:
                icon_cx, icon_cy = LEFT_W // 2, video_h // 2
                # Semi-transparent circle behind icon
                overlay = canvas[icon_cy - 30:icon_cy + 30, icon_cx - 30:icon_cx + 30].copy()
                circle_bg = np.zeros_like(overlay)
                cv2.circle(circle_bg, (30, 30), 28, (40, 40, 40), -1)
                cv2.addWeighted(circle_bg, 0.6, overlay, 0.4, 0, overlay)
                canvas[icon_cy - 30:icon_cy + 30, icon_cx - 30:icon_cx + 30] = overlay
                draw_play_icon(canvas, icon_cx, icon_cy, 24, TEXT_WHITE)
        else:
            # No media placeholder
            draw_filled_rounded_rect(canvas, (20, 20), (LEFT_W - 20, video_h - 20),
                                     BG_CARD, radius=16)
            msg = "System Media Controller Active"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(msg, font, 0.7, 1)
            cv2.putText(canvas, msg,
                        (LEFT_W // 2 - tw // 2, video_h // 2 - 20),
                        font, 0.7, NEON_CYAN, 1, cv2.LINE_AA)
            sub_msg = "Control Spotify, YouTube, Netflix"
            (tw2, th2), _ = cv2.getTextSize(sub_msg, font, 0.5, 1)
            cv2.putText(canvas, sub_msg,
                        (LEFT_W // 2 - tw2 // 2, video_h // 2 + 20),
                        font, 0.5, TEXT_DIM, 1, cv2.LINE_AA)

        # ── Progress bar ─────────────────────────────────────────────────
        prog_area_y = video_h + 2
        prog_bar_y = prog_area_y + 4
        prog_bar_h = 6
        prog_margin = 16
        prog_bar_w = LEFT_W - 2 * prog_margin
        progress = player.get_progress()

        # Track background
        draw_filled_rounded_rect(canvas,
                                 (prog_margin, prog_bar_y),
                                 (prog_margin + prog_bar_w, prog_bar_y + prog_bar_h),
                                 BG_CARD, radius=prog_bar_h // 2)

        # Filled progress
        prog_fill_w = max(2, int(progress * prog_bar_w))
        if prog_fill_w > 2:
            draw_gradient_bar(canvas, prog_margin, prog_bar_y, prog_fill_w, prog_bar_h,
                              NEON_CYAN_DIM, NEON_CYAN, radius=prog_bar_h // 2)

        # Progress knob
        knob_x = prog_margin + max(4, min(prog_fill_w, prog_bar_w - 4))
        knob_cy = prog_bar_y + prog_bar_h // 2
        cv2.circle(canvas, (knob_x, knob_cy), 8, NEON_CYAN_DIM, -1)
        cv2.circle(canvas, (knob_x, knob_cy), 5, NEON_CYAN, -1)
        cv2.circle(canvas, (knob_x, knob_cy), 8, TEXT_WHITE, 1, cv2.LINE_AA)

        # Time labels
        time_y = prog_bar_y + prog_bar_h + 16
        cv2.putText(canvas, player.format_time(player.current_time_sec),
                    (prog_margin, time_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_LIGHT, 1, cv2.LINE_AA)
        dur_text = player.format_time(player.duration_sec)
        (tw, _), _ = cv2.getTextSize(dur_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(canvas, dur_text,
                    (prog_margin + prog_bar_w - tw, time_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_DIM, 1, cv2.LINE_AA)

        # ── Webcam feed (bottom-left) ────────────────────────────────────
        webcam_y_start = video_h + 28
        webcam_actual_h = min(webcam_area_h, TOTAL_H - webcam_y_start)

        if webcam_actual_h > 40:
            webcam_small = cv2.resize(webcam_frame, (LEFT_W, webcam_actual_h))

            # Apply subtle vignette
            rows, cols = webcam_small.shape[:2]
            X = cv2.getGaussianKernel(cols, cols * 0.8)
            Y = cv2.getGaussianKernel(rows, rows * 0.8)
            M = Y * X.T
            M = M / M.max()
            for i in range(3):
                webcam_small[:, :, i] = (webcam_small[:, :, i] * M).astype(np.uint8)

            canvas[webcam_y_start:webcam_y_start + webcam_actual_h, 0:LEFT_W] = webcam_small

            # "GESTURE CAM" label
            draw_pill_badge(canvas, "GESTURE CAM", 75, webcam_y_start + 18, NEON_CYAN_DIM,
                            TEXT_WHITE, scale=0.32, padding_x=10, padding_y=4)

            # Subtle border around webcam
            draw_rounded_rect_border(canvas,
                                     (2, webcam_y_start + 1),
                                     (LEFT_W - 2, webcam_y_start + webcam_actual_h - 1),
                                     BORDER_SUBTLE, 1, radius=4)

        # ── Separator ────────────────────────────────────────────────────
        # Gradient separator line
        for row in range(TOTAL_H):
            t = row / TOTAL_H
            sep_color = lerp_color(NEON_CYAN_DIM, BG_DARK, t)
            canvas[row, LEFT_W] = sep_color
            canvas[row, LEFT_W + 1] = lerp_color(sep_color, BG_DARK, 0.5)

        # ── RIGHT PANEL ──────────────────────────────────────────────────
        px = LEFT_W + 2  # panel content x start
        pw = PANEL_W - 4  # panel content width

        # Panel gradient background
        for row in range(TOTAL_H):
            t = row / TOTAL_H
            c = lerp_color((32, 28, 22), (18, 15, 12), t)
            canvas[row, px:px + pw + 2] = c

        # ── HEADER ───────────────────────────────────────────────────────
        py = 20
        draw_glow_text(canvas, "GESTURE MEDIA",
                       (px + 20, py + 24),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.62, NEON_CYAN, 2,
                       glow_color=NEON_CYAN_DIM, glow_size=1)
        cv2.putText(canvas, "PLAYER",
                    (px + 280, py + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, TEXT_LIGHT, 1, cv2.LINE_AA)

        # Gesture status
        g_label_color = GESTURE_COLORS.get(current_gesture, (TEXT_DIM,))[0]
        g_text = f"Gesture: {current_gesture}"
        cv2.putText(canvas, g_text,
                    (px + 20, py + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, g_label_color, 1, cv2.LINE_AA)

        # Thin accent line
        cv2.line(canvas, (px + 16, py + 58), (px + pw - 16, py + 58), BORDER_SUBTLE, 1)

        # ── NOW PLAYING CARD ─────────────────────────────────────────────
        card_x = px + 12
        card_w = pw - 24
        np_card_y = py + 68
        np_card_h = 100

        draw_filled_rounded_rect(canvas,
                                 (card_x, np_card_y),
                                 (card_x + card_w, np_card_y + np_card_h),
                                 BG_CARD, radius=12, alpha=0.85)
        draw_rounded_rect_border(canvas,
                                 (card_x, np_card_y),
                                 (card_x + card_w, np_card_y + np_card_h),
                                 BORDER_SUBTLE, 1, radius=12)

        # Section label
        cv2.putText(canvas, "NOW PLAYING",
                    (card_x + 14, np_card_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, NEON_CYAN, 1, cv2.LINE_AA)

        # Track name
        track_name = player.get_track_name()
        if len(track_name) > 30:
            track_name = track_name[:27] + "..."
        cv2.putText(canvas, track_name,
                    (card_x + 14, np_card_y + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT_WHITE, 1, cv2.LINE_AA)

        # Time + status
        track_info = f"{player.format_time(player.current_time_sec)} / {player.format_time(player.duration_sec)}"
        cv2.putText(canvas, track_info,
                    (card_x + 14, np_card_y + 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_MID, 1, cv2.LINE_AA)

        # Status pill
        status_text = "PLAYING" if player.is_playing else "PAUSED"
        status_color = NEON_GREEN if player.is_playing else NEON_RED
        draw_pill_badge(canvas, status_text, card_x + card_w - 50, np_card_y + 60,
                        status_color, TEXT_WHITE, scale=0.32,
                        padding_x=10, padding_y=4)

        # Track number
        if player.playlist:
            pl_text = f"Track {player.current_index + 1} of {len(player.playlist)}"
            cv2.putText(canvas, pl_text,
                        (card_x + 14, np_card_y + 86),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, TEXT_DIM, 1, cv2.LINE_AA)

        # ── VOLUME CARD ──────────────────────────────────────────────────
        vol_card_y = np_card_y + np_card_h + 10
        vol_card_h = 72

        draw_filled_rounded_rect(canvas,
                                 (card_x, vol_card_y),
                                 (card_x + card_w, vol_card_y + vol_card_h),
                                 BG_CARD, radius=12, alpha=0.85)
        draw_rounded_rect_border(canvas,
                                 (card_x, vol_card_y),
                                 (card_x + card_w, vol_card_y + vol_card_h),
                                 BORDER_SUBTLE, 1, radius=12)

        cv2.putText(canvas, "VOLUME",
                    (card_x + 14, vol_card_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, NEON_CYAN, 1, cv2.LINE_AA)

        # Volume label (SYSTEM)
        cv2.putText(canvas, "SYSTEM",
                    (card_x + 80, vol_card_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, TEXT_DIM, 1, cv2.LINE_AA)

        # Smooth volume animation
        smooth_volume += SMOOTH_FACTOR * (player.volume - smooth_volume)

        # Speaker icon
        speaker_cx = card_x + 28
        speaker_cy = vol_card_y + 48
        draw_speaker_icon(canvas, speaker_cx, speaker_cy, 16,
                          TEXT_LIGHT if not player.is_muted else NEON_RED,
                          muted=player.is_muted)

        # Volume bar
        vol_bar_x = card_x + 52
        vol_bar_y = vol_card_y + 40
        vol_bar_w = card_w - 110
        vol_bar_h = 14

        draw_filled_rounded_rect(canvas,
                                 (vol_bar_x, vol_bar_y),
                                 (vol_bar_x + vol_bar_w, vol_bar_y + vol_bar_h),
                                 (25, 22, 18), radius=vol_bar_h // 2)

        effective_vol = 0 if player.is_muted else smooth_volume
        vol_fill_w = max(2, int((effective_vol / 100.0) * vol_bar_w))

        if not player.is_muted and vol_fill_w > 2:
            if player.volume > 70:
                vol_c1, vol_c2 = (50, 190, 255), (80, 210, 255)   # amber for high
            elif player.volume > 30:
                vol_c1, vol_c2 = NEON_CYAN_DIM, NEON_CYAN          # cyan for mid
            else:
                vol_c1, vol_c2 = (180, 100, 60), (220, 140, 80)   # blue for low
            draw_gradient_bar(canvas, vol_bar_x, vol_bar_y, vol_fill_w, vol_bar_h,
                              vol_c1, vol_c2, radius=vol_bar_h // 2)

        # Volume knob
        vol_knob_x = vol_bar_x + max(4, min(vol_fill_w, vol_bar_w - 4))
        vol_knob_cy = vol_bar_y + vol_bar_h // 2
        if not player.is_muted:
            cv2.circle(canvas, (vol_knob_x, vol_knob_cy), 7, NEON_CYAN_DIM, -1)
            cv2.circle(canvas, (vol_knob_x, vol_knob_cy), 4, NEON_CYAN, -1)
            cv2.circle(canvas, (vol_knob_x, vol_knob_cy), 7, TEXT_WHITE, 1, cv2.LINE_AA)

        # Volume percentage
        vol_text = "MUTED" if player.is_muted else f"{player.volume}%"
        vol_text_color = NEON_RED if player.is_muted else TEXT_WHITE
        cv2.putText(canvas, vol_text,
                    (vol_bar_x + vol_bar_w + 10, vol_bar_y + vol_bar_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, vol_text_color, 1, cv2.LINE_AA)

        # ── GESTURE GUIDE CARD ───────────────────────────────────────────
        guide_card_y = vol_card_y + vol_card_h + 10
        guide_card_h = 185

        draw_filled_rounded_rect(canvas,
                                 (card_x, guide_card_y),
                                 (card_x + card_w, guide_card_y + guide_card_h),
                                 BG_CARD, radius=12, alpha=0.85)
        draw_rounded_rect_border(canvas,
                                 (card_x, guide_card_y),
                                 (card_x + card_w, guide_card_y + guide_card_h),
                                 BORDER_SUBTLE, 1, radius=12)

        cv2.putText(canvas, "GESTURE GUIDE",
                    (card_x + 14, guide_card_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, NEON_CYAN, 1, cv2.LINE_AA)

        gestures_guide = [
            ("Circle Right", "Vol Up",      NEON_CYAN),
            ("Circle Left",  "Vol Down",    NEON_BLUE),
            ("2 Fingers",    "Mute",        NEON_RED),
            ("3 Fingers",    "Unmute",      NEON_GREEN),
            ("Open Palm",    "Play/Pause",  NEON_GREEN),
            ("Thumb Up",     "Next Track",  NEON_AMBER),
            ("Thumb Down",   "Prev Track",  NEON_PURPLE),
        ]

        for i, (gesture_name, action_name, color) in enumerate(gestures_guide):
            gy = guide_card_y + 38 + i * 18
            # Color dot
            cv2.circle(canvas, (card_x + 20, gy - 4), 3, color, -1)
            cv2.putText(canvas, gesture_name,
                        (card_x + 30, gy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, TEXT_LIGHT, 1, cv2.LINE_AA)
            cv2.putText(canvas, action_name,
                        (card_x + 170, gy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, TEXT_DIM, 1, cv2.LINE_AA)

        # ── ACTION LOG CARD ──────────────────────────────────────────────
        log_card_y = guide_card_y + guide_card_h + 10
        log_card_h = TOTAL_H - log_card_y - 10

        if log_card_h > 40:
            draw_filled_rounded_rect(canvas,
                                     (card_x, log_card_y),
                                     (card_x + card_w, log_card_y + log_card_h),
                                     BG_CARD, radius=12, alpha=0.85)
            draw_rounded_rect_border(canvas,
                                     (card_x, log_card_y),
                                     (card_x + card_w, log_card_y + log_card_h),
                                     BORDER_SUBTLE, 1, radius=12)

            cv2.putText(canvas, "ACTION LOG",
                        (card_x + 14, log_card_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, NEON_CYAN, 1, cv2.LINE_AA)

            max_entries = max(1, (log_card_h - 35) // 18)
            for i, (ts, action_text) in enumerate(action_log[:max_entries]):
                ly = log_card_y + 40 + i * 18
                if ly + 10 > log_card_y + log_card_h:
                    break
                alpha = max(0.3, 1.0 - i * 0.15)
                text_col = lerp_color(TEXT_DIM, TEXT_LIGHT, alpha)
                # Timestamp
                cv2.putText(canvas, ts,
                            (card_x + 18, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, TEXT_DIM, 1, cv2.LINE_AA)
                # Action
                cv2.putText(canvas, action_text,
                            (card_x + 90, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, text_col, 1, cv2.LINE_AA)

        # ── STATUS BAR (bottom) ──────────────────────────────────────────
        status_y = TOTAL_H
        # Dark status bar background
        draw_filled_rounded_rect(canvas, (0, status_y), (TOTAL_W, WINDOW_H),
                                 (22, 18, 14), radius=0)
        cv2.line(canvas, (0, status_y), (TOTAL_W, status_y), BORDER_SUBTLE, 1)

        # FPS calculation
        fps_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            display_fps = fps_count / elapsed
            fps_count = 0
            fps_timer = time.time()

        sb_cy = status_y + STATUS_BAR_H // 2

        # FPS indicator
        fps_color = NEON_GREEN if display_fps >= 20 else NEON_AMBER if display_fps >= 10 else NEON_RED
        cv2.circle(canvas, (18, sb_cy), 4, fps_color, -1)
        cv2.putText(canvas, f"{display_fps:.0f} FPS",
                    (28, sb_cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_LIGHT, 1, cv2.LINE_AA)

        # Hand detection status
        hand_dot_color = NEON_GREEN if hand_detected else TEXT_DIM
        hand_text = "HAND OK" if hand_detected else "NO HAND"
        cv2.circle(canvas, (120, sb_cy), 4, hand_dot_color, -1)
        cv2.putText(canvas, hand_text,
                    (130, sb_cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_LIGHT, 1, cv2.LINE_AA)

        # Playback status
        play_dot_color = NEON_GREEN if player.is_playing else NEON_AMBER
        play_text = "PLAYING" if player.is_playing else "PAUSED"
        cv2.circle(canvas, (240, sb_cy), 4, play_dot_color, -1)
        cv2.putText(canvas, play_text,
                    (250, sb_cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_LIGHT, 1, cv2.LINE_AA)

        # Volume indicator in status bar
        vol_sb_text = f"VOL: {player.volume}%" if not player.is_muted else "VOL: MUTED"
        vol_sb_color = TEXT_LIGHT if not player.is_muted else NEON_RED
        cv2.putText(canvas, vol_sb_text,
                    (350, sb_cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, vol_sb_color, 1, cv2.LINE_AA)

        # Branding
        brand = "Gesture Media Player"
        (btw, _), _ = cv2.getTextSize(brand, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.putText(canvas, brand,
                    (TOTAL_W - btw - 100, sb_cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_DIM, 1, cv2.LINE_AA)

        # Timestamp
        timestamp = time.strftime("%H:%M:%S")
        (ttw, _), _ = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(canvas, timestamp,
                    (TOTAL_W - ttw - 14, sb_cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_MID, 1, cv2.LINE_AA)

        # ── Display ─────────────────────────────────────────────────────
        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        break

# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────
webcam.release()
if player.video_cap is not None:
    player.video_cap.release()
hand_landmarker.close()
cv2.destroyAllWindows()
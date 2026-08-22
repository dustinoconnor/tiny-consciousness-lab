#!/usr/bin/env python3
"""Leakage-resistant target deck and scoring for a remote-perception pilot."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import secrets
from pathlib import Path

from PIL import Image, ImageDraw


PALETTES = {
    "red": (220, 52, 58),
    "blue": (48, 111, 220),
    "yellow": (238, 194, 45),
    "green": (54, 163, 91),
    "purple": (137, 75, 190),
    "orange": (235, 125, 38),
}
SHAPES = ("circle", "triangle", "square", "star", "cross", "wave")
COUNTS = (1, 2, 3, 5)
ARRANGEMENTS = ("horizontal", "vertical", "diagonal", "radial")
TEXTURES = ("solid", "striped", "dotted")
BACKGROUNDS = ("light", "dark")
DESCRIPTOR_FIELDS = (
    "dominant_color",
    "shape",
    "count",
    "arrangement",
    "texture",
    "background",
)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_json(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def target_space():
    return [
        dict(zip(DESCRIPTOR_FIELDS, values))
        for values in itertools.product(
            PALETTES, SHAPES, COUNTS, ARRANGEMENTS, TEXTURES, BACKGROUNDS
        )
    ]


def opaque_id(rng, prefix):
    return f"{prefix}-{rng.getrandbits(64):016x}"


def object_centers(count, arrangement, size=384):
    center = size / 2
    if count == 1:
        return [(center, center)]
    spacing = 60 if count < 5 else 47
    offsets = [(index - (count - 1) / 2) * spacing for index in range(count)]
    if arrangement == "horizontal":
        return [(center + offset, center) for offset in offsets]
    if arrangement == "vertical":
        return [(center, center + offset) for offset in offsets]
    if arrangement == "diagonal":
        return [(center + offset * 0.72, center + offset * 0.72) for offset in offsets]
    radius = 76 if count < 5 else 91
    import math

    return [
        (
            center + radius * math.cos(2 * math.pi * index / count),
            center + radius * math.sin(2 * math.pi * index / count),
        )
        for index in range(count)
    ]


def shape_mask(shape, radius=34):
    size = radius * 2 + 8
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    c = size / 2
    r = radius
    if shape == "circle":
        draw.ellipse((c - r, c - r, c + r, c + r), fill=255)
    elif shape == "square":
        draw.rectangle((c - r, c - r, c + r, c + r), fill=255)
    elif shape == "triangle":
        draw.polygon([(c, c - r), (c + r, c + r), (c - r, c + r)], fill=255)
    elif shape == "star":
        import math

        points = []
        for index in range(10):
            angle = -math.pi / 2 + index * math.pi / 5
            length = r if index % 2 == 0 else r * 0.42
            points.append((c + length * math.cos(angle), c + length * math.sin(angle)))
        draw.polygon(points, fill=255)
    elif shape == "cross":
        q = r * 0.38
        draw.polygon(
            [
                (c - q, c - r), (c + q, c - r), (c + q, c - q),
                (c + r, c - q), (c + r, c + q), (c + q, c + q),
                (c + q, c + r), (c - q, c + r), (c - q, c + q),
                (c - r, c + q), (c - r, c - q), (c - q, c - q),
            ],
            fill=255,
        )
    elif shape == "wave":
        import math

        points = []
        for x in range(4, size - 4):
            y = c + math.sin((x - 4) / (size - 8) * 2.5 * math.pi) * r * 0.45
            points.append((x, y))
        draw.line(points, fill=255, width=15)
    else:
        raise ValueError(f"unknown_shape:{shape}")
    return mask


def textured_tile(mask, color, texture):
    tile = Image.new("RGB", mask.size, color)
    draw = ImageDraw.Draw(tile)
    accent = tuple(max(0, channel - 75) for channel in color)
    if texture == "striped":
        for offset in range(-mask.size[1], mask.size[0], 12):
            draw.line((offset, 0, offset + mask.size[1], mask.size[1]), fill=accent, width=5)
    elif texture == "dotted":
        for x in range(5, mask.size[0], 13):
            for y in range(5, mask.size[1], 13):
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=accent)
    elif texture != "solid":
        raise ValueError(f"unknown_texture:{texture}")
    tile.putalpha(mask)
    return tile


def render_card(metadata, path, size=384):
    background = (239, 239, 231) if metadata["background"] == "light" else (24, 27, 35)
    image = Image.new("RGB", (size, size), background)
    mask = shape_mask(metadata["shape"])
    tile = textured_tile(mask, PALETTES[metadata["dominant_color"]], metadata["texture"])
    for x, y in object_centers(metadata["count"], metadata["arrangement"], size):
        image.paste(tile, (round(x - tile.width / 2), round(y - tile.height / 2)), tile)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def validate_prediction(prediction):
    if set(prediction) != set(DESCRIPTOR_FIELDS):
        raise ValueError("remote_prediction_schema_mismatch")
    allowed = {
        "dominant_color": set(PALETTES),
        "shape": set(SHAPES),
        "count": set(COUNTS),
        "arrangement": set(ARRANGEMENTS),
        "texture": set(TEXTURES),
        "background": set(BACKGROUNDS),
    }
    normalized = dict(prediction)
    normalized["count"] = int(normalized["count"])
    for field in DESCRIPTOR_FIELDS:
        if normalized[field] not in allowed[field]:
            raise ValueError(f"remote_prediction_invalid_{field}")
    return normalized


def descriptor_score(prediction, target):
    prediction = validate_prediction(prediction)
    return sum(prediction[field] == target[field] for field in DESCRIPTOR_FIELDS)


def rank_candidates(prediction, candidates):
    scored = [
        {"target_id": item["target_id"], "score": descriptor_score(prediction, item)}
        for item in candidates
    ]
    return sorted(scored, key=lambda item: (-item["score"], item["target_id"]))


def prediction_commitment(trial_id, prediction, nonce):
    return sha256_json(
        {"trial_id": str(trial_id), "prediction": validate_prediction(prediction), "nonce": str(nonce)}
    )


def make_trials(cards, rng):
    trials = []
    for target in cards:
        alternatives = [item for item in cards if item["target_id"] != target["target_id"]]
        decoys = rng.sample(alternatives, 3)
        candidate_ids = [target["target_id"], *(item["target_id"] for item in decoys)]
        rng.shuffle(candidate_ids)
        trials.append(
            {
                "trial_id": opaque_id(rng, "RV"),
                "target_id": target["target_id"],
                "candidate_ids": candidate_ids,
            }
        )
    return trials


def generate_deck(output_dir, training=20, reserved=100, seed=None):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"remote_target_directory_not_empty:{output_dir}")
    seed = secrets.randbits(128) if seed is None else int(seed)
    rng = random.Random(seed)
    space = target_space()
    rng.shuffle(space)
    selected = space[: training + reserved]
    cards = []
    for index, metadata in enumerate(selected):
        split = "training" if index < training else "reserved"
        target_id = opaque_id(rng, "T")
        card = {"target_id": target_id, "split": split, **metadata}
        cards.append(card)
        render_card(card, output_dir / split / f"{target_id}.png")
    training_cards = [item for item in cards if item["split"] == "training"]
    reserved_cards = [item for item in cards if item["split"] == "reserved"]
    private = {
        "format": "remote_perception_deck_v1",
        "seed": seed,
        "descriptor_fields": list(DESCRIPTOR_FIELDS),
        "training_cards": training_cards,
        "reserved_cards": reserved_cards,
        "training_trials": make_trials(training_cards, rng),
        "reserved_trials": make_trials(reserved_cards, rng),
    }
    commitment = {
        "format": private["format"],
        "training_targets": training,
        "reserved_targets": reserved,
        "chance_top1": 0.25,
        "private_manifest_sha256": sha256_json(private),
        "reserved_images_sha256": sha256_json(
            {
                item["target_id"]: hashlib.sha256(
                    (output_dir / "reserved" / f"{item['target_id']}.png").read_bytes()
                ).hexdigest()
                for item in reserved_cards
            }
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "private_manifest.json").write_text(
        json.dumps(private, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "public_commitment.json").write_text(
        json.dumps(commitment, indent=2, sort_keys=True) + "\n"
    )
    return private, commitment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/remote_perception_deck_20260818")
    parser.add_argument("--training", type=int, default=20)
    parser.add_argument("--reserved", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    _private, commitment = generate_deck(
        args.output, training=args.training, reserved=args.reserved, seed=args.seed
    )
    print(json.dumps(commitment, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

# Minimal Unity Trap Course

This is the reproducible Unity side of Tiny Consciousness Lab. It intentionally
contains no terrain packs, purchased assets, robot model, or generated Library
directory. The six courses, test deck, capsule robot, food target, colliders,
camera, and HUD use Unity primitives.

## Requirements

- Unity 6.4 (`6000.4.2f1` was used for validation)
- Python dependencies from the repository root

## Build the scene

1. Add `unity/TrapCourseLab` to Unity Hub and open it.
2. Choose `Tiny Consciousness > Build Minimal Trap Course` once if the generated
   scene is missing.
3. Open `Assets/Scenes/TrapCourseLab.unity` and press Play.
4. From the repository root, run:

```bash
python3 embodied_unity_loop.py \
  --shadow-policy checkpoints/unity_mpc/best.pt \
  --shadow-control course \
  --shadow-control-confidence 0.0 \
  --shadow-mpc
```

The course advances after each pickup. Press `N`/`B` for next/previous course,
`R` to reset the episode, and `H` to hide or show the compact status HUD.

The project communicates only over localhost UDP ports 5055 and 5056. Raw
telemetry recordings remain ignored by Git because long runs can be large.

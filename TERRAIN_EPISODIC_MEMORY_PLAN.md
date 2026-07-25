# Terrain Episodic Memory Plan

This plan preserves the current low-failure Unity terrain while creating a
separate difficult environment for AIR-guided episodic learning.

## Preserve the Garden of Eden Baseline

The active terrain project currently uses:

- scene: `Assets/Scenes/SampleScene.unity`
- terrain data: `Assets/New Terrain.asset`
- generated rocks and mushrooms: serialized scene GameObjects
- painted terrain trees: serialized inside the terrain data asset

Duplicating only the scene is not enough. Two scenes that reference the same
`TerrainData` will share painted-tree and terrain edits.

Before changing density:

1. Duplicate `SampleScene.unity` as `GardenOfEdenBaseline.unity`.
2. Duplicate `New Terrain.asset` as `GardenOfEdenBaseline Terrain.asset`.
3. Open the baseline scene and assign the duplicated terrain data to its
   `Terrain` and `TerrainCollider`.
4. Reopen the original scene and verify it still references `New Terrain.asset`.
5. Preserve generated rock and mushroom parents in the baseline scene.
6. Run a short baseline smoke test before creating difficult terrain.

The difficult copy should also receive its own scene and terrain data, for
example `EpisodicTrainingTerrain.unity` and
`EpisodicTraining Terrain.asset`.

## Four-Context Conductor Sequence

The Python benchmark establishes the intended routing policy:

| Context | Specialist |
| --- | --- |
| Clear terrain | Recurrent exploration |
| Familiar hidden goal | Episodic ART playback |
| Visible target | Predictive MPC |
| Genuine physical wedge | Stable fallback |

The first Unity integration should remain passive. It should log conductor
probabilities, gate entropy, context transitions, proposed specialist, active
specialist, handoff latency, and disagreement without changing movement.

## AIR-Guided Encoding

Terrain memory should not record every telemetry frame. Build one
intermediate-level egocentric packet from:

- normalized radial obstacle distances;
- local slope and body clearance;
- visible or recently lost food direction;
- displacement, angular velocity, and current action;
- ART resonance and novelty;
- prediction error and uncertainty;
- hunger, valence, and reward outcome.

A learned attention gate should promote only a bounded number of packets into
working memory and episodic storage. Candidate promotion events include:

- first detection or loss of food;
- unexpected collision or prediction error;
- discovery of a new opening;
- successful wedge escape;
- sharp valence change;
- mushroom consumption;
- abandonment of a barren region.

## Required Controls

Compare equal-capacity episodic libraries:

1. encode every frame, then uniformly downsample;
2. randomly select packets;
3. hand-engineered salience selection;
4. learned AIR-style attention;
5. learned attention with selected packets scrambled;
6. learned attention without valence;
7. learned attention without prediction error.

Evaluate retrieval precision, false recall, storage size, route success,
collisions, time to food, and compute cost on translated, rotated, and altered
terrain patches withheld from training.

## Live-Control Boundary

Do not replay absolute world-space paths as the final mechanism. Store local
trajectory deltas in the robot's egocentric frame and transform them into the
current frame after ART resonance. Abort recall immediately when ray geometry
mismatches, displacement stalls, food evidence changes, or route uncertainty
exceeds its calibrated bound.

Only after passive recall predicts a measurable advantage should the conductor
receive bounded authority to activate terrain episodic playback.

RECURRENT MIDI LAB 1.0

System requirements
- Apple Silicon Mac (M1 or newer)
- macOS 13 or newer
- A MIDI destination such as the macOS IAC Driver or a DAW virtual input

Install
1. Move Recurrent MIDI Lab.app into Applications if desired.
2. This research build is ad-hoc signed but not Apple-notarized. On first launch,
   Control-click the app, choose Open, and confirm Open if macOS blocks it.
3. Select a MIDI output, arm a software instrument in your DAW, and press Start.

The app produces MIDI messages, not audio. The selected DAW or instrument creates
the sound. Panic sends all-notes-off if a synth keeps holding a note.

Research boundary
The app uses compact recurrent pitch and rhythm policies trained for a procedural
symbolic-music task. It demonstrates architecture-level transfer and retraining,
not reuse of navigation weights or general musical intelligence.

Source and evidence
https://github.com/dustinoconnor/tiny-consciousness-lab

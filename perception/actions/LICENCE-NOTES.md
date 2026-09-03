# Action recognition — licence position

**Unresolved. Resolve before any of this ships.** These notes are a starting
list to check, not settled fact: terms change, and they were not verified from
inside this repo.

This is the same class of problem as the AGPL detector backend, which was
caught early and cost nothing. Discovering it after a classifier has been
trained and embedded is a different matter entirely.

## The three to check

**NTU RGB+D.** Research-only, under an academic use agreement. PoseC3D
checkpoints are commonly NTU-pretrained, and fine-tuning does not launder it —
a derived model inherits the restriction. This is the main one.

**It is now in the tree, not hypothetical.**
`perception/models/pose_only_20230228-fa40054e.pth` and the ONNX exported from
it are NTU-60 trained: `cls_head.fc_cls` is a 60-class head over NTU's daily
-living vocabulary. It is here to prove the pipeline end to end and nothing
else. Two consequences worth being explicit about — a model fine-tuned from
this checkpoint inherits the research-only restriction, and the weights are
gitignored precisely so this does not become a distribution question by
accident.

**Kinetics-pretrained weights**, where a checkpoint has them in its history.
The annotations are permissively licensed; the underlying YouTube videos are
not uniformly so.

**RTMPose-t, the pose model.** The lightest of the three, but not nothing. The
MMPose code is Apache-2.0 and so is the reimplementation in `rtmpose_arch.py`.
The checkpoint in use here is `rtmpose-tiny_simcc-aic-coco`, trained on COCO
and **AI Challenger**. COCO images are Flickr-sourced under per-image terms
with CC-BY-4.0 annotations; AI Challenger has its own terms, and it is the one
to read rather than assume. If it does not clear, the same checkpoint is
published COCO-only — a little less accurate, one fewer question to answer.

## If they do not clear

- A checkpoint pretrained on permissively licensed data, terms checked
  independently
- Train from scratch on your own site footage. More data needed, but it
  sidesteps the question entirely and the data is yours
- Buy a commercial licence

## Datasets, if you train

**Withdrawn over consent problems** — MS-Celeb-1M, DukeMTMC, and others. Using
a withdrawn dataset in a commercial product is a reputational problem as much
as a legal one.

**Research-only** — NTU RGB+D, Something-Something V2, most person re-ID sets.

**Generally usable** — the PPE sets (SHWD, CHV, Pictor-v3) and much of Roboflow
Universe, though quality varies and site-specific retraining is needed anyway.

## Your own footage

Filming workers to build a training set is a **different purpose** from
monitoring them, and needs its own legal basis: worker notification, and in the
EU usually a DPIA and often works council agreement. Sort this before the camera
goes up, not after.

See the regulatory triage document — continuous behavioural monitoring against
a prohibited-action list is Annex III.4 employment high-risk under the EU AI
Act, which is a stronger position than the process monitoring scoped elsewhere.

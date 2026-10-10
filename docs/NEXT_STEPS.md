# Next steps -- ai-vision-platform (training side)

Branch: `claude/exe-update-data-safety-plk61b`. App-side notes: `NEXT_STEPS.md` in the flutter-ai-platform repo
(branch `claude/yolo26-end2end-decode`).

## Already built
- **Data Segments** (Training Hub): rules over per-image class counts (+ total annotations), saved per project,
  preview with thumbnails / unmatched list / gallery with boxes, "Create project copy", used as a filter by
  Seed / Main / Segmentation and as the class label by the Classifier (crop class + segment label).
- Classifier: class balancing, Auto epochs, saved jobs / live log, train-loss curve fix, segment-aware UI.
- Seed / Main: recommended epochs (dataset-size based, segment-aware).
- Review: "Auto-annotated only" filter. Delete key removes the current image. Imports mark images with boxes as annotated.
- Project copy / duplicate fixed for video-frame images (subfolder paths).

## Key finding
Classifier validation was 100% but on-site results fail. Frames from the same video are near-identical and the split
is **by image**, so validation measures memory, not generalisation. (Plus low-res phone frames, Preprocess ON in
training but not in the app, only a few distinct scenes.)

## Backlog (priority order)
1. **Split by video (group split)** in classifier (and detector) training: all frames of one video on one side
   (frame paths already contain `video_frames/<video_id>/`). Report honest validation accuracy.
2. **Hard-example loop** (idea from the user):
   - Run new videos through the current detector + classifier (the sequence runner / the APK).
   - Collect frames where it fails: no detection, uncertain, or label != the video's declared state; skip blurred frames.
   - Review them (quick UI), add to the dataset (keep all old data), retrain from the previous weights
     (custom weights) for detector and classifier, repeat with the next videos.
   - Protocol: use each new video first as a HELD-OUT test (measure real accuracy), then mine failures, then train on it.
3. **Third class `unclear`** (blurred, wrong view, misdetected box) so the classifier can say "not sure".
4. **Phone-like augmentation** for the classifier: blur, brightness, downscale; train with Preprocess OFF for phone models.
5. Classifier Stop button keeps best-so-far instead of discarding.
6. Optional: "simulate phone resolution" in the classifier Test panel (shrink the test image) to measure the resolution effect.
7. Data collection plan: many different cars / positions / lighting / distances, shaky + blurry clips, saved on-site frames.

## Open questions
- How many separate videos does each of locked / unlocked come from?
- Was Preprocess on for the trained classifier / detector?

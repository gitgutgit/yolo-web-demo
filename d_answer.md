# YOLO Visibility Investigation

## Current Situation
You reported that YOLO detection boxes are still not visible in the new `index.html`, but `index_what.html` (when renamed) shows something different (though maybe not perfect). You also mentioned the game screen looking "shifted up" and questioned if low confidence or HTML issues are to blame.

## Hypotheses & Investigation Plan

### 1. Coordinate Mapping & Screen Shift (Most Likely)
**User Observation:** "Game screen seems half shifted up."
**Analysis:** If the CSS or HTML structure in the new `index.html` is scaling or positioning the canvas differently than the server expects, the YOLO boxes (which are drawn based on server coordinates) will be misaligned.
*   **Action:** I will compare the CSS/Canvas setup in `index.html` vs `index_what.html`.
*   **Action:** I will verify if the `renderDetections` function in `index.html` correctly maps the normalized (0-1) or pixel coordinates sent by the server.

### 2. YOLO Confidence
**User Observation:** "Is YOLO seeing everything with near 0 confidence?"
**Analysis:** It's possible. If the server-side rendering (`game.render_frame()`) produces an image that is slightly different from what the model was trained on (e.g., different colors, background), detections might fail.
*   **Action:** I will add server-side logging to print exactly what YOLO is detecting and with what confidence.

### 3. Rendering Logic Difference
**User Observation:** `index_what.html` behaves differently.
**Analysis:** `index_what.html` might have a working `renderDetections` implementation that was lost or altered during the merge.
*   **Action:** I will read `index_what.html` to see how it renders boxes.

## Next Steps
1.  **Analyze `index_what.html`**: I will read this file immediately to understand its rendering logic.
2.  **Debug Server**: I'll add print statements to `app.py` to confirm if detections are actually being generated.
3.  **Fix `index.html`**: I will correct the rendering logic or CSS based on the findings.

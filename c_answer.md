# YOLO Integration & Visualization Answer

## 1. AI Input Source
**Question:** "Is the AI currently using YOLO detections as the main input, or just the raw state like before?"

**Answer:**
**YES, the AI is using YOLO detections.**
I analyzed `web_app/app.py` and `web_app/modules/ai_module.py`, and the workflow is as follows:
1.  **Render**: The server renders the current game state into an image (`game.render_frame()`).
2.  **Detect**: This image is passed to the YOLO model (`game.run_yolo(frame)`).
3.  **Encode**: The detection results (bounding boxes, classes) are converted into a state vector (`encode_state(detections)`).
4.  **Decide**: This vector is passed to the PPO model to decide the action.

So, the AI **relies on YOLO's ability to "see" the game**. If YOLO fails to detect meteors, the AI will not know they are there (unless it falls back to the heuristic).

## 2. Missing Detection Boxes
**Question:** "Why are the detection boxes not visible? The toggle also seems not to work."

**Answer:**
There are two reasons for this:

1.  **YOLO Only Runs in AI Mode (Currently)**:
    In `app.py`, the YOLO inference (`game.run_yolo`) is currently called *inside* the `ai_decision` function. This function is **only executed when the game mode is 'AI'**.
    *   **Result**: In 'Human' mode, YOLO is never run, so no detection data is generated or sent to the client.

2.  **Broken Toggle Key**:
    The server has a `toggle_detections` event listener, but the client-side code (`index.html`) **does not have a handler for the 'G' key**. Pressing 'G' currently does nothing.

## 3. Solution Plan
I will perform the following fixes to address this:

1.  **Enable YOLO in Human Mode (Optional)**:
    I will modify `app.py` to run YOLO inference in 'Human' mode *if* the detection boxes are enabled (`game.show_detections` is True). This will allow you to verify that YOLO is working correctly even while playing manually.

2.  **Fix the Toggle Key**:
    I will add the 'G' key handler to `index.html` so you can toggle the boxes on and off.

3.  **Visualization**:
    The visualization logic I added to `index.html` (`renderDetections`) is correct for the data format the server sends. Once the data starts flowing, the boxes will appear.

# Tennis Shot Classification System

## Height-Based Shot Classification

The system classifies tennis shots based on ball apex height (maximum trajectory height) using professional tennis standards.

### Classification Criteria

#### 1️⃣ **Lob Shot** — Very High Trajectory
- **Apex Height Range**: 6–10 meters above ground
- **Net Clearance**: 1.5–3.0 meters above net
- **Trajectory Type**: High parabolic arc with long air-time
- **Label**: `SHOT_TYPE = lob`, `HEIGHT_CLASS = very_high`
- **Use Cases**: Defensive play, passing shots, buying time

#### 2️⃣ **Drop Shot** — Very Low Trajectory
- **Apex Height Range**: 1.2–1.8 meters above ground
- **Net Clearance**: 0.2–0.6 meters (minimal clearance)
- **Trajectory Type**: Short, shallow arc with rapid downward decay
- **Label**: `SHOT_TYPE = drop_shot`, `HEIGHT_CLASS = very_low`
- **Use Cases**: Short game, catching opponent off-guard

#### 3️⃣ **Cross-Court Groundstroke** — Medium-High Trajectory
- **Apex Height Range**: 2.5–4.0 meters above ground
- **Net Clearance**: 0.8–1.5 meters (moderate to high)
- **Trajectory Type**: Stable parabolic arc, diagonal path
- **Label**: `SHOT_TYPE = cross_court_groundstroke`, `HEIGHT_CLASS = medium_high`
- **Use Cases**: Rally baseline play, controlling court position

#### 4️⃣ **Down-the-Line Groundstroke** — Medium-Low Trajectory
- **Apex Height Range**: 2.0–3.0 meters above ground
- **Net Clearance**: 0.4–1.0 meters (lower than cross-court)
- **Trajectory Type**: Flatter arc, linear forward path
- **Label**: `SHOT_TYPE = down_the_line_groundstroke`, `HEIGHT_CLASS = medium_low`
- **Use Cases**: Aggressive baseline play, winner attempts

### Additional Classifications

#### 5️⃣ **High Lob** — Extremely High
- **Apex Height Range**: > 10 meters
- **Label**: `SHOT_TYPE = high_lob`, `HEIGHT_CLASS = extremely_high`

#### 6️⃣ **Low Shot** — Extremely Low
- **Apex Height Range**: < 1.2 meters
- **Label**: `SHOT_TYPE = low_shot`, `HEIGHT_CLASS = extremely_low`

#### 7️⃣ **Slice/Defensive** — Low Trajectory
- **Apex Height Range**: 1.8–2.5 meters
- **Label**: `SHOT_TYPE = slice`, `HEIGHT_CLASS = low`

#### 8️⃣ **Regular Groundstroke** — Overlapping Ranges
- **Apex Height Range**: 2.0–4.0 meters (when direction is not clearly cross-court or down-the-line)
- **Label**: `SHOT_TYPE = groundstroke`, `HEIGHT_CLASS = medium_low/medium_high`

---

## Direction Classification

### Horizontal Movement
- **Straight**: < 20% of court width movement
- **Cross-Court Right**: Significant rightward movement (dx > 0)
- **Cross-Court Left**: Significant leftward movement (dx < 0)

### Vertical Movement
- **Flat**: < 15% of court height movement
- **Deep**: Moving toward baseline (dy > 0)
- **Short**: Moving toward net (dy < 0)

### Combined Directions
- `straight`
- `down_the_line_deep`
- `down_the_line_short`
- `cross_court_left_deep`
- `cross_court_left_short`
- `cross_court_right_deep`
- `cross_court_right_short`

---

## Technical Implementation

### Apex Height Calculation
```python
# Y-axis is inverted in image coordinates (0 at top)
lowest_y_px = max(y_coordinates)  # Bottom of trajectory
apex_y_px = min(y_coordinates)    # Top of trajectory
apex_height_px = lowest_y_px - apex_y_px
apex_height_m = apex_height_px * pixels_per_meter
```

### Classification Logic
1. **Calculate Apex Height**: Find maximum vertical displacement from lowest point
2. **Determine Direction**: Analyze start-to-end position vector
3. **Apply Height Thresholds**: Match apex height to shot type ranges
4. **Combine with Direction**: Refine groundstroke classification based on trajectory direction
5. **Output Confidence**: Based on number of frames tracked (more frames = higher confidence)

---

## Example Output

```
Track 12: Lob (Very High, 8.3m apex) - Cross Court Right Short @ 130.0 km/h
  - Type: lob
  - Height Class: very_high
  - Apex Height: 8.35m
  - Direction: cross_court_right_short
  - Zone: front_left → mid_center
  - Court Side: left
  - Confidence: 0.85
```

---

## Calibration Requirements

For accurate height measurements:
- **Pixels-per-meter ratio**: Must be calibrated using known court dimensions
- **Baseline-to-baseline distance**: 23.77 meters (standard tennis court)
- **Example calibration**: 22.1 pixels/meter for 1280x720 video

Use `calibrate_court.py` to interactively calibrate using known court features.

---

## Visual Display

Real-time overlay shows:
- **Track ID**: Unique identifier
- **Shot Type**: Classified shot name
- **Apex Height**: Height in meters with height class
- **Speed**: Average speed in km/h
- **Direction**: Movement pattern

---

## Files

- `trajectory_analyzer.py`: Apex height calculation (`calculate_apex_height()`)
- `shot_classifier.py`: Height-based classification (`classify_shot_type()`)
- `track_and_analyze.py`: Integration and visualization
- `calibrate_court.py`: Pixel-to-meter calibration tool

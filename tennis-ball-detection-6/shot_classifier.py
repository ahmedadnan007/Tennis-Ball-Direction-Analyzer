"""
Shot Classifier - Simplified
Shows: Forehand / Backhand / Serve + Direction + Speed
Removed: Topspin, Slice, Flat spin detection
"""

import numpy as np
from typing import Dict, List, Optional


class ShotClassifier:
    """Simplified tennis shot classifier - Forehand/Backhand/Serve + Direction."""

    def __init__(self, court_width=1280, court_height=720,
                 court_orientation='horizontal', court_detector=None,
                 player_handed='right'):
        self.court_width    = court_width
        self.court_height   = court_height
        self.court_detector = court_detector
        self.player_handed  = player_handed
        self.zones          = self._define_court_zones()

    def _define_court_zones(self):
        zones    = {}
        h_third  = self.court_height // 3
        w_third  = self.court_width  // 3
        names    = [
            ['back_left',  'back_center',  'back_right'],
            ['mid_left',   'mid_center',   'mid_right'],
            ['front_left', 'front_center', 'front_right'],
        ]
        for row in range(3):
            for col in range(3):
                zones[names[row][col]] = {
                    'x_min': col * w_third,
                    'x_max': (col + 1) * w_third,
                    'y_min': row * h_third,
                    'y_max': (row + 1) * h_third,
                }
        return zones

    def get_zone(self, position):
        x, y = position
        for name, b in self.zones.items():
            if b['x_min'] <= x < b['x_max'] and b['y_min'] <= y < b['y_max']:
                return name
        return 'unknown'

    # ============================================================
    # DIRECTION: Cross Court / Down The Line / Straight
    # ============================================================
    def classify_direction(self, start_pos, end_pos):
        x1, y1 = start_pos
        x2, y2 = end_pos
        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            return 'straight'

        relative_dx = abs(dx) / self.court_width
        relative_dy = abs(dy) / self.court_height

        # Horizontal direction
        if relative_dx < 0.10:
            horizontal = 'straight'
        elif dx > 0:
            horizontal = 'cross_court_right'
        else:
            horizontal = 'cross_court_left'

        # Vertical (depth)
        if relative_dy < 0.10:
            vertical = 'flat'
        elif dy > 0:
            vertical = 'deep'
        else:
            vertical = 'short'

        if horizontal == 'straight':
            if vertical in ('flat', 'deep'):
                return 'down_the_line'
            return 'down_the_line_short'
        else:
            return horizontal  # cross_court_right / cross_court_left

    # ============================================================
    # SHOT TYPE: Forehand / Backhand / Serve
    # ============================================================
    def classify_stroke(self, start_pos, end_pos, trajectory_analysis):
        """
        Detect Serve, Forehand, or Backhand.

        Serve detection:
        - Ball starts near top of frame (y < 25% of height)
        - Ball has high downward velocity
        - High speed (>100 km/h typical)

        Forehand/Backhand:
        - Based on ball position relative to court center
        - Right-handed: ball on right side = forehand
        """
        x1, y1 = start_pos
        x2, y2 = end_pos

        speed_kmh  = trajectory_analysis.get('avg_speed_kmh', 0)
        num_frames = trajectory_analysis.get('num_frames', 0)
        positions  = trajectory_analysis.get('positions_pixels', [])

        # ---- SERVE DETECTION ----
        # Ball starts high up in frame + moving downward fast
        start_y_ratio = y1 / self.court_height

        # Check net position if calibrated
        net_y = self.court_height * 0.4  # default
        if self.court_detector and hasattr(self.court_detector, 'net_position'):
            net_pos = self.court_detector.net_position
            if net_pos and net_pos > 0:
                net_y = net_pos

        # Ball starts above net level and moves down = likely serve
        moving_down = y2 > y1
        starts_high = y1 < net_y * 0.8  # starts above 80% of net height

        if starts_high and moving_down and speed_kmh > 50:
            return 'serve'

        # ---- FOREHAND / BACKHAND ----
        court_center = self.court_width / 2
        ball_x       = x1  # where ball was when hit

        # 10% buffer around center
        buffer = self.court_width * 0.10

        if ball_x < court_center - buffer:
            # Ball on LEFT side of court
            stroke = 'backhand' if self.player_handed == 'right' else 'forehand'
        elif ball_x > court_center + buffer:
            # Ball on RIGHT side of court
            stroke = 'forehand' if self.player_handed == 'right' else 'backhand'
        else:
            # Near center - use direction of travel to decide
            if (dx := x2 - x1) > 0:
                stroke = 'forehand' if self.player_handed == 'right' else 'backhand'
            elif dx < 0:
                stroke = 'backhand' if self.player_handed == 'right' else 'forehand'
            else:
                stroke = 'forehand'  # default

        return stroke

    # ============================================================
    # SHOT TYPE: Groundstroke / Lob / Drop Shot
    # ============================================================
    def classify_shot_type(self, trajectory_analysis):
        apex_info    = trajectory_analysis.get('apex_height', {})
        apex_height_m = apex_info.get('apex_height_meters', 0) if apex_info else 0
        direction    = trajectory_analysis.get('shot_direction', 'unknown')

        if apex_height_m > 4.5:
            return 'lob'
        elif apex_height_m >= 1.5:
            return 'groundstroke'
        elif apex_height_m > 0 and apex_height_m < 1.5:
            return 'drop_shot'

        # Pixel fallback
        positions = trajectory_analysis.get('positions_pixels', [])
        if len(positions) >= 3:
            y_vals    = [p[1] for p in positions]
            min_y     = min(y_vals)
            apex_drop = positions[0][1] - min_y
            rel_apex  = apex_drop / self.court_height

            if rel_apex > 0.35:
                return 'lob'
            elif rel_apex > 0.08:
                return 'groundstroke'
            else:
                return 'drop_shot'

        return 'groundstroke'

    def _calculate_confidence(self, trajectory_analysis):
        num_frames = trajectory_analysis.get('num_frames', 0)
        positions  = trajectory_analysis.get('positions_pixels', [])
        score      = 0.0

        if num_frames >= 40:
            score += 0.40
        elif num_frames >= 20:
            score += 0.30
        elif num_frames >= 10:
            score += 0.20
        elif num_frames >= 5:
            score += 0.10

        if len(positions) >= 5:
            x_diffs  = np.diff([p[0] for p in positions])
            y_diffs  = np.diff([p[1] for p in positions])
            x_var    = np.std(x_diffs) / (np.mean(np.abs(x_diffs)) + 1e-6)
            y_var    = np.std(y_diffs) / (np.mean(np.abs(y_diffs)) + 1e-6)
            smooth   = 1.0 / (1.0 + (x_var + y_var) / 2)
            score   += 0.30 * smooth

        apex_info = trajectory_analysis.get('apex_height', {})
        if apex_info and apex_info.get('apex_height_meters', 0) > 0:
            score += 0.30
        elif len(positions) >= 3:
            score += 0.15

        return min(score, 1.0)

    def classify_shot(self, trajectory_analysis):
        """Main classification - returns Forehand/Backhand/Serve + Direction."""
        if not trajectory_analysis or len(trajectory_analysis.get('positions_pixels', [])) < 2:
            return {
                'type':       'groundstroke',
                'stroke':     'unknown',
                'direction':  'straight',
                'spin':       '',           # empty - not shown
                'start_zone': 'unknown',
                'end_zone':   'unknown',
                'confidence': 0.0,
                'avg_speed_kmh':   0,
                'max_speed_kmh':   0,
                'duration_seconds': 0,
            }

        positions  = trajectory_analysis['positions_pixels']
        start_pos  = positions[0]
        end_pos    = positions[-1]

        direction  = self.classify_direction(start_pos, end_pos)
        trajectory_analysis['shot_direction'] = direction

        shot_type  = self.classify_shot_type(trajectory_analysis)
        stroke     = self.classify_stroke(start_pos, end_pos, trajectory_analysis)
        confidence = self._calculate_confidence(trajectory_analysis)

        apex_info     = trajectory_analysis.get('apex_height', {})
        apex_height_m = apex_info.get('apex_height_meters', 0) if apex_info else 0

        return {
            'type':              shot_type,
            'stroke':            stroke,         # forehand / backhand / serve
            'direction':         direction,       # cross_court / down_the_line / straight
            'spin':              '',              # REMOVED - not shown
            'height_class':      '',
            'apex_height_meters': apex_height_m,
            'start_zone':        self.get_zone(start_pos),
            'end_zone':          self.get_zone(end_pos),
            'side':              'unknown',
            'confidence':        confidence,
            'avg_speed_kmh':     trajectory_analysis.get('avg_speed_kmh', 0),
            'max_speed_kmh':     trajectory_analysis.get('max_speed_kmh', 0),
            'duration_seconds':  trajectory_analysis.get('duration_seconds', 0),
            'advanced_zone':     {},
        }

    def get_shot_summary(self, classification):
        stroke    = classification.get('stroke',    'unknown').title()
        direction = classification.get('direction', 'straight').replace('_', ' ').title()
        speed     = classification.get('avg_speed_kmh', 0)
        conf      = classification.get('confidence', 0)

        summary = f"{stroke} - {direction} @ {speed:.1f} km/h [{conf:.0%}]"
        return summary

    def analyze_rally(self, trajectory_analyses):
        if not trajectory_analyses:
            return {}
        classifications = [self.classify_shot(t) for t in trajectory_analyses]
        speeds = [c['avg_speed_kmh'] for c in classifications if c['avg_speed_kmh'] > 0]
        strokes    = {}
        directions = {}
        for c in classifications:
            strokes[c['stroke']]       = strokes.get(c['stroke'], 0) + 1
            directions[c['direction']] = directions.get(c['direction'], 0) + 1
        return {
            'num_shots':       len(classifications),
            'strokes':         strokes,
            'directions':      directions,
            'avg_speed':       np.mean(speeds) if speeds else 0,
            'max_speed':       np.max(speeds)  if speeds else 0,
            'classifications': classifications,
        }
"""Professional broadcast-style overlay matching real tennis match graphics."""

import cv2
import numpy as np
from typing import Dict, Optional


class BroadcastOverlay:
    """Creates professional tennis match overlay with dual-player statistics."""
    
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Player statistics tracking
        self.player1_stats = {
            'shot_speed': 0.0,
            'player_speed': 0.0,
            'shot_speeds': [],  # History for average
            'point_speeds': []  # Point speeds
        }
        
        self.player2_stats = {
            'shot_speed': 0.0,
            'player_speed': 0.0,
            'shot_speeds': [],
            'point_speeds': []
        }
        
        # Current shot analysis
        self.current_shot = {
            'shot_type': '-',
            'spin': '-',
            'direction': '-'
        }
        
        # Colors
        self.bg_color = (30, 30, 30)  # Dark background
        self.accent_color = (255, 200, 50)  # Yellow accent
        self.text_color = (255, 255, 255)  # White text
        self.value_color = (0, 255, 100)  # Green for values
    
    def update_shot(self, shot_data: Dict, is_player1: bool = True):
        """Update shot statistics for a player.
        
        Args:
            shot_data: Dictionary with shot analysis data
            is_player1: True for player 1, False for player 2
        """
        player = self.player1_stats if is_player1 else self.player2_stats
        
        # Update shot speed
        if 'speed' in shot_data:
            player['shot_speed'] = shot_data['speed']
            player['shot_speeds'].append(shot_data['speed'])
            player['point_speeds'].append(shot_data['speed'])
            
            # Keep only last 100 shots for average
            if len(player['shot_speeds']) > 100:
                player['shot_speeds'] = player['shot_speeds'][-100:]
        
        # Update current shot analysis
        if 'shot_type' in shot_data:
            self.current_shot['shot_type'] = shot_data['shot_type']
        if 'spin' in shot_data:
            self.current_shot['spin'] = shot_data['spin']
        if 'direction' in shot_data:
            self.current_shot['direction'] = shot_data['direction']
    
    def reset_point(self):
        """Reset point-specific statistics."""
        self.player1_stats['point_speeds'] = []
        self.player2_stats['point_speeds'] = []
    
    def draw(self, frame: np.ndarray, player1_name: str = "Player 1", 
             player2_name: str = "Player 2") -> np.ndarray:
        """Draw the complete broadcast overlay on frame.
        
        Args:
            frame: Video frame
            player1_name: Name of player 1
            player2_name: Name of player 2
            
        Returns:
            Frame with overlay
        """
        # Position: top-left corner
        x = 10
        y = 10
        panel_width = 350
        panel_height = 280
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Main panel background
        cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height),
                     self.bg_color, -1)
        
        # Yellow accent bar at top
        cv2.rectangle(overlay, (x, y), (x + panel_width, y + 8),
                     self.accent_color, -1)
        
        # Border
        cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height),
                     (80, 80, 80), 2)
        
        # Blend with transparency
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Draw content
        self._draw_header(frame, x, y, panel_width, player1_name, player2_name)
        self._draw_speed_stats(frame, x, y + 40, panel_width)
        self._draw_shot_analysis(frame, x, y + 180, panel_width)
        
        return frame
    
    def _draw_header(self, frame, x, y, width, p1_name, p2_name):
        """Draw player names header."""
        # Player 1 name
        cv2.putText(frame, p1_name, (x + 20, y + 30),
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, self.text_color, 2, cv2.LINE_AA)
        
        # Player 2 name
        cv2.putText(frame, p2_name, (x + width - 120, y + 30),
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, self.text_color, 2, cv2.LINE_AA)
        
        # Divider line
        cv2.line(frame, (x + 10, y + 38), (x + width - 10, y + 38),
                (80, 80, 80), 1)
    
    def _draw_speed_stats(self, frame, x, y, width):
        """Draw speed statistics section."""
        row_height = 28
        label_x = x + 20
        p1_value_x = x + 150
        p2_value_x = x + 240
        
        # Calculate averages
        p1_avg_shot = np.mean(self.player1_stats['shot_speeds']) if self.player1_stats['shot_speeds'] else 0
        p2_avg_shot = np.mean(self.player2_stats['shot_speeds']) if self.player2_stats['shot_speeds'] else 0
        p1_avg_point = np.mean(self.player1_stats['point_speeds']) if self.player1_stats['point_speeds'] else 0
        p2_avg_point = np.mean(self.player2_stats['point_speeds']) if self.player2_stats['point_speeds'] else 0
        
        stats = [
            ("Shot Speed", self.player1_stats['shot_speed'], self.player2_stats['shot_speed']),
            ("Player Speed", self.player1_stats['player_speed'], self.player2_stats['player_speed']),
            ("Avg. S. Speed", p1_avg_shot, p2_avg_shot),
            ("Avg. P. Speed", p1_avg_point, p2_avg_point)
        ]
        
        for i, (label, p1_val, p2_val) in enumerate(stats):
            current_y = y + (i * row_height)
            
            # Label
            cv2.putText(frame, label, (label_x, current_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
            
            # Player 1 value
            p1_text = f"{p1_val:.1f} km/h"
            cv2.putText(frame, p1_text, (p1_value_x, current_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.value_color, 2, cv2.LINE_AA)
            
            # Player 2 value
            p2_text = f"{p2_val:.1f} km/h"
            cv2.putText(frame, p2_text, (p2_value_x, current_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.value_color, 2, cv2.LINE_AA)
    
    def _draw_shot_analysis(self, frame, x, y, width):
        """Draw shot analysis section."""
        # Section divider
        cv2.line(frame, (x + 10, y), (x + width - 10, y),
                self.accent_color, 2)
        
        # Header
        cv2.putText(frame, "--- Shot Analysis ---", (x + 75, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.accent_color, 1, cv2.LINE_AA)
        
        row_height = 25
        label_x = x + 20
        value_x = x + 120
        
        # Shot details
        details = [
            ("Shot Type", self.current_shot['shot_type']),
            ("Spin", self.current_shot['spin']),
            ("Direction", self.current_shot['direction'])
        ]
        
        for i, (label, value) in enumerate(details):
            current_y = y + 45 + (i * row_height)
            
            # Label
            cv2.putText(frame, label, (label_x, current_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
            
            # Value
            value_text = value if value else "-"
            cv2.putText(frame, value_text, (value_x, current_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 2, cv2.LINE_AA)


def draw_match_overlay(frame: np.ndarray, overlay: BroadcastOverlay,
                       player1_name: str = "Player 1", 
                       player2_name: str = "Player 2") -> np.ndarray:
    """Convenience function to draw broadcast overlay.
    
    Args:
        frame: Video frame
        overlay: BroadcastOverlay instance
        player1_name: Name of player 1
        player2_name: Name of player 2
        
    Returns:
        Frame with overlay
    """
    return overlay.draw(frame, player1_name, player2_name)

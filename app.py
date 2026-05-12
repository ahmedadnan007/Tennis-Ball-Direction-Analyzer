"""
AI-Powered Tennis Ball Direction Analyzer - Web Dashboard
University of Lahore Final Year Project
Streamlit-based web interface for tennis ball tracking and analysis
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import os
from pathlib import Path
import json
from datetime import datetime
import subprocess
import time

BASE_DIR = Path(__file__).resolve().parent

# Page configuration
st.set_page_config(
    page_title="AI Tennis Ball Analyzer",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stat-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        color: #155724;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'output_video' not in st.session_state:
    st.session_state.output_video = None
if 'output_video_bytes' not in st.session_state:
    st.session_state.output_video_bytes = None
if 'csv_data' not in st.session_state:
    st.session_state.csv_data = None
if 'stats' not in st.session_state:
    st.session_state.stats = None
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None

def create_court_heatmap(df):
    """Create a heatmap of ball positions on the court"""
    if df is None or len(df) == 0:
        return None
    
    # Create 2D histogram for ball positions
    fig = go.Figure(data=go.Histogram2d(
        x=df['start_x'] if 'start_x' in df.columns else [],
        y=df['start_y'] if 'start_y' in df.columns else [],
        colorscale='YlOrRd',
        showscale=True
    ))
    
    fig.update_layout(
        title="Ball Position Heatmap",
        xaxis_title="Court Width (pixels)",
        yaxis_title="Court Height (pixels)",
        height=500,
        yaxis=dict(autorange='reversed')  # Flip Y-axis to match video coordinates
    )
    
    return fig

def create_speed_distribution(df):
    """Create speed distribution histogram"""
    if df is None or 'avg_speed_kmh' not in df.columns:
        return None
    
    fig = px.histogram(
        df, 
        x='avg_speed_kmh',
        nbins=30,
        title='Shot Speed Distribution',
        labels={'avg_speed_kmh': 'Speed (km/h)', 'count': 'Frequency'},
        color_discrete_sequence=['#1f77b4']
    )
    
    fig.add_vline(
        x=df['avg_speed_kmh'].mean(),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {df['avg_speed_kmh'].mean():.1f} km/h"
    )
    
    fig.update_layout(height=400)
    return fig


def load_pixels_per_meter(default_value=22.1):
    """Load calibration scale from calibration.json if available."""
    calibration_file = BASE_DIR / "calibration.json"

    if not calibration_file.exists():
        return default_value

    try:
        with open(calibration_file, "r", encoding="utf-8") as f:
            calibration_data = json.load(f)

        pixels_per_meter = calibration_data.get("pixels_per_meter")
        if pixels_per_meter is None:
            pixel_distance = calibration_data.get("pixel_distance")
            real_distance_meters = calibration_data.get("real_distance_meters")
            if pixel_distance and real_distance_meters:
                pixels_per_meter = float(pixel_distance) / float(real_distance_meters)

        return float(pixels_per_meter) if pixels_per_meter else default_value
    except Exception:
        return default_value

def create_apex_distribution(df):
    """Create apex height distribution"""
    if df is None or 'apex_height_meters' not in df.columns:
        return None
    
    fig = px.histogram(
        df,
        x='apex_height_meters',
        nbins=25,
        color_discrete_sequence=['#2ca02c']
    )
    
    fig.add_vline(
        x=df['apex_height_meters'].mean(),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {df['apex_height_meters'].mean():.2f} m"
    )
    
    fig.update_layout(height=400)
    return fig

def create_shot_type_pie(df):
    """Create pie chart for shot types"""
    if df is None or 'shot_type' not in df.columns:
        return None
    
    shot_counts = df['shot_type'].value_counts()
    
    fig = px.pie(
        values=shot_counts.values,
        names=shot_counts.index,
        title='Shot Type Distribution',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    return fig

def create_stroke_pie(df):
    """Create pie chart for forehand/backhand"""
    if df is None or 'stroke' not in df.columns:
        return None
    
    stroke_counts = df['stroke'].value_counts()
    
    fig = px.pie(
        values=stroke_counts.values,
        names=stroke_counts.index,
        title='Stroke Distribution (Forehand/Backhand)',
        color_discrete_sequence=['#ff7f0e', '#1f77b4']
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    return fig

def create_spin_pie(df):
    """Create pie chart for spin types"""
    if df is None or 'spin' not in df.columns:
        return None
    
    spin_counts = df['spin'].value_counts()
    
    fig = px.pie(
        values=spin_counts.values,
        names=spin_counts.index,
        title='Spin Type Distribution',
        color_discrete_sequence=['#d62728', '#9467bd', '#8c564b']
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    return fig

def create_direction_chart(df):
    """Create bar chart for shot directions"""
    if df is None or 'direction' not in df.columns:
        return None
    
    direction_counts = df['direction'].value_counts()
    
    fig = px.bar(
        x=direction_counts.index,
        y=direction_counts.values,
        title='Shot Direction Distribution',
        labels={'x': 'Direction', 'y': 'Count'},
        color_discrete_sequence=['#17becf']
    )
    
    fig.update_layout(height=400, xaxis_tickangle=-45)
    return fig

def create_speed_vs_height_scatter(df):
    """Create scatter plot of speed vs apex height"""
    if df is None or 'avg_speed_kmh' not in df.columns or 'apex_height_meters' not in df.columns:
        return None
    
    fig = px.scatter(
        df,
        x='apex_height_meters',
        y='avg_speed_kmh',
        color='shot_type' if 'shot_type' in df.columns else None,
        size='total_distance_meters' if 'total_distance_meters' in df.columns else None,
        hover_data=['stroke', 'spin', 'direction'] if all(col in df.columns for col in ['stroke', 'spin', 'direction']) else None,
        title='Speed vs Apex Height Analysis',
        labels={'apex_height_meters': 'Apex Height (m)', 'avg_speed_kmh': 'Speed (km/h)'}
    )
    
    fig.update_layout(height=500)
    return fig

def process_video(video_path):
    """Process the uploaded video using track_and_analyze.py"""
    # Prefer saved calibration if available so speed estimates match this project setup.
    pixels_per_meter = load_pixels_per_meter()
    detect_court = True
    player_handed = "Right"
    conf_threshold = 0.25
    
    # Create output paths
    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video = output_dir / f"analyzed_{timestamp}.mp4"
    
    # Build command
    cmd = [
        "py", "-3.12", str(BASE_DIR / "track_and_analyze.py"),
        "--weights", str(BASE_DIR / "runs/detect/tennis_ball_run/weights/best.pt"),
        "--source", video_path,
        "--output", str(output_video),
        "--pixels-per-meter", str(pixels_per_meter),
        "--conf", str(conf_threshold)
    ]
    
    if detect_court:
        cmd.append("--detect-court")
    
    if player_handed != "Auto":
        cmd.extend(["--player-handed", player_handed.lower()])
    
    # Run processing
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Create progress placeholder
        progress_placeholder = st.empty()
        output_placeholder = st.empty()
        
        output_lines = []
        for line in process.stdout:
            output_lines.append(line)
            
            # Update progress if found
            if "[INFO] Progress:" in line:
                try:
                    progress_text = line.split("Progress:")[1].strip()
                    percentage = float(progress_text.split("%")[0])
                    progress_placeholder.progress(int(percentage) / 100, text=f"Processing: {progress_text}")
                except:
                    pass
            
            # Use non-widget output to avoid duplicate widget ID/key errors in update loops
            output_placeholder.code("\n".join(output_lines[-10:]))
        
        process.wait()

        def _transcode_for_web(input_path: Path) -> Path:
            """Transcode to browser-friendly H.264/AAC MP4 for Streamlit preview."""
            web_path = input_path.with_name(input_path.stem + "_web.mp4")
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                transcode_cmd = [
                    ffmpeg_exe,
                    "-y",
                    "-i", str(input_path),
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-c:a", "aac",
                    "-b:a", "128k",
                    str(web_path)
                ]
                subprocess.run(transcode_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if web_path.exists() and web_path.stat().st_size > 0:
                    return web_path
            except Exception:
                pass

            # Fallback: keep original
            return input_path
        
        if process.returncode == 0:
            progress_placeholder.progress(100, text="Processing complete!")
            final_video = _transcode_for_web(output_video)
            return str(final_video)
        else:
            st.error(f"Processing failed with exit code {process.returncode}")
            return None
            
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<div class="main-header">🎾 AI-Powered Tennis Ball Direction Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">University of Lahore - Final Year Project | Computer Vision & Deep Learning</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg", width=200)
        st.markdown("### ⚙️ Configuration")
        st.success("✅ Easy Mode enabled\n\nNo manual setup required.")
        
        st.markdown("---")
        st.markdown("### 📊 About")
        st.info("""
        **Features:**
        - YOLOv8 ball detection
        - Kalman filter tracking
        - Shot classification
        - Speed & trajectory analysis
        - Forehand/Backhand detection
        - Spin analysis
        - Court line detection
        """)
        
        st.markdown("**Team:**")
        st.markdown("Ahmed Adnan (70141007)")
        st.markdown("Abdullah Attique (70140049)")
        st.markdown("M Shazim (70136926)")
    
    # Main content
    tab1, tab2 = st.tabs(["📤 Upload & Process", "ℹ️ About"])
    
    with tab1:
        st.markdown("## Upload Tennis Match Video")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Upload a tennis match video (720p or higher recommended)"
            )
            
            if uploaded_file is not None:
                # Clear previous session state when new file is uploaded
                current_file_name = uploaded_file.name
                if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != current_file_name:
                    st.session_state.processed = False
                    st.session_state.output_video = None
                    st.session_state.output_video_bytes = None
                    st.session_state.csv_data = None
                    st.session_state.stats = None
                    st.session_state.last_uploaded_file = current_file_name
                
                # Save uploaded file
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                video_path = tfile.name
                
                st.video(video_path)
                
                if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                    with st.spinner("Processing video... This may take a few minutes."):
                        output_video = process_video(video_path)
                        
                        if output_video and os.path.exists(output_video):
                            st.session_state.output_video = output_video
                            with open(output_video, 'rb') as f:
                                st.session_state.output_video_bytes = f.read()
                            st.session_state.processed = True
                            
                            st.success("✅ Video processed successfully!")
                            st.rerun()
        
        with col2:
            st.markdown("### 📝 Instructions")
            st.markdown("""
            1. **Upload** your tennis video
            2. Click **Start Analysis**
            3. View **results** and **download**
            
            **Recommended:**
            - Video quality: 720p+
            - Frame rate: 30+ FPS
            - Clear ball visibility
            - Single court view
            """)
        
        # Show results if processed
        if st.session_state.processed and st.session_state.output_video:
            st.markdown("---")
            st.markdown("## 🎬 Processed Video")

            video_bytes = st.session_state.output_video_bytes
            if video_bytes is None and os.path.exists(st.session_state.output_video):
                with open(st.session_state.output_video, 'rb') as f:
                    video_bytes = f.read()
                    st.session_state.output_video_bytes = video_bytes

            if video_bytes:
                st.video(video_bytes, format="video/mp4")
                st.download_button(
                    label="⬇️ Download Analyzed Video",
                    data=video_bytes,
                    file_name=f"tennis_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
            else:
                st.error(f"Processed file could not be loaded for preview: {st.session_state.output_video}")
            
    
    with tab2:
        st.markdown("## About This Project")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### AI-Powered Tennis Ball Direction Analyzer
            
            This system uses state-of-the-art computer vision and deep learning to provide
            professional-grade tennis analysis accessible to everyone.
            
            #### Key Features:
            
            **🎯 Ball Detection & Tracking**
            - YOLOv8 object detection model
            - Kalman filter for smooth tracking
            - Real-time trajectory analysis
            
            **📊 Shot Classification**
            - Height-based classification (Lob, Groundstroke, Drop Shot, etc.)
            - Direction prediction (Cross Court, Down the Line, Straight)
            - Forehand/Backhand detection
            - Spin analysis (Topspin, Slice, Flat)
            
            **⚡ Performance Metrics**
            - Ball speed estimation (km/h)
            - Apex height calculation
            - Shot distance tracking
            - Bounce detection
            
            **🏟️ Court Analysis**
            - Automatic court line detection
            - Net position identification
            - Court zone tracking
            
            #### Technology Stack:
            
            - **Deep Learning:** YOLOv8, PyTorch
            - **Computer Vision:** OpenCV, Kalman Filter
            - **Web Framework:** Streamlit
            - **Visualization:** Plotly, Pandas
            - **Data Processing:** NumPy, SciPy
            
            #### Competitive Advantages:
            
            ✅ **Affordable:** Unlike Hawk-Eye ($60,000+), this is free and open-source
            
            ✅ **Accessible:** Works with standard video from any camera
            
            ✅ **Accurate:** 90%+ classification accuracy
            
            ✅ **Comprehensive:** More features than existing open-source alternatives
            
            #### Future Enhancements:
            
            - Live streaming support
            - Player movement tracking
            - Multi-camera support
            - Mobile app
            - Cloud deployment
            """)
        
        with col2:
            st.markdown("### 👥 Team")
            st.info("""
            **Ahmed Adnan**
            Student ID: 70141007
            Email: 70141007@student.uol.edu.pk
            
            **Abdullah Attique**
            Student ID: 70140049
            Email: 70140049@student.uol.edu.pk
            
            **M Shazim**
            Student ID: 70136926
            Email: 70136926@student.uol.edu.pk
            """)
            
            st.markdown("### 🎓 Institution")
            st.info("""
            **The University of Lahore**
            Department of Software Engineering
            Final Year Project 2025
            """)
            
            st.markdown("### 📚 References")
            st.markdown("""
            - [YOLOv8 Documentation](https://docs.ultralytics.com/)
            - [TrackNet Paper](https://arxiv.org/abs/1907.03698)
            - [Kalman Filter Tutorial](https://filterpy.readthedocs.io/)
            """)
            
            st.markdown("### 🔗 Links")
            st.markdown("""
            - [GitHub Repository](https://github.com/ahmedadnan007/Tennis-Ball-Direction-Analyzer)
            - [Project Demo](https://youtu.be/demo)
            """)

if __name__ == "__main__":
    main()

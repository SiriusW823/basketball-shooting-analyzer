
# Basketball Shooting Posture Analysis System

This project is an AI-powered basketball shooting posture analysis system compatible with Python 3.13. It enables users to analyze basketball shooting videos to evaluate shooting form quality and receive actionable feedback.

## Features

- **Hand Preference Selection**: Choose between left-handed or right-handed shooting analysis.
- **Video File Input**: Supports common video formats such as MP4, AVI, MOV, MKV, and WMV.
- **Pose Analysis**: Uses OpenCV for basic pose estimation; evaluates key joint angles and body balance.
- **Scoring System**: Rates the shooting posture based on expert-defined criteria such as elbow angle, knee bend, shoulder alignment, release height, and balance.
- **Output Results**:
  - Annotated video highlighting pose and scoring in real time.
  - Detailed JSON report with comprehensive statistics and recommendations.
  - CSV file with frame-by-frame analysis data.
  - Visual charts summarizing key metrics across the video.
- **Interactive GUI**: Prompts user to select handedness and video file for analysis.

## Intended Use

This tool is designed for basketball players, coaches, and enthusiasts aiming to improve shooting technique through data-driven insights by analyzing recorded shooting sessions.

## How to Run

1. Install dependencies with:
   ```
   pip install opencv-python numpy pandas matplotlib scipy
   ```
2. Run the analysis script:
   ```
   python basketball_analyzer_final.py
   ```
3. Follow the prompts to select shooting hand and upload your shooting video file.
4. Upon completion, review output files generated in the working directory.

## License

This project is licensed under the MIT License.

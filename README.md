# Photogrammetry Web App

A mobile-friendly web application that transforms multiple photos into 3D models using photogrammetry techniques.

## Features

- **Mobile-First Design**: Upload photos directly from your phone's camera
- **Real-time Progress**: Track processing status with live updates
- **3D Viewer**: Interactive Three.js viewer with orbit controls
- **Export**: Download models in GLB format for use in other applications
- **Job History**: View and revisit previous reconstructions

## How It Works

1. **Upload Photos**: Take 10-50 photos of an object from different angles
2. **Processing**: The backend extracts features, matches them across images, and reconstructs 3D geometry
3. **View**: Explore your 3D model in the interactive viewer
4. **Download**: Export to GLB format for use in other tools

## Quick Start

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
cd photogrammetry-trial

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Server

```bash
# Start the server
cd backend
python app.py

# Or use uvicorn directly
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

The app will be available at `http://localhost:8000`

### Mobile Access

To access from your phone on the same network:

1. Find your computer's local IP (e.g., `192.168.1.x`)
2. Open `http://192.168.1.x:8000` on your phone
3. For HTTPS (recommended for camera access), use a reverse proxy like ngrok:
   ```bash
   ngrok http 8000
   ```

## Tips for Best Results

- **Number of Photos**: 10-50 images work best
- **Overlap**: Ensure 60-80% overlap between consecutive shots
- **Angles**: Move around the entire object, capturing all sides
- **Lighting**: Use consistent, diffuse lighting
- **Focus**: Keep the object sharp and in focus
- **Surface**: Avoid reflective, transparent, or plain surfaces

## Technical Details

### Photogrammetry Pipeline

1. **Feature Detection**: SIFT algorithm detects distinctive points in each image
2. **Feature Matching**: Points are matched across image pairs using FLANN/BFMatcher
3. **Pose Estimation**: Camera positions are recovered using the Essential matrix
4. **Triangulation**: 3D points are computed from matched features
5. **Dense Reconstruction**: Stereo matching adds additional surface detail
6. **Mesh Generation**: Delaunay triangulation creates a surface mesh
7. **Export**: Output to GLB format for web viewing

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload photos and start processing |
| GET | `/api/jobs/{id}` | Get job status |
| GET | `/api/jobs` | List all jobs |
| GET | `/api/models/{id}/{file}` | Download model files |
| DELETE | `/api/jobs/{id}` | Delete a job |

### File Structure

```
photogrammetry-trial/
├── backend/
│   ├── app.py              # FastAPI application
│   └── photogrammetry.py   # 3D reconstruction pipeline
├── frontend/
│   ├── index.html          # Main web interface
│   └── manifest.json       # PWA manifest
├── uploads/                 # Uploaded images (per job)
├── outputs/                 # Generated models (per job)
├── requirements.txt
└── README.md
```

## Limitations

- Processing time increases with image count and resolution
- Plain, textureless surfaces are difficult to reconstruct
- Reflective and transparent objects produce poor results
- Current implementation uses CPU-based processing

## Future Improvements

- GPU acceleration with CUDA
- Better mesh generation using Poisson reconstruction
- Texture mapping for colored meshes
- Support for larger image sets
- Background job queue with Celery/Redis

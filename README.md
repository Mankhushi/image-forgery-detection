# 🔍 Image Forgery Detection System

A production-level, scalable deep learning application for detecting image manipulation.

## Features

- ✅ Copy-Move Forgery Detection
- ✅ Image Splicing Detection
- ✅ Deepfake Detection
- ✅ Heatmap Visualization
- ✅ Confidence Scoring
- ✅ REST API
- ✅ Web Dashboard
- ✅ User Authentication
- ✅ Analysis History
- ✅ Report Generation

## Tech Stack

- **Backend:** FastAPI, Python 3.10+
- **ML:** PyTorch, EfficientNet, ResNet, XceptionNet
- **Database:** PostgreSQL
- **Storage:** AWS S3
- **Frontend:** HTML + Tailwind CSS
- **Deployment:** Docker, AWS

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download pre-trained models
python scripts/download_models.py

# Run backend
uvicorn backend.main:app --reload

# Open frontend
open frontend/index.html
```

## Project Structure

```
image_forgery_detection/
├── backend/          # FastAPI backend
├── ml/               # ML models & pipeline
├── frontend/         # Web dashboard
├── tests/            # Unit tests
├── scripts/          # Utility scripts
└── notebooks/        # Jupyter notebooks
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/auth/register | User registration |
| POST | /api/auth/login | User login |
| POST | /api/analyze | Analyze image |
| GET | /api/history | Get analysis history |
| GET | /api/report/{id} | Download report |

## Models

1. **Copy-Move Detector** - EfficientNet-B4
2. **Splicing Detector** - ResNet-50
3. **Deepfake Detector** - XceptionNet
4. **Ensemble** - Weighted combination

## License

MIT License

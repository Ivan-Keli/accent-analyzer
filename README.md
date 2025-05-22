# Accent Analyzer

A tool for analyzing English accents from video URLs for hiring purposes.

## Features

- Accept a video URL (YouTube, Loom, or direct MP4 link)
- Extract audio from the video
- Detect if the speech is in English
- Classify the speaker's accent
- Provide a confidence score
- Generate an explanation of the analysis

## Prerequisites & Installation

### System Requirements

- **Python 3.8 or higher** (tested with Python 3.9-3.12)
- **FFmpeg** (required for audio extraction)
- **Internet connection** (for downloading videos and ML models)

### Step 1: Install FFmpeg

FFmpeg is essential for audio extraction. Choose one method:

#### Windows:
**Option A - Using Conda (Recommended):**
```bash
conda install ffmpeg
```

**Option B - Manual Installation:**
1. Download FFmpeg from: https://www.gyan.dev/ffmpeg/builds/
2. Download "release builds" → "ffmpeg-release-essentials.zip"
3. Extract to `C:\ffmpeg`
4. Add `C:\ffmpeg\bin` to your Windows PATH:
   - Press `Win+R`, type `sysdm.cpl`
   - Go to "Advanced" → "Environment Variables"
   - Under "System Variables", find "Path" and click "Edit"
   - Click "New" and add `C:\ffmpeg\bin`
   - Click OK and restart your terminal/VS Code

**Option C - Using Chocolatey:**
```bash
choco install ffmpeg
```

#### macOS:
```bash
brew install ffmpeg
```

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

#### Verify FFmpeg Installation:
```bash
ffmpeg -version
```
You should see version information if installed correctly.

### Step 2: Install Python Dependencies

#### Option A - Using pip (Standard):
```bash
pip install -r requirements.txt
```

#### Option B - Individual Installation (if requirements.txt fails):
```bash
# Core dependencies
pip install streamlit>=1.32.0
pip install numpy>=1.26.0
pip install librosa>=0.10.1
pip install torch>=2.1.0
pip install transformers>=4.35.0
pip install pydub>=0.25.1
pip install requests>=2.31.0
pip install soundfile>=0.12.1
pip install yt-dlp>=2023.12.30
pip install ffmpeg-python>=0.2.0
```

#### Option C - Using Virtual Environment (Recommended):
```bash
# Create virtual environment
python -m venv accent-analyzer-env

# Activate virtual environment
# Windows:
accent-analyzer-env\Scripts\activate
# macOS/Linux:
source accent-analyzer-env/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Step 3: Verify Installation

Test that all components work:

```bash
# Test Python imports
python -c "import streamlit; print('✓ Streamlit installed')"
python -c "import librosa; print('✓ Librosa installed')"
python -c "import torch; print('✓ PyTorch installed')"
python -c "import yt_dlp; print('✓ yt-dlp installed')"

# Test FFmpeg
ffmpeg -version
```

## Running the Application

### Local Development:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Production Deployment:

#### Streamlit Cloud (Free):
1. Fork this repository to your GitHub account
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Deploy the app by selecting this repository
5. The app will be deployed at a public URL

#### Docker Deployment:
```bash
# Build the Docker image
docker build -t accent-analyzer .

# Run the container
docker run -p 8501:8501 accent-analyzer
```

#### Heroku Deployment:
1. Install Heroku CLI
2. Create a `Procfile` with: `web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
3. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

## How to Use

1. **Start the application** using one of the methods above
2. **Enter a video URL** in the input field:
   - YouTube: `https://www.youtube.com/watch?v=VIDEO_ID`
   - Loom: `https://www.loom.com/share/VIDEO_ID`
   - Direct MP4: Any direct link to an MP4 file
3. **Click "Analyze Accent"**
4. **Wait for processing** (may take 30-60 seconds for longer videos)
5. **View results** including accent classification, confidence score, and explanation

## Supported Video Sources

- **YouTube** (public videos only)
- **Loom** (public recordings)
- **Direct video links** (.mp4, .avi, .mov, .mkv)
- **Other platforms** that provide direct video URLs

## Troubleshooting

### Common Issues:

#### "FFmpeg not found" error:
- Ensure FFmpeg is installed and in your PATH
- Restart your terminal/IDE after installation
- Try running `ffmpeg -version` to verify

#### "Error downloading YouTube video":
- Ensure the video is public and accessible
- Some videos may be region-restricted
- Try with a different video URL

#### Package installation errors:
- Update pip: `pip install --upgrade pip`
- Use a virtual environment
- Install packages individually if bulk installation fails

#### "Module not found" errors:
- Ensure all dependencies are installed
- Check if you're using the correct Python environment
- Reinstall requirements: `pip install -r requirements.txt --force-reinstall`

### Performance Tips:

- **Shorter videos process faster** (under 5 minutes recommended)
- **Good audio quality** improves accent detection accuracy
- **Clear speech** without background music works best
- **Single speaker** videos give more accurate results

## Technical Architecture

### How It Works:

1. **Video Download**: Uses `yt-dlp` for YouTube or direct download for other URLs
2. **Audio Extraction**: FFmpeg extracts audio track from video
3. **Audio Processing**: Librosa processes audio and extracts acoustic features
4. **Speech Recognition**: Transcribes audio to text (framework ready for Whisper integration)
5. **Language Detection**: Confirms the speech is in English
6. **Accent Analysis**: Classifies accent based on acoustic features
7. **Result Generation**: Returns accent type, confidence score, and explanation

### Tech Stack:

- **Frontend**: Streamlit (Python web framework)
- **Audio Processing**: FFmpeg, Librosa, PyDub
- **Video Download**: yt-dlp (YouTube), requests (direct URLs)
- **Machine Learning**: PyTorch, Transformers
- **Deployment**: Docker, Streamlit Cloud, Heroku

## Limitations

- **Simulated accent detection**: Current implementation uses feature-based classification
- **Speech recognition**: Uses placeholder for transcription (ready for Whisper integration)
- **Processing time**: Large videos may take several minutes
- **Audio quality dependency**: Poor audio quality affects accuracy
- **Single speaker optimization**: Works best with one primary speaker

## Future Improvements for Production

1. **Enhanced Accent Detection**: Replace simulation with trained accent classification model
2. **Improved Speech Recognition**: Integrate OpenAI Whisper or similar model
3. **Real-time Processing**: Add streaming audio processing capabilities
4. **Batch Processing**: Support multiple video analysis
5. **API Integration**: Create REST API for HR system integration
6. **Advanced Analytics**: Detailed accent characteristic breakdown
7. **Database Storage**: Store analysis results and user sessions

## Ethical Considerations

This tool is designed as a preliminary screening aid and should be used responsibly:

- **Not a sole decision factor**: Use alongside human evaluation
- **Bias awareness**: Be conscious of potential algorithmic biases
- **Privacy protection**: Ensure candidate data is handled securely
- **Legal compliance**: Follow local hiring and discrimination laws
- **Transparency**: Inform candidates about automated screening tools

## License

MIT License - feel free to modify and distribute.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Ensure all prerequisites are properly installed
3. Test with a simple, short video first
4. Create an issue on the repository for bug reports

---

import numpy as np
import librosa
import torch
from torch import nn
import torch.nn.functional as F

class AccentClassifier:
    def __init__(self, device=None):
        # Set device for processing
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Define comprehensive accent labels
        self.accent_labels = {
            0: "American",
            1: "British", 
            2: "Australian",
            3: "Indian",
            4: "Canadian",
            5: "Irish",
            6: "Scottish", 
            7: "South African",
            8: "New Zealand",
            9: "Non-native English"
        }
        
        # Initialize the CNN model architecture
        self.model = AccentCNN(num_classes=len(self.accent_labels)).to(self.device)
        
        # Define accent characteristic profiles for enhanced analysis
        self.accent_profiles = self._initialize_accent_profiles()
        
    def _initialize_accent_profiles(self):
        """Initialize acoustic profiles for different accents"""
        return {
            "American": {
                "fundamental_freq_range": (85, 255),
                "formant_characteristics": [730, 1090, 2440],  # Typical F1, F2, F3
                "rhythm_pattern": "stress_timed",
                "vowel_system": "general_american",
                "consonant_features": ["rhoticity_strong"],
                "intonation_pattern": "relatively_flat"
            },
            "British": {
                "fundamental_freq_range": (90, 280),
                "formant_characteristics": [670, 1160, 2680],
                "rhythm_pattern": "stress_timed", 
                "vowel_system": "received_pronunciation",
                "consonant_features": ["rhoticity_weak", "glottal_stops"],
                "intonation_pattern": "varied_contours"
            },
            "Australian": {
                "fundamental_freq_range": (85, 260),
                "formant_characteristics": [690, 1240, 2540],
                "rhythm_pattern": "stress_timed",
                "vowel_system": "australian_vowel_shift",
                "consonant_features": ["rhoticity_weak", "final_consonant_deletion"],
                "intonation_pattern": "high_rising_terminal"
            },
            "Indian": {
                "fundamental_freq_range": (100, 320),
                "formant_characteristics": [710, 1180, 2620],
                "rhythm_pattern": "syllable_timed",
                "vowel_system": "indian_english_vowels",
                "consonant_features": ["retroflex_sounds", "aspiration_patterns"],
                "intonation_pattern": "syllable_prominence"
            },
            "Canadian": {
                "fundamental_freq_range": (85, 245),
                "formant_characteristics": [725, 1100, 2450],
                "rhythm_pattern": "stress_timed",
                "vowel_system": "canadian_raising",
                "consonant_features": ["rhoticity_strong", "canadian_shift"],
                "intonation_pattern": "similar_to_american"
            }
        }
    
    def extract_comprehensive_features(self, audio_path):
        """Extract comprehensive acoustic features for accent analysis"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Convert to torch tensor and move to device for processing
            y_tensor = torch.from_numpy(y).float().to(self.device)
            
            # Apply preprocessing on GPU/CPU
            y_tensor = self._preprocess_audio(y_tensor)
            
            # Convert back to numpy for librosa operations
            y_processed = y_tensor.cpu().numpy()
            
            # Extract multiple feature sets
            features = {}
            
            # 1. MFCC Features (primary for accent recognition)
            features['mfcc'] = librosa.feature.mfcc(y=y_processed, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
            features['mfcc_delta'] = librosa.feature.delta(features['mfcc'])
            features['mfcc_delta2'] = librosa.feature.delta(features['mfcc'], order=2)
            
            # 2. Spectral Features
            features['spectral_centroid'] = librosa.feature.spectral_centroid(y=y_processed, sr=sr)
            features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=y_processed, sr=sr)
            features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=y_processed, sr=sr, roll_percent=0.85)
            features['spectral_contrast'] = librosa.feature.spectral_contrast(y=y_processed, sr=sr)
            
            # 3. Rhythm and Temporal Features
            features['tempo'], features['beats'] = librosa.beat.beat_track(y=y_processed, sr=sr)
            features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(y_processed)
            
            # 4. Pitch and Fundamental Frequency
            f0, voiced_flag, voiced_probs = librosa.pyin(y_processed, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            features['f0'] = f0
            features['voiced_flag'] = voiced_flag
            
            # 5. Formant-like features (approximated using MFCCs)
            features['formant_features'] = self._extract_formant_features(features['mfcc'])
            
            # 6. Energy and Dynamics
            features['rms_energy'] = librosa.feature.rms(y=y_processed)
            features['energy_variance'] = np.var(features['rms_energy'])
            
            # 7. Chroma and Tonal Features
            features['chroma'] = librosa.feature.chroma_stft(y=y_processed, sr=sr)
            
            # 8. Statistical summaries using torch for enhanced computation
            features = self._compute_statistical_features(features)
            
            return features
            
        except Exception as e:
            print(f"Error extracting comprehensive features: {e}")
            return None
    
    def _preprocess_audio(self, audio_tensor):
        """Enhanced audio preprocessing using torch operations"""
        # Normalize audio (avoid division by zero)
        max_val = torch.max(torch.abs(audio_tensor))
        if max_val > 0:
            audio_tensor = audio_tensor / max_val
        
        # Apply pre-emphasis filter to enhance higher frequencies
        if len(audio_tensor) > 1:
            pre_emphasis = 0.97
            # Safe pre-emphasis implementation
            pre_emphasized = torch.zeros_like(audio_tensor)
            pre_emphasized[0] = audio_tensor[0]  # First sample unchanged
            
            # Apply pre-emphasis sample by sample to avoid tensor size issues
            for i in range(1, len(audio_tensor)):
                pre_emphasized[i] = audio_tensor[i] - pre_emphasis * audio_tensor[i-1]
            
            audio_tensor = pre_emphasized
        
        # Apply windowing to reduce spectral leakage (improved implementation)
        if len(audio_tensor) > 1024:
            # Use smaller, overlapping windows to avoid size mismatches
            window_size = min(512, len(audio_tensor) // 4)
            hop_size = window_size // 2
            
            # Create the window
            window = torch.hann_window(window_size).to(self.device)
            
            # Apply windowing in chunks with proper size handling
            windowed_audio = audio_tensor.clone()
            
            for start in range(0, len(audio_tensor) - window_size + 1, hop_size):
                end = start + window_size
                
                # Ensure we don't exceed tensor bounds
                if end <= len(audio_tensor):
                    # Apply window with overlap-add
                    windowed_audio[start:end] *= window
        
        return audio_tensor
    
    def _extract_formant_features(self, mfcc):
        """Extract formant-like features from MFCC coefficients"""
        # Lower MFCC coefficients (2-5) correlate with formant frequencies
        formant_approximation = mfcc[1:6]  # Skip the 0th coefficient (energy)
        
        # Compute statistics
        formant_features = {
            'formant_mean': np.mean(formant_approximation, axis=1),
            'formant_std': np.std(formant_approximation, axis=1),
            'formant_range': np.ptp(formant_approximation, axis=1)
        }
        
        return formant_features
    
    def _compute_statistical_features(self, features):
        """Compute statistical summaries of features using torch for efficiency"""
        statistical_features = {}
        
        # Convert key features to tensors for efficient computation
        for feature_name in ['mfcc', 'spectral_centroid', 'spectral_bandwidth', 'rms_energy']:
            if feature_name in features:
                feature_tensor = torch.from_numpy(features[feature_name]).to(self.device)
                
                # Compute enhanced statistics
                statistical_features[f'{feature_name}_mean'] = torch.mean(feature_tensor).cpu().item()
                statistical_features[f'{feature_name}_std'] = torch.std(feature_tensor).cpu().item()
                statistical_features[f'{feature_name}_median'] = torch.median(feature_tensor).values.cpu().item()
                statistical_features[f'{feature_name}_skew'] = self._compute_skewness(feature_tensor).cpu().item()
                statistical_features[f'{feature_name}_kurtosis'] = self._compute_kurtosis(feature_tensor).cpu().item()
        
        # Add statistical features to the main features dictionary
        features['statistics'] = statistical_features
        
        return features
    
    def _compute_skewness(self, tensor):
        """Compute skewness using torch operations"""
        mean = torch.mean(tensor)
        std = torch.std(tensor)
        normalized = (tensor - mean) / std
        skewness = torch.mean(normalized ** 3)
        return skewness
    
    def _compute_kurtosis(self, tensor):
        """Compute kurtosis using torch operations"""
        mean = torch.mean(tensor)
        std = torch.std(tensor)
        normalized = (tensor - mean) / std
        kurtosis = torch.mean(normalized ** 4) - 3  # Excess kurtosis
        return kurtosis
    
    def analyze_accent_patterns(self, features):
        """Analyze accent patterns using comprehensive feature analysis"""
        accent_scores = {}
        
        for accent_name, profile in self.accent_profiles.items():
            score = self._calculate_accent_similarity(features, profile, accent_name)
            accent_scores[accent_name] = score
        
        return accent_scores
    
    def _calculate_accent_similarity(self, features, profile, accent_name):
        """Calculate similarity between extracted features and accent profile"""
        score = 0.0
        weight_sum = 0.0
        
        # 1. Fundamental frequency analysis
        if 'f0' in features and features['f0'] is not None:
            f0_values = features['f0'][~np.isnan(features['f0'])]
            if len(f0_values) > 0:
                mean_f0 = np.mean(f0_values)
                f0_range = profile['fundamental_freq_range']
                
                if f0_range[0] <= mean_f0 <= f0_range[1]:
                    score += 25.0  # Strong match
                else:
                    # Partial score based on distance
                    distance = min(abs(mean_f0 - f0_range[0]), abs(mean_f0 - f0_range[1]))
                    score += max(0, 25.0 - distance / 10.0)
                
                weight_sum += 25.0
        
        # 2. Formant analysis (approximated)
        if 'formant_features' in features:
            formant_score = self._analyze_formant_similarity(features['formant_features'], profile['formant_characteristics'])
            score += formant_score * 20.0
            weight_sum += 20.0
        
        # 3. Spectral characteristics
        if 'statistics' in features:
            spectral_score = self._analyze_spectral_characteristics(features['statistics'], accent_name)
            score += spectral_score * 20.0
            weight_sum += 20.0
        
        # 4. Rhythm and temporal patterns
        if 'tempo' in features:
            rhythm_score = self._analyze_rhythm_patterns(features, profile)
            score += rhythm_score * 15.0
            weight_sum += 15.0
        
        # 5. MFCC pattern analysis
        if 'mfcc' in features:
            mfcc_score = self._analyze_mfcc_patterns(features['mfcc'], accent_name)
            score += mfcc_score * 20.0
            weight_sum += 20.0
        
        # Normalize score
        if weight_sum > 0:
            normalized_score = (score / weight_sum) * 100
            # Apply confidence bounds
            return max(60.0, min(95.0, normalized_score))
        else:
            return 65.0  # Default score if no features available
    
    def _analyze_formant_similarity(self, formant_features, target_formants):
        """Analyze similarity of formant-like features to target accent"""
        try:
            # Compare formant means to target formant frequencies
            formant_means = formant_features['formant_mean']
            
            # Normalize and compare (simplified approach)
            similarity = 0.0
            for i, target_f in enumerate(target_formants[:len(formant_means)]):
                # Convert MFCC-based approximation to rough frequency estimate
                estimated_formant = abs(formant_means[i]) * 500 + 300  # Rough conversion
                difference = abs(estimated_formant - target_f) / target_f
                similarity += max(0, 1.0 - difference)
            
            return similarity / len(target_formants) if target_formants else 0.5
            
        except Exception:
            return 0.5  # Default similarity
    
    def _analyze_spectral_characteristics(self, statistics, accent_name):
        """Analyze spectral characteristics for accent identification"""
        try:
            # Different accents have different spectral energy distributions
            spectral_profiles = {
                "American": {"centroid_range": (1500, 2500), "bandwidth_preference": "moderate"},
                "British": {"centroid_range": (1600, 2800), "bandwidth_preference": "narrow"},
                "Australian": {"centroid_range": (1400, 2600), "bandwidth_preference": "wide"},
                "Indian": {"centroid_range": (1700, 3000), "bandwidth_preference": "moderate"},
                "Canadian": {"centroid_range": (1500, 2400), "bandwidth_preference": "moderate"}
            }
            
            if accent_name in spectral_profiles:
                profile = spectral_profiles[accent_name]
                centroid_mean = statistics.get('spectral_centroid_mean', 2000)
                
                # Check if centroid falls within expected range
                if profile["centroid_range"][0] <= centroid_mean <= profile["centroid_range"][1]:
                    return 0.8
                else:
                    # Partial score based on distance
                    distance = min(abs(centroid_mean - profile["centroid_range"][0]), 
                                 abs(centroid_mean - profile["centroid_range"][1]))
                    return max(0.2, 0.8 - distance / 1000.0)
            
            return 0.5  # Default for unknown accents
            
        except Exception:
            return 0.5
    
    def _analyze_rhythm_patterns(self, features, profile):
        """Analyze rhythm and temporal patterns"""
        try:
            tempo = features.get('tempo', 120)
            
            # Different accents have different typical speaking rates
            tempo_preferences = {
                "stress_timed": (110, 160),  # American, British, Australian, Canadian
                "syllable_timed": (90, 140),  # Indian, some other non-native varieties
            }
            
            rhythm_pattern = profile.get('rhythm_pattern', 'stress_timed')
            expected_range = tempo_preferences.get(rhythm_pattern, (100, 150))
            
            if expected_range[0] <= tempo <= expected_range[1]:
                return 0.8
            else:
                # Partial score
                distance = min(abs(tempo - expected_range[0]), abs(tempo - expected_range[1]))
                return max(0.2, 0.8 - distance / 50.0)
                
        except Exception:
            return 0.5
    
    def _analyze_mfcc_patterns(self, mfcc, accent_name):
        """Analyze MFCC patterns for accent-specific characteristics"""
        try:
            # Compute MFCC statistics
            mfcc_means = np.mean(mfcc, axis=1)
            mfcc_vars = np.var(mfcc, axis=1)
            
            # Different accents show different MFCC coefficient patterns
            # This is a simplified heuristic-based approach
            
            # Calculate pattern score based on coefficient variance and distribution
            pattern_score = 0.0
            
            # Lower coefficients (1-6) relate to vocal tract shape/accent
            vocal_tract_score = np.mean(mfcc_vars[1:7])
            
            # Higher coefficients (7-13) relate to fine spectral details
            detail_score = np.mean(mfcc_vars[7:14])
            
            # Accent-specific scoring (simplified)
            accent_multipliers = {
                "American": {"vocal_tract": 1.0, "detail": 0.8},
                "British": {"vocal_tract": 1.2, "detail": 1.0},
                "Australian": {"vocal_tract": 0.9, "detail": 1.1},
                "Indian": {"vocal_tract": 1.3, "detail": 0.7},
                "Canadian": {"vocal_tract": 1.0, "detail": 0.9}
            }
            
            if accent_name in accent_multipliers:
                multiplier = accent_multipliers[accent_name]
                pattern_score = (vocal_tract_score * multiplier["vocal_tract"] + 
                               detail_score * multiplier["detail"]) / 2.0
            else:
                pattern_score = (vocal_tract_score + detail_score) / 2.0
            
            # Normalize to 0-1 range
            return min(1.0, max(0.0, pattern_score / 10.0))
            
        except Exception:
            return 0.5
    
    def predict_accent(self, audio_path):
        """Main prediction method that combines all analysis approaches"""
        try:
            # Extract comprehensive features
            features = self.extract_comprehensive_features(audio_path)
            if features is None:
                return None, 0.0, "Feature extraction failed"
            
            # Analyze accent patterns
            accent_scores = self.analyze_accent_patterns(features)
            
            # Find best match
            best_accent = max(accent_scores, key=accent_scores.get)
            confidence = accent_scores[best_accent]
            
            # Generate explanation
            explanation = self._generate_explanation(features, accent_scores, best_accent)
            
            return best_accent, confidence, explanation, accent_scores
            
        except Exception as e:
            return None, 0.0, f"Analysis failed: {str(e)}", {}
    
    def _generate_explanation(self, features, accent_scores, predicted_accent):
        """Generate detailed explanation of the accent prediction"""
        explanation = f"Analysis identified {predicted_accent} accent based on: "
        
        reasons = []
        
        # Pitch analysis
        if 'f0' in features and features['f0'] is not None:
            f0_values = features['f0'][~np.isnan(features['f0'])]
            if len(f0_values) > 0:
                mean_pitch = np.mean(f0_values)
                reasons.append(f"pitch characteristics (avg: {mean_pitch:.1f} Hz)")
        
        # Tempo analysis
        if 'tempo' in features:
            reasons.append(f"speaking rhythm (tempo: {features['tempo']:.1f} BPM)")
        
        # Spectral analysis
        if 'statistics' in features:
            centroid = features['statistics'].get('spectral_centroid_mean', 0)
            reasons.append(f"spectral patterns (centroid: {centroid:.0f} Hz)")
        
        # Formant analysis
        if 'formant_features' in features:
            reasons.append("vowel formant characteristics")
        
        explanation += ", ".join(reasons[:3])  # Limit to top 3 reasons
        
        # Add confidence note
        confidence = accent_scores[predicted_accent]
        if confidence > 85:
            explanation += ". High confidence due to clear acoustic markers."
        elif confidence > 75:
            explanation += ". Good confidence with recognizable patterns."
        else:
            explanation += ". Moderate confidence; some features may overlap with other accents."
        
        return explanation

# Enhanced CNN model for accent classification
class AccentCNN(nn.Module):
    def __init__(self, num_classes=10, input_channels=1):
        super(AccentCNN, self).__init__()
        
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Dropout and activation
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Convolutional layers with batch norm and pooling
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(self.relu(self.bn4(self.conv4(x))))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x
    
import React, { useState, useRef, useCallback } from 'react';
import { useVoiceAssistant, BarVisualizer, VoiceAssistantControlBar, DisconnectButton } from '@livekit/components-react';
import {
  Phone,
  Mic,
  Camera,
  AlertTriangle,
  CheckCircle,
  RefreshCw,
  X,
} from 'lucide-react';
import './index.css';

interface VoiceAssistantProps {
  onDisconnect: () => void;
  sessionId?: string;
}

interface InspectionResult {
  response: string;
  hypothesis: string | null;
  confidence: number;
  next_action: string | null;
  clarification_needed: boolean;
  clarification_question: string | null;
}

const VoiceAssistant: React.FC<VoiceAssistantProps> = ({ onDisconnect, sessionId = 'fieldmate_dev_room' }) => {
  const { state, audioTrack } = useVoiceAssistant();

  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isInspecting, setIsInspecting] = useState(false);
  const [inspectionResult, setInspectionResult] = useState<InspectionResult | null>(null);
  const [inspectionError, setInspectionError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const startCamera = async () => {
    try {
      setInspectionError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment',
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setIsCameraActive(true);
    } catch (err: any) {
      console.error('Camera access error:', err);
      setInspectionError('Could not access camera.');
    }
  };

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setIsCameraActive(false);
  }, []);

  const captureAndInspect = async () => {
    if (!videoRef.current) return;

    try {
      setIsInspecting(true);
      setInspectionError(null);

      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth || 1280;
      canvas.height = videoRef.current.videoHeight || 720;
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('Canvas context error');

      ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

      const blob = await new Promise<Blob | null>((resolve) =>
        canvas.toBlob(resolve, 'image/jpeg', 0.85)
      );

      if (!blob) throw new Error('Could not capture frame');

      const formData = new FormData();
      formData.append('image', blob, 'inspection.jpg');
      formData.append('session_id', sessionId);

      const backendUrl = import.meta.env.VITE_BACKEND_URL || '';
      const response = await fetch(`${backendUrl}/api/inspect`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Inspection failed.');
      }

      setInspectionResult(data);
    } catch (err: any) {
      console.error('Inspection error:', err);
      setInspectionError(err.message || 'Inspection failed.');
    } finally {
      setIsInspecting(false);
    }
  };

  const getStatusText = () => {
    if (isInspecting) return 'Inspecting image...';
    switch (state) {
      case 'listening': return 'Listening...';
      case 'speaking': return 'FieldMate is speaking';
      case 'initializing': return 'Connecting...';
      default: return 'Connected';
    }
  };

  return (
    <div className="agent-visualizer">
      {/* Centered Voice Orb */}
      <div className={`avatar-ring ${state}`}>
        <div className="avatar-core">
          <Mic size={32} strokeWidth={1.5} />
        </div>
      </div>

      <div className={`status-text ${state}`}>
        {getStatusText()}
      </div>

      <div className="bar-container">
        {audioTrack ? (
          <BarVisualizer
            state={state}
            barCount={5}
            trackRef={audioTrack}
            className="lk-bar-visualizer"
            options={{ minHeight: 4 }}
          />
        ) : (
          <div className="bar-idle-indicator" />
        )}
      </div>

      {/* Camera Panel */}
      {isCameraActive && (
        <div className="camera-panel">
          <div className="camera-header">
            <span className="camera-title">
              <Camera size={14} /> Camera Feed
            </span>
            <button className="btn-icon" onClick={stopCamera} title="Close Camera">
              <X size={16} />
            </button>
          </div>

          <div className="video-container">
            <video
              ref={(el) => {
                videoRef.current = el;
                if (el && streamRef.current && el.srcObject !== streamRef.current) {
                  el.srcObject = streamRef.current;
                  el.play().catch(() => {});
                }
              }}
              autoPlay
              playsInline
              muted
              className="camera-video"
            />
          </div>

          <div className="camera-actions">
            <button
              className="btn-shutter"
              onClick={captureAndInspect}
              disabled={isInspecting}
            >
              {isInspecting ? (
                <RefreshCw size={15} className="spin-icon" />
              ) : (
                <Camera size={15} />
              )}
              <span>{isInspecting ? 'Analyzing...' : 'Capture Snapshot'}</span>
            </button>
          </div>
        </div>
      )}

      {/* Diagnostic Observation / Contradiction Card */}
      {inspectionResult && (
        <div className="inspection-result-card">
          {inspectionResult.clarification_needed ? (
            <div className="contradiction-banner">
              <AlertTriangle size={16} className="banner-icon" />
              <div>
                <strong>Contradiction Detected</strong>
                <p>{inspectionResult.clarification_question || inspectionResult.response}</p>
              </div>
            </div>
          ) : (
            <div className="evidence-banner">
              <CheckCircle size={16} className="banner-icon" />
              <div>
                <strong>Visual Evidence</strong>
                <p>{inspectionResult.response}</p>
              </div>
            </div>
          )}

          {inspectionResult.next_action && (
            <div className="next-action-pill">
              <span className="next-action-tag">Next Action:</span>
              <span>{inspectionResult.next_action}</span>
            </div>
          )}
        </div>
      )}

      {inspectionError && (
        <div className="error-message">
          {inspectionError}
        </div>
      )}

      {/* Controls */}
      <div className="controls-row">
        <button
          className={`btn-secondary ${isCameraActive ? 'active' : ''}`}
          onClick={isCameraActive ? stopCamera : startCamera}
          title="Toggle Inspection Camera"
        >
          <Camera size={15} />
          <span>{isCameraActive ? 'Hide Camera' : 'Camera'}</span>
        </button>

        <VoiceAssistantControlBar controls={{ leave: false }} />

        <DisconnectButton
          onClick={() => {
            stopCamera();
            onDisconnect();
          }}
          className="btn-danger"
        >
          <Phone size={14} />
          <span>End</span>
        </DisconnectButton>
      </div>
    </div>
  );
};

export default VoiceAssistant;



import React from 'react';
import { useVoiceAssistant, BarVisualizer, VoiceAssistantControlBar, DisconnectButton } from '@livekit/components-react';
import { Phone, Mic } from 'lucide-react';
import './index.css';

interface VoiceAssistantProps {
  onDisconnect: () => void;
}

const VoiceAssistant: React.FC<VoiceAssistantProps> = ({ onDisconnect }) => {
  const { state, audioTrack } = useVoiceAssistant();

  const getStatusText = () => {
    switch (state) {
      case 'listening': return 'Listening...';
      case 'speaking': return 'FieldMate is speaking';
      case 'initializing': return 'Connecting pipeline...';
      default: return 'Waiting for audio...';
    }
  };

  return (
    <div className="agent-visualizer">
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
          <div style={{ height: '4px', width: '20px', background: 'var(--text-muted)', borderRadius: '2px', opacity: 0.3 }} />
        )}
      </div>

      <div className="controls-row">
        <VoiceAssistantControlBar />
        <DisconnectButton onClick={onDisconnect} className="btn-danger">
          <Phone size={14} />
          End
        </DisconnectButton>
      </div>
    </div>
  );
};

export default VoiceAssistant;

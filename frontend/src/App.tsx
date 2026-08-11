import { useState, useCallback } from 'react';
import { LiveKitRoom, RoomAudioRenderer } from '@livekit/components-react';
import { Mic, Activity } from 'lucide-react';
import VoiceAssistant from './VoiceAssistant';
import '@livekit/components-styles';
import './index.css';

function App() {
  const [token, setToken] = useState<string | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const connectToAgent = useCallback(async () => {
    try {
      setIsConnecting(true);
      const backendUrl = import.meta.env.VITE_BACKEND_URL || '';
      const response = await fetch(`${backendUrl}/api/token`);
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Failed to initialize session');
      }
      setToken(data.token);
      setUrl(data.url);
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Could not connect to FieldMate server.');
    } finally {
      setIsConnecting(false);
    }
  }, []);

  const handleDisconnect = useCallback(() => {
    setToken(null);
    setUrl(null);
  }, []);

  return (
    <div className="app-layout">
      <div className="header">
        <div className="badge">
          <Activity size={12} />
          Diagnostic Engine
        </div>
        <h1 className="title">FieldMate</h1>
        <p className="subtitle">Voice diagnostic partner</p>
      </div>

      <div className="main-card">
        {!token ? (
          <div style={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            {error && (
              <div className="error-message">
                {error}
              </div>
            )}
            
            <button 
              className="btn-primary" 
              onClick={connectToAgent}
              disabled={isConnecting}
            >
              <Mic size={18} />
              {isConnecting ? 'Connecting...' : 'Start Session'}
            </button>
          </div>
        ) : (
          <LiveKitRoom
            token={token}
            serverUrl={url || undefined}
            connect={true}
            audio={true}
            video={false}
            onDisconnected={handleDisconnect}
            style={{ width: '100%' }}
          >
            <RoomAudioRenderer />
            <VoiceAssistant onDisconnect={handleDisconnect} />
          </LiveKitRoom>
        )}
      </div>
    </div>
  );
}

export default App;

import { useState, useCallback } from 'react';
import { LiveKitRoom, RoomAudioRenderer } from '@livekit/components-react';
import { Mic, Wrench } from 'lucide-react';
import VoiceAssistant from './VoiceAssistant';
import '@livekit/components-styles';
import './index.css';

function App() {
  const [token, setToken] = useState<string | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [room, setRoom] = useState<string>('fieldmate_dev_room');
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const connectToAgent = useCallback(async () => {
    try {
      setIsConnecting(true);
      setError(null);
      const backendUrl = import.meta.env.VITE_BACKEND_URL || '';
      const response = await fetch(`${backendUrl}/api/token`);
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Failed to initialize session');
      }
      setToken(data.token);
      setUrl(data.url);
      if (data.room) {
        setRoom(data.room);
      }
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
      <header className="header">
        <div className="brand-badge">
          <Wrench size={13} />
          <span>Diagnostic System</span>
        </div>
        <h1 className="title">FieldMate</h1>
        <p className="subtitle">Voice & Vision PC Troubleshooting</p>
      </header>

      <main className="main-container">
        {!token ? (
          <div className="main-card">
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
              <span>{isConnecting ? 'Connecting...' : 'Start Session'}</span>
            </button>
          </div>
        ) : (
          <div className="main-card">
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
              <VoiceAssistant onDisconnect={handleDisconnect} sessionId={room} />
            </LiveKitRoom>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;



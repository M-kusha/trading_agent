import { useEffect, useRef, useCallback, useState } from 'react';

export const useStableWebSocket = (url, options = {}) => {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState(null);
  const ws = useRef(null);
  const reconnectTimer = useRef(null);
  const messageQueue = useRef([]);
  const lastDataHash = useRef({});
  const isMounted = useRef(false);
  
  const connect = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN) return;
    
    try {
      ws.current = new WebSocket(url);
      
      ws.current.onopen = () => {
        if (!isMounted.current) return;
        setIsConnected(true);
        console.log('WebSocket connected');
        
        // Send queued messages
        while (messageQueue.current.length > 0) {
          const msg = messageQueue.current.shift();
          ws.current.send(JSON.stringify(msg));
        }
        
        // Send initial ping
        ws.current.send(JSON.stringify({ type: 'ping' }));
      };
      
      ws.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          // Simple hash to check if data actually changed
          const dataHash = JSON.stringify({
            type: data.type,
            timestamp: Math.floor(Date.parse(data.timestamp || data.data?.timestamp) / 5000)
          });
          
          // Skip if this is duplicate data
          if (lastDataHash.current[data.type] === dataHash && 
              data.type !== 'training_metrics' && 
              data.type !== 'alerts_update') {
            return;
          }
          
          lastDataHash.current[data.type] = dataHash;
          if (isMounted.current) setLastMessage(data);
          
        } catch (error) {
          console.error('WebSocket message parsing error:', error);
        }
      };
      
      ws.current.onclose = () => {
        if (!isMounted.current) return;
        setIsConnected(false);
        console.log('WebSocket disconnected');
        
        // Reconnect after delay
        if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
        reconnectTimer.current = setTimeout(() => {
          if (!isMounted.current) return;
          console.log('Attempting to reconnect...');
          connect();
        }, 3000);
      };
      
      ws.current.onerror = (error) => {
        // Browsers emit a generic Event for WS errors; keep log concise
        console.error('WebSocket error:', error);
      };
      
    } catch (error) {
      console.error('WebSocket connection error:', error);
      if (isMounted.current) setIsConnected(false);
    }
  }, [url]);
  
  const sendMessage = useCallback((message) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message));
    } else {
      messageQueue.current.push(message);
    }
  }, []);
  
  const disconnect = useCallback(() => {
    if (reconnectTimer.current) {
      clearTimeout(reconnectTimer.current);
      reconnectTimer.current = null;
    }
    if (ws.current) {
      ws.current.close();
      ws.current = null;
    }
  }, []);
  
  useEffect(() => {
    isMounted.current = true;
    connect();
    return () => {
      isMounted.current = false;
      disconnect();
    };
  }, [connect, disconnect]);
  
  return { isConnected, lastMessage, sendMessage, reconnect: connect };
};
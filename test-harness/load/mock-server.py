#!/usr/bin/env python3
"""
Mock HTTP server for k6 load testing validation
Created by: TEAM-107 | 2025-10-18

Provides minimal endpoints to test k6 scripts without full rbee stack.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import time
import random

class MockHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        """Suppress request logging"""
        pass
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'healthy'}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/v2/tasks':
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            
            try:
                request_data = json.loads(body)
                
                # Simulate processing time (10-100ms)
                time.sleep(random.uniform(0.01, 0.1))
                
                # Mock response
                response = {
                    'task_id': f'task-{random.randint(1000, 9999)}',
                    'status': 'completed',
                    'result': {
                        'text': f'Mock response to: {request_data.get("prompt", "unknown")}',
                        'tokens': ['mock', 'token', 'response'],
                        'model': request_data.get('model', 'unknown'),
                    },
                    'latency_ms': random.randint(50, 200),
                }
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port=8080):
    """Run the mock server"""
    server = HTTPServer(('0.0.0.0', port), MockHandler)
    print(f'🚀 Mock server running on http://localhost:{port}')
    print(f'   Health: http://localhost:{port}/health')
    print(f'   Tasks: POST http://localhost:{port}/v2/tasks')
    print(f'   Press Ctrl+C to stop')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n✅ Server stopped')

if __name__ == '__main__':
    run_server()

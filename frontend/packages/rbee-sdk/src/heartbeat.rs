// TEAM-286: Heartbeat monitoring via SSE
// Port from: bin/10_queen_rbee/examples/heartbeat_monitor.html

use wasm_bindgen::prelude::*;
use web_sys::{EventSource, MessageEvent};
use js_sys::Function;

/// Heartbeat monitor for real-time system status
///
/// TEAM-286: This connects to GET /v1/heartbeats/stream and receives
/// HeartbeatSnapshot events every 5 seconds.
///
/// Port from: bin/10_queen_rbee/examples/heartbeat_monitor.html
#[wasm_bindgen]
pub struct HeartbeatMonitor {
    event_source: Option<EventSource>,
    base_url: String,
}

#[wasm_bindgen]
impl HeartbeatMonitor {
    /// Create a new HeartbeatMonitor
    ///
    /// # Arguments
    /// * `base_url` - Base URL of queen-rbee (e.g., "http://localhost:8500")
    #[wasm_bindgen(constructor)]
    pub fn new(base_url: String) -> Self {
        Self {
            event_source: None,
            base_url,
        }
    }

    /// Start monitoring heartbeats
    ///
    /// TEAM-286: Connects to /v1/heartbeats/stream and calls the callback
    /// for each heartbeat event.
    ///
    /// Port from: heartbeat_monitor.html lines 290-321
    ///
    /// # Arguments
    /// * `on_update` - Callback function called with each HeartbeatSnapshot
    ///
    /// # JavaScript Example
    /// ```javascript
    /// const monitor = new HeartbeatMonitor('http://localhost:8500');
    /// monitor.start((snapshot) => {
    ///   console.log('Workers online:', snapshot.workers_online);
    ///   console.log('Hives online:', snapshot.hives_online);
    /// });
    /// ```
    #[wasm_bindgen]
    pub fn start(&mut self, on_update: Function) -> Result<(), JsValue> {
        // TEAM-286: Close existing connection if any
        if let Some(es) = &self.event_source {
            es.close();
        }

        // TEAM-286: Connect to heartbeat stream
        let url = format!("{}/v1/heartbeats/stream", self.base_url);
        web_sys::console::log_1(&JsValue::from_str(&format!("🐝 [SDK] Connecting to SSE: {}", url)));
        
        let event_source = EventSource::new(&url)
            .map_err(|e| JsValue::from_str(&format!("Failed to create EventSource: {:?}", e)))?;

        web_sys::console::log_1(&JsValue::from_str(&format!("🐝 [SDK] EventSource created, ready_state: {}", event_source.ready_state())));

        // TEAM-288: Add open event listener
        let open_closure = Closure::wrap(Box::new(move |_event: web_sys::Event| {
            web_sys::console::log_1(&JsValue::from_str("🐝 [SDK] SSE connection OPENED"));
        }) as Box<dyn FnMut(web_sys::Event)>);
        event_source.add_event_listener_with_callback("open", open_closure.as_ref().unchecked_ref())?;
        open_closure.forget();

        // TEAM-288: Add error event listener
        // TEAM-XXX: Improved error handling - Event object doesn't contain useful error details
        // We need to check EventSource readyState and provide context
        let es_for_error = event_source.clone();
        let error_closure = Closure::wrap(Box::new(move |_event: web_sys::Event| {
            let ready_state = es_for_error.ready_state();
            let state_str = match ready_state {
                0 => "CONNECTING",
                1 => "OPEN",
                2 => "CLOSED",
                _ => "UNKNOWN",
            };
            
            // SSE errors are typically connection failures
            // The browser will automatically retry, so this is informational
            web_sys::console::warn_1(&JsValue::from_str(&format!(
                "🐝 [SDK] SSE connection error (state: {} [{}]). Browser will retry automatically.",
                state_str, ready_state
            )));
        }) as Box<dyn FnMut(web_sys::Event)>);
        event_source.add_event_listener_with_callback("error", error_closure.as_ref().unchecked_ref())?;
        error_closure.forget();

        // TEAM-286: Set up event listener for 'heartbeat' events
        // Port from: heartbeat_monitor.html lines 290-302
        let callback = on_update.clone();
        let closure = Closure::wrap(Box::new(move |event: MessageEvent| {
            web_sys::console::log_1(&JsValue::from_str("🐝 [SDK] Received 'heartbeat' event"));
            
            // Parse the event data
            if let Some(data) = event.data().as_string() {
                web_sys::console::log_1(&JsValue::from_str(&format!("🐝 [SDK] Event data: {}", data)));
                
                // Try to parse as JSON
                match js_sys::JSON::parse(&data) {
                    Ok(json_value) => {
                        web_sys::console::log_1(&JsValue::from_str("🐝 [SDK] JSON parsed successfully, calling callback"));
                        // Call the JavaScript callback with the parsed data
                        let _ = callback.call1(&JsValue::null(), &json_value);
                    }
                    Err(e) => {
                        web_sys::console::error_1(&JsValue::from_str(&format!("🐝 [SDK] JSON parse error: {:?}", e)));
                        // If parsing fails, just pass the string
                        let _ = callback.call1(&JsValue::null(), &JsValue::from_str(&data));
                    }
                }
            } else {
                web_sys::console::warn_1(&JsValue::from_str("🐝 [SDK] Event has no data"));
            }
        }) as Box<dyn FnMut(MessageEvent)>);

        event_source.add_event_listener_with_callback(
            "heartbeat",
            closure.as_ref().unchecked_ref()
        )?;

        web_sys::console::log_1(&JsValue::from_str("🐝 [SDK] 'heartbeat' event listener registered"));

        // TEAM-286: Keep the closure alive
        closure.forget();

        // TEAM-286: Store the event source
        self.event_source = Some(event_source);

        web_sys::console::log_1(&JsValue::from_str("🐝 [SDK] HeartbeatMonitor.start() complete"));
        Ok(())
    }

    /// Stop monitoring heartbeats
    ///
    /// TEAM-286: Closes the SSE connection
    ///
    /// Port from: heartbeat_monitor.html lines 330-334
    #[wasm_bindgen]
    pub fn stop(&mut self) {
        if let Some(es) = &self.event_source {
            es.close();
        }
        self.event_source = None;
    }

    /// Check if currently connected
    #[wasm_bindgen(js_name = isConnected)]
    pub fn is_connected(&self) -> bool {
        if let Some(es) = &self.event_source {
            es.ready_state() == EventSource::OPEN
        } else {
            false
        }
    }

    /// Get the connection state
    ///
    /// Returns: 0 = CONNECTING, 1 = OPEN, 2 = CLOSED
    #[wasm_bindgen(js_name = readyState)]
    pub fn ready_state(&self) -> u16 {
        if let Some(es) = &self.event_source {
            es.ready_state()
        } else {
            2 // CLOSED
        }
    }
}

// TEAM-286: Implement Drop to ensure cleanup
impl Drop for HeartbeatMonitor {
    fn drop(&mut self) {
        self.stop();
    }
}

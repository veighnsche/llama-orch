// TEAM-374: Heartbeat monitoring via SSE for Hive
// Copied from: bin/10_queen_rbee/ui/packages/queen-rbee-sdk/src/heartbeat.rs

use wasm_bindgen::prelude::*;
use web_sys::{EventSource, MessageEvent};
use js_sys::Function;

/// Heartbeat monitor for real-time hive status
///
/// TEAM-374: This connects to GET /v1/heartbeats/stream and receives
/// HiveHeartbeatEvent events every 1 second.
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
    /// * `base_url` - Base URL of rbee-hive (e.g., "http://localhost:7835")
    #[wasm_bindgen(constructor)]
    pub fn new(base_url: String) -> Self {
        Self {
            event_source: None,
            base_url,
        }
    }

    /// Start monitoring heartbeats
    ///
    /// TEAM-374: Connects to /v1/heartbeats/stream and calls the callback
    /// for each heartbeat event.
    ///
    /// # Arguments
    /// * `on_update` - Callback function called with each HiveHeartbeatEvent
    ///
    /// # JavaScript Example
    /// ```javascript
    /// const monitor = new HeartbeatMonitor('http://localhost:7835');
    /// monitor.start((event) => {
    ///   console.log('Workers:', event.workers);
    ///   console.log('Hive ID:', event.hive_id);
    /// });
    /// ```
    #[wasm_bindgen]
    pub fn start(&mut self, on_update: Function) -> Result<(), JsValue> {
        // Close existing connection if any
        if let Some(es) = &self.event_source {
            es.close();
        }

        // Connect to heartbeat stream
        let url = format!("{}/v1/heartbeats/stream", self.base_url);
        
        let event_source = EventSource::new(&url)
            .map_err(|e| JsValue::from_str(&format!("Failed to create EventSource: {:?}", e)))?;

        // Add open event listener (silent)
        let open_closure = Closure::wrap(Box::new(move |_event: web_sys::Event| {
            // Connection opened - no log needed
        }) as Box<dyn FnMut(web_sys::Event)>);
        event_source.add_event_listener_with_callback("open", open_closure.as_ref().unchecked_ref())?;
        open_closure.forget();

        // Add error event listener
        let es_for_error = event_source.clone();
        let error_closure = Closure::wrap(Box::new(move |_event: web_sys::Event| {
            let ready_state = es_for_error.ready_state();
            let state_str = match ready_state {
                0 => "CONNECTING",
                1 => "OPEN",
                2 => "CLOSED",
                _ => "UNKNOWN",
            };
            
            web_sys::console::warn_1(&JsValue::from_str(&format!(
                "🐝 [Hive SDK] SSE connection error (state: {} [{}]). Browser will retry automatically.",
                state_str, ready_state
            )));
        }) as Box<dyn FnMut(web_sys::Event)>);
        event_source.add_event_listener_with_callback("error", error_closure.as_ref().unchecked_ref())?;
        error_closure.forget();

        // Set up event listener for 'heartbeat' events
        let callback = on_update.clone();
        let closure = Closure::wrap(Box::new(move |event: MessageEvent| {
            // Parse the event data
            if let Some(data) = event.data().as_string() {
                // Try to parse as JSON
                match js_sys::JSON::parse(&data) {
                    Ok(json_value) => {
                        // Call callback without logging
                        let _ = callback.call1(&JsValue::null(), &json_value);
                    }
                    Err(e) => {
                        web_sys::console::error_1(&JsValue::from_str(&format!("🐝 [Hive SDK] JSON parse error: {:?}", e)));
                        // If parsing fails, just pass the string
                        let _ = callback.call1(&JsValue::null(), &JsValue::from_str(&data));
                    }
                }
            } else {
                web_sys::console::warn_1(&JsValue::from_str("🐝 [Hive SDK] Event has no data"));
            }
        }) as Box<dyn FnMut(MessageEvent)>);

        event_source.add_event_listener_with_callback(
            "heartbeat",
            closure.as_ref().unchecked_ref()
        )?;

        web_sys::console::log_1(&JsValue::from_str("🐝 [Hive SDK] 'heartbeat' event listener registered"));

        // Keep the closure alive
        closure.forget();

        // Store the event source
        self.event_source = Some(event_source);

        web_sys::console::log_1(&JsValue::from_str("🐝 [Hive SDK] HeartbeatMonitor.start() complete"));
        Ok(())
    }

    /// Stop monitoring heartbeats
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

    /// Check if hive is reachable before starting SSE
    ///
    /// TEAM-374: Prevents noisy CORS errors when hive is offline.
    /// Returns a Promise that resolves to true if /health returns 200.
    #[wasm_bindgen(js_name = checkHealth)]
    pub async fn check_health(&self) -> Result<bool, JsValue> {
        use wasm_bindgen_futures::JsFuture;
        use web_sys::{Request, RequestInit, Response};

        let health_url = format!("{}/health", self.base_url);
        
        let mut opts = RequestInit::new();
        opts.method("GET");
        
        let request = Request::new_with_str_and_init(&health_url, &opts)
            .map_err(|e| JsValue::from_str(&format!("Failed to create request: {:?}", e)))?;

        let window = web_sys::window().ok_or_else(|| JsValue::from_str("No window object"))?;
        let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;
        let resp: Response = resp_value.dyn_into()?;

        Ok(resp.ok())
    }
}

// Implement Drop to ensure cleanup
impl Drop for HeartbeatMonitor {
    fn drop(&mut self) {
        self.stop();
    }
}

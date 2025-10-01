# error-handler

**Worker-level error handling and recovery**

Handles GPU-specific errors (CUDA OOM, kernel failures), inference errors (invalid input), and recovery without crashing the worker.

**Key responsibilities:**
- Catch CUDA errors (OOM, invalid memory access)
- Handle inference failures (malformed prompt, context overflow)
- Clear GPU state after error
- Report failures to pool-managerd
- Graceful degradation (reduce batch size on OOM)
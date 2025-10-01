# job-timeout

**Job timeout enforcement and hung job cleanup**

Monitors running jobs, aborts jobs that exceed max execution time, and cleans up resources from hung jobs.

**Key responsibilities:**
- Track job start time
- Enforce max execution time (e.g., 5 minutes)
- Abort jobs that exceed timeout
- Clean up zombie jobs (worker died but job still "running")
- Return timeout error to client
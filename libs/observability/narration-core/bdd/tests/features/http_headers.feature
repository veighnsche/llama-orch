Feature: HTTP Header Propagation Behaviors
  As a Cloud Profile developer
  I want to propagate correlation and trace context via HTTP headers
  So that I can trace requests across services

  Scenario: B-HTTP-010 - Extract correlation ID from header
    When I set header "X-Correlation-Id" to "req-xyz"
    And I extract context from headers
    Then extracted correlation_id is "req-xyz"

  Scenario: B-HTTP-011 - Extract trace ID from header
    When I set header "X-Trace-Id" to "trace-123"
    And I extract context from headers
    Then extracted trace_id is "trace-123"

  Scenario: B-HTTP-012 - Extract span ID from header
    When I set header "X-Span-Id" to "span-456"
    And I extract context from headers
    Then extracted span_id is "span-456"

  Scenario: B-HTTP-013 - Extract parent span ID from header
    When I set header "X-Parent-Span-Id" to "parent-789"
    And I extract context from headers
    Then extracted parent_span_id is "parent-789"

  Scenario: B-HTTP-014 - Missing header returns None
    When I extract context from headers
    Then extracted correlation_id is None

  Scenario: B-HTTP-016 - All headers missing returns all None
    When I extract context from headers
    Then all extracted fields are None

  Scenario: B-HTTP-020 - Inject correlation ID into headers
    When I inject correlation_id "req-xyz" into headers
    Then header "X-Correlation-Id" is "req-xyz"

  Scenario: B-HTTP-021 - Inject trace ID into headers
    When I inject trace_id "trace-123" into headers
    Then header "X-Trace-Id" is "trace-123"

  Scenario: B-HTTP-022 - Inject span ID into headers
    When I inject span_id "span-456" into headers
    Then header "X-Span-Id" is "span-456"

  Scenario: B-HTTP-023 - Inject parent span ID into headers
    When I inject parent_span_id "parent-789" into headers
    Then header "X-Parent-Span-Id" is "parent-789"

  Scenario: B-HTTP-024 - None field skips header insertion
    When I inject correlation_id None into headers
    Then header "X-Correlation-Id" is not present

  Scenario: B-HTTP-025 - All None fields insert no headers
    When I inject all None fields into headers
    Then no headers are present

#!/usr/bin/env python3
"""
TEAM-077: Feature file extraction script
Extracts scenarios from test-001.feature into focused feature files
"""

# Scenario line numbers from test-001.feature
scenarios = {
    # File 01: SSH Registry (already created)
    "01-ssh-registry-management.feature": [37, 77, 103, 129, 144, 170, 190, 201, 224, 250],
    
    # File 02: Model Provisioning (12 scenarios)
    "02-model-provisioning.feature": [459, 468, 485, 500, 515, 530, 548, 564, 581, 590, 602, 614],
    
    # File 03: Worker Preflight (9 scenarios)
    "03-worker-preflight-checks.feature": [625, 634, 655, 671, 679, 698, 716, 734, 752],
    
    # File 04: Worker Lifecycle (10 scenarios)
    "04-worker-lifecycle.feature": [771, 789, 807, 823, 837, 855, 874, 888, 900, 914],
    
    # File 05: Inference Execution (6 scenarios)
    "05-inference-execution.feature": [934, 966, 1003, 1022, 1038, 1058],
    
    # File 06: Error Handling Network (10 scenarios)
    "06-error-handling-network.feature": [402, 420, 444, 530, 1003, 1071, 1084, 1098, 1115, 1136],
    
    # File 07: Error Handling Resources (8 scenarios)
    "07-error-handling-resources.feature": [1058, 1188, 1204, 1228, 1251, 1275, 1301, 1326],
    
    # File 08: Daemon Lifecycle (9 scenarios)
    "08-daemon-lifecycle.feature": [1342, 1353, 1364, 1374, 1386, 1401, 1411, 1421, 1435],
    
    # File 09: Happy Path Flows (17 scenarios)
    "09-happy-path-flows.feature": [274, 328, 362, 373, 378, 387, 435, 1450, 1475, 1490, 1500, 1515, 1530, 1546, 1552, 1557, 1564],
}

# Print summary
total = sum(len(v) for v in scenarios.values())
print(f"Total scenarios mapped: {total}")
print(f"Expected: 91")
print(f"Match: {total == 91}")

for filename, lines in scenarios.items():
    print(f"{filename}: {len(lines)} scenarios")

#!/usr/bin/env python3
"""
Chaos Testing Controller
Created by: TEAM-107 | 2025-10-18

Orchestrates chaos scenarios using toxiproxy and Docker.
"""

import json
import time
import requests
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class ToxiproxyClient:
    """Client for toxiproxy API"""
    
    def __init__(self, base_url: str = "http://toxiproxy:8474"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def create_proxy(self, name: str, listen: str, upstream: str) -> Dict:
        """Create a new proxy"""
        payload = {
            "name": name,
            "listen": listen,
            "upstream": upstream,
            "enabled": True
        }
        resp = self.session.post(f"{self.base_url}/proxies", json=payload)
        resp.raise_for_status()
        return resp.json()
    
    def add_toxic(self, proxy: str, toxic_type: str, attributes: Dict, 
                  stream: str = "downstream") -> Dict:
        """Add a toxic to a proxy"""
        payload = {
            "type": toxic_type,
            "stream": stream,
            "toxicity": 1.0,
            "attributes": attributes
        }
        resp = self.session.post(
            f"{self.base_url}/proxies/{proxy}/toxics",
            json=payload
        )
        resp.raise_for_status()
        return resp.json()
    
    def remove_toxic(self, proxy: str, toxic_name: str):
        """Remove a toxic from a proxy"""
        resp = self.session.delete(
            f"{self.base_url}/proxies/{proxy}/toxics/{toxic_name}"
        )
        resp.raise_for_status()
    
    def list_proxies(self) -> List[Dict]:
        """List all proxies"""
        resp = self.session.get(f"{self.base_url}/proxies")
        resp.raise_for_status()
        return resp.json()
    
    def reset(self):
        """Reset all proxies"""
        resp = self.session.post(f"{self.base_url}/reset")
        resp.raise_for_status()


class ChaosController:
    """Main chaos testing controller"""
    
    def __init__(self, toxiproxy_url: str, results_dir: Path):
        self.toxiproxy = ToxiproxyClient(toxiproxy_url)
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def setup_proxies(self):
        """Set up toxiproxy proxies for services"""
        print("🔧 Setting up toxiproxy proxies...")
        
        proxies = [
            ("queen-proxy", "0.0.0.0:8080", "queen-rbee:8081"),
            ("hive-proxy", "0.0.0.0:9200", "rbee-hive:9201"),
        ]
        
        for name, listen, upstream in proxies:
            try:
                self.toxiproxy.create_proxy(name, listen, upstream)
                print(f"  ✅ Created proxy: {name} ({listen} -> {upstream})")
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 409:
                    print(f"  ⚠️  Proxy {name} already exists")
                else:
                    raise
    
    def run_network_failure_scenario(self, scenario: Dict) -> Dict:
        """Run a network failure scenario"""
        scenario_id = scenario["id"]
        print(f"\n🌪️  Running scenario: {scenario_id} - {scenario['name']}")
        
        result = {
            "scenario_id": scenario_id,
            "name": scenario["name"],
            "type": "network_failure",
            "start_time": datetime.utcnow().isoformat(),
            "status": "running"
        }
        
        try:
            # Add toxic to queen-proxy
            toxic = self.toxiproxy.add_toxic(
                "queen-proxy",
                scenario["toxic"],
                scenario["attributes"]
            )
            toxic_name = toxic["name"]
            print(f"  ✅ Added toxic: {scenario['toxic']}")
            
            # Wait for duration
            duration = scenario["duration_seconds"]
            print(f"  ⏳ Running for {duration} seconds...")
            time.sleep(duration)
            
            # Remove toxic
            self.toxiproxy.remove_toxic("queen-proxy", toxic_name)
            print(f"  ✅ Removed toxic")
            
            result["status"] = "completed"
            result["error"] = None
            
        except Exception as e:
            print(f"  ❌ Scenario failed: {e}")
            result["status"] = "failed"
            result["error"] = str(e)
        
        result["end_time"] = datetime.utcnow().isoformat()
        self.results.append(result)
        return result
    
    def run_worker_crash_scenario(self, scenario: Dict) -> Dict:
        """Run a worker crash scenario"""
        scenario_id = scenario["id"]
        print(f"\n💥 Running scenario: {scenario_id} - {scenario['name']}")
        
        result = {
            "scenario_id": scenario_id,
            "name": scenario["name"],
            "type": "worker_crash",
            "start_time": datetime.utcnow().isoformat(),
            "status": "running"
        }
        
        try:
            # Get container ID
            target = scenario["target"]
            cmd = ["docker", "ps", "-q", "-f", f"name={target}"]
            container_id = subprocess.check_output(cmd).decode().strip()
            
            if not container_id:
                raise RuntimeError(f"Container {target} not found")
            
            print(f"  🎯 Target container: {container_id[:12]}")
            
            # Send signal
            signal = scenario["signal"]
            cmd = ["docker", "kill", "-s", signal, container_id]
            subprocess.check_call(cmd)
            print(f"  ✅ Sent {signal} to {target}")
            
            # Wait for recovery
            time.sleep(10)
            
            result["status"] = "completed"
            result["error"] = None
            
        except Exception as e:
            print(f"  ❌ Scenario failed: {e}")
            result["status"] = "failed"
            result["error"] = str(e)
        
        result["end_time"] = datetime.utcnow().isoformat()
        self.results.append(result)
        return result
    
    def run_resource_exhaustion_scenario(self, scenario: Dict) -> Dict:
        """Run a resource exhaustion scenario"""
        scenario_id = scenario["id"]
        print(f"\n🔥 Running scenario: {scenario_id} - {scenario['name']}")
        
        result = {
            "scenario_id": scenario_id,
            "name": scenario["name"],
            "type": "resource_exhaustion",
            "start_time": datetime.utcnow().isoformat(),
            "status": "running"
        }
        
        try:
            resource = scenario["resource"]
            target = scenario["target"]
            
            # Get container ID
            cmd = ["docker", "ps", "-q", "-f", f"name={target}"]
            container_id = subprocess.check_output(cmd).decode().strip()
            
            if not container_id:
                raise RuntimeError(f"Container {target} not found")
            
            print(f"  🎯 Target: {target} ({container_id[:12]})")
            
            # Execute resource exhaustion based on type
            if resource == "cpu":
                # Use stress-ng to saturate CPU
                cmd = [
                    "docker", "exec", container_id,
                    "stress-ng", "--cpu", "0", "--timeout", 
                    f"{scenario.get('duration_seconds', 60)}s"
                ]
                subprocess.run(cmd, check=False)
                
            elif resource == "memory":
                # Allocate memory until threshold
                cmd = [
                    "docker", "exec", container_id,
                    "stress-ng", "--vm", "1", "--vm-bytes", "90%",
                    "--timeout", "30s"
                ]
                subprocess.run(cmd, check=False)
                
            elif resource == "disk":
                # Fill disk with large file
                cmd = [
                    "docker", "exec", container_id,
                    "dd", "if=/dev/zero", "of=/tmp/fill", "bs=1M", "count=1000"
                ]
                subprocess.run(cmd, check=False)
                
            print(f"  ✅ Resource exhaustion completed")
            
            result["status"] = "completed"
            result["error"] = None
            
        except Exception as e:
            print(f"  ❌ Scenario failed: {e}")
            result["status"] = "failed"
            result["error"] = str(e)
        
        result["end_time"] = datetime.utcnow().isoformat()
        self.results.append(result)
        return result
    
    def run_all_scenarios(self):
        """Run all chaos scenarios"""
        scenarios_dir = Path("/scenarios")
        
        # Network failures
        with open(scenarios_dir / "network-failures.json") as f:
            network_scenarios = json.load(f)
        
        for scenario in network_scenarios["scenarios"]:
            self.run_network_failure_scenario(scenario)
            time.sleep(5)  # Cool-down between scenarios
        
        # Worker crashes
        with open(scenarios_dir / "worker-crashes.json") as f:
            crash_scenarios = json.load(f)
        
        for scenario in crash_scenarios["scenarios"]:
            self.run_worker_crash_scenario(scenario)
            time.sleep(5)
        
        # Resource exhaustion
        with open(scenarios_dir / "resource-exhaustion.json") as f:
            resource_scenarios = json.load(f)
        
        for scenario in resource_scenarios["scenarios"]:
            self.run_resource_exhaustion_scenario(scenario)
            time.sleep(5)
    
    def save_results(self):
        """Save test results to file"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"chaos_results_{timestamp}.json"
        
        summary = {
            "test_run": timestamp,
            "total_scenarios": len(self.results),
            "completed": sum(1 for r in self.results if r["status"] == "completed"),
            "failed": sum(1 for r in self.results if r["status"] == "failed"),
            "scenarios": self.results
        }
        
        with open(results_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 Results saved to: {results_file}")
        return summary


def main():
    """Main entry point"""
    print("🌪️  CHAOS TESTING CONTROLLER")
    print("=" * 60)
    print("Created by: TEAM-107 | 2025-10-18\n")
    
    toxiproxy_url = "http://toxiproxy:8474"
    results_dir = Path("/results")
    
    controller = ChaosController(toxiproxy_url, results_dir)
    
    # Wait for services to be ready
    print("⏳ Waiting for services to be ready...")
    time.sleep(10)
    
    # Setup proxies
    controller.setup_proxies()
    
    # Run all scenarios
    print("\n🚀 Starting chaos scenarios...\n")
    controller.run_all_scenarios()
    
    # Save results
    summary = controller.save_results()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 CHAOS TESTING SUMMARY")
    print("=" * 60)
    print(f"Total scenarios: {summary['total_scenarios']}")
    print(f"Completed: {summary['completed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success rate: {summary['completed'] / summary['total_scenarios'] * 100:.1f}%")
    
    # Exit with error if any scenarios failed
    sys.exit(0 if summary['failed'] == 0 else 1)


if __name__ == "__main__":
    main()

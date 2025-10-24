// TEAM-188: SSH connection test example
// Test SSH connection to a remote host

use queen_rbee_ssh_client::{test_ssh_connection, SshConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Get host from command line args or use default
    let args: Vec<String> = std::env::args().collect();
    let host = args.get(1).map(|s| s.as_str()).unwrap_or("workstation.arpa.home");
    let user = args.get(2).map(|s| s.as_str()).unwrap_or("vince");

    println!("🔐 Testing SSH connection to {}@{}", user, host);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let config =
        SshConfig { host: host.to_string(), port: 22, user: user.to_string(), timeout_secs: 5 };

    let result = test_ssh_connection(config).await?;

    println!("\n📊 Result:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    if result.success {
        println!("✅ SUCCESS");
        if let Some(output) = result.test_output {
            println!("📝 Test command output: {}", output);
        }
    } else {
        println!("❌ FAILED");
        if let Some(error) = result.error {
            println!("🚨 Error: {}", error);
        }
    }

    Ok(())
}

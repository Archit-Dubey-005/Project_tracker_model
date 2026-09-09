// Example Node.js script to test the IPT Model API
// Run with: node test_client.js

const API_URL = process.env.IPT_API_URL || 'http://localhost:8000';
const API_KEY = process.env.IPT_API_KEY || '';

async function testMatch() {
  console.log(`[+] Testing IPT Match API at: ${API_URL}/api/v1/match\n`);

  const payload = {
    query_text: "RCC footing reinforcement completed at Block A",
    project_id: "P00010",
    top_k: 3
  };

  try {
    const response = await fetch(`${API_URL}/api/v1/match`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(API_KEY ? { 'Authorization': `Bearer ${API_KEY}` } : {})
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`HTTP ${response.status}: ${errorText}`);
    }

    const data = await response.json();
    console.log("=== Match Result from Model ===");
    console.log("Top Match Activity ID  :", data.data.top_match.activity_id);
    console.log("Activity Name          :", data.data.top_match.activity_name);
    console.log("AI Confidence Score    :", data.data.top_match["AI Confidence Score"]);
    console.log("Schedule Status        :", data.data.top_match["Schedule Status"]);
    console.log("Delay / Early Duration :", data.data.top_match["Delay/Early Duration"]);
    console.log("Activity % Completed   :", data.data.top_match["Activity Completion Percentage"]);
    console.log("Total Candidates Found :", data.data.candidates.length);
    console.log("\nFull Response:\n", JSON.stringify(data, null, 2));

  } catch (err) {
    console.error("[-] Request failed:", err.message);
  }
}

testMatch();

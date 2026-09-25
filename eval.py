"""
SHL Recommender Evaluation Harness
Tests the /chat endpoint against conversation traces and computes Recall@10.

Usage:
    python eval.py --url http://localhost:8000 --traces traces/

Trace format (JSON):
{
  "persona": "Sarah is a recruiter at a fintech company...",
  "facts": {
    "role": "Java developer",
    "level": "mid-level",
    "years_experience": 4,
    ...
  },
  "expected_assessments": ["Java 8 (New)", "OPQ32r", "Verify Numerical Reasoning"]
}
"""

import argparse
import json
import time
from pathlib import Path
import requests


def recall_at_k(recommended: list[str], relevant: list[str], k: int = 10) -> float:
    """Compute Recall@K."""
    if not relevant:
        return 1.0
    top_k = recommended[:k]
    hits = sum(1 for r in relevant if r in top_k)
    return hits / len(relevant)


def run_conversation(base_url: str, trace: dict, max_turns: int = 8) -> dict:
    """
    Simulate a conversation for a given trace.
    The simulated user starts with the role description and answers follow-up questions.
    """
    facts = trace.get("facts", {})
    
    # Opening message from the simulated user
    opening = f"I'm hiring a {facts.get('role', 'candidate')}."
    if facts.get('description'):
        opening = facts['description']

    messages = [{"role": "user", "content": opening}]
    final_recommendations = []
    turns = 0

    while turns < max_turns:
        turns += 1
        
        try:
            resp = requests.post(
                f"{base_url}/chat",
                json={"messages": messages},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"    ERROR on turn {turns}: {e}")
            break

        reply = data.get("reply", "")
        recs = data.get("recommendations", [])
        eoc = data.get("end_of_conversation", False)

        messages.append({"role": "assistant", "content": reply})

        if recs:
            final_recommendations = [r["name"] for r in recs]

        if eoc:
            break

        if recs:
            # Agent gave a shortlist — simulated user says they're done
            messages.append({"role": "user", "content": "Thank you, that looks good!"})
            # One more turn to let agent say goodbye
            try:
                resp2 = requests.post(
                    f"{base_url}/chat",
                    json={"messages": messages},
                    timeout=30,
                )
                final_data = resp2.json()
                messages.append({"role": "assistant", "content": final_data.get("reply", "")})
            except Exception:
                pass
            break

        # Simulated user responds to clarifying questions
        # Pull relevant fact to answer the question
        user_reply = generate_user_reply(reply, facts)
        messages.append({"role": "user", "content": user_reply})

    return {
        "turns": turns,
        "final_recommendations": final_recommendations,
        "conversation": messages,
    }


def generate_user_reply(agent_reply: str, facts: dict) -> str:
    """
    Heuristic simulated user: answer agent's question from facts.
    In production, replace this with an LLM call given the persona and facts.
    """
    reply_lower = agent_reply.lower()

    if any(w in reply_lower for w in ["seniority", "level", "experience", "junior", "senior"]):
        level = facts.get("level", facts.get("seniority", "mid-level"))
        years = facts.get("years_experience", "")
        return f"{level}" + (f", around {years} years of experience" if years else "")
    
    if any(w in reply_lower for w in ["role", "position", "job title", "what kind"]):
        return facts.get("role", "software developer")
    
    if any(w in reply_lower for w in ["skill", "competenc", "assess", "focus"]):
        competencies = facts.get("competencies", facts.get("skills", []))
        if competencies:
            return f"I want to assess: {', '.join(competencies)}"
        return "I have no specific preference"
    
    if any(w in reply_lower for w in ["remote", "online", "in-person"]):
        remote = facts.get("remote_testing", True)
        return "Yes, remote testing is required" if remote else "In-person is fine"
    
    if any(w in reply_lower for w in ["language", "location", "country"]):
        lang = facts.get("language", "English")
        return f"We need {lang}"
    
    if any(w in reply_lower for w in ["industry", "sector"]):
        return facts.get("industry", "I have no preference on industry")
    
    # Default: no preference
    return "I have no preference on that"


def run_eval(base_url: str, traces_dir: str) -> None:
    """Run evaluation against all traces in the directory."""
    traces_path = Path(traces_dir)
    trace_files = list(traces_path.glob("*.json"))
    
    if not trace_files:
        print(f"No trace files found in {traces_dir}")
        return

    print(f"Running eval against {len(trace_files)} traces at {base_url}\n")
    
    # Health check
    try:
        health = requests.get(f"{base_url}/health", timeout=5)
        print(f"Health: {health.json()}\n")
    except Exception as e:
        print(f"Health check failed: {e}\n")

    results = []

    for trace_file in sorted(trace_files):
        with open(trace_file) as f:
            trace = json.load(f)

        expected = trace.get("expected_assessments", [])
        persona = trace.get("persona", trace_file.stem)
        
        print(f"Trace: {trace_file.name}")
        print(f"  Persona: {persona[:80]}...")
        print(f"  Expected ({len(expected)}): {expected}")

        result = run_conversation(base_url, trace)
        
        r_at_10 = recall_at_k(result["final_recommendations"], expected, k=10)
        
        print(f"  Got ({len(result['final_recommendations'])}): {result['final_recommendations']}")
        print(f"  Recall@10: {r_at_10:.2f} | Turns: {result['turns']}")
        print()

        results.append({
            "trace": trace_file.name,
            "recall_at_10": r_at_10,
            "turns": result["turns"],
            "expected": expected,
            "recommended": result["final_recommendations"],
        })

        time.sleep(0.5)

    # Summary
    mean_recall = sum(r["recall_at_10"] for r in results) / len(results)
    print("=" * 60)
    print(f"Mean Recall@10: {mean_recall:.3f}")
    print(f"Avg turns: {sum(r['turns'] for r in results) / len(results):.1f}")
    
    # Save results
    with open("eval_results.json", "w") as f:
        json.dump({
            "mean_recall_at_10": mean_recall,
            "traces": results
        }, f, indent=2)
    print("Results saved to eval_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SHL Recommender")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the service")
    parser.add_argument("--traces", default="traces/", help="Directory with trace JSON files")
    args = parser.parse_args()
    
    run_eval(args.url, args.traces)

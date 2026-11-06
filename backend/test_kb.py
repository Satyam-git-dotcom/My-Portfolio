import json
import os

def test_knowledge_base():
    kb_path = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
    assert os.path.exists(kb_path), f"knowledge_base.json not found at {kb_path}"
    
    with open(kb_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Required keys validation
    required_keys = [
        "profile",
        "contact",
        "education",
        "work_experience",
        "projects",
        "technical_skills",
        "certifications_and_achievements",
        "frequently_asked_questions",
        "strict_guardrails"
    ]
    for k in required_keys:
        assert k in data, f"Missing required key: {k}"
        
    # Profile checks
    assert data["profile"]["full_name"] == "Satyam"
    assert data["education"]["cgpa"] == "9.3 / 10.0"
    
    # Projects checks
    projects = data["projects"]
    assert len(projects) >= 3, "Expected at least 3 projects"
    flagship = next(p for p in projects if p.get("is_flagship"))
    assert flagship["name"] == "Synapse"
    
    print("✅ Knowledge Base JSON verified successfully!")
    print(f"Candidate: {data['profile']['full_name']}")
    print(f"Projects count: {len(projects)} (Flagship: {flagship['name']})")
    print(f"Strict Guardrails: {len(data['strict_guardrails']['allowed_topics'])} allowed topics configured.")

if __name__ == "__main__":
    test_knowledge_base()


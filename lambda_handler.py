"""
AWS Lambda Handler for Satyam's Portfolio AI Copilot
Powered by Amazon Bedrock (Anthropic Claude 3.5 Sonnet / Amazon Nova)

DEPLOYMENT INSTRUCTIONS:
1. Requirements: Python 3.12, boto3 (included in AWS Lambda runtime).
2. IAM Permissions:
   Attach 'bedrock:InvokeModel' to the Lambda Execution Role.
3. Environment Variables (Optional):
   - BEDROCK_MODEL_ID: "apac.anthropic.claude-3-5-sonnet-20241022-v2:0" (default)
   - BEDROCK_FALLBACK_MODEL_ID: "apac.amazon.nova-micro-v1:0"
4. Trigger:
   Connect to an API Gateway HTTP API or Lambda Function URL with CORS enabled.
"""

import json
import os
import random
import boto3
from botocore.exceptions import ClientError

# Configuration
DEFAULT_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "apac.anthropic.claude-3-5-sonnet-20241022-v2:0")
FALLBACK_MODEL_ID = os.environ.get("BEDROCK_FALLBACK_MODEL_ID", "apac.amazon.nova-micro-v1:0")

# Initialize Bedrock Runtime client
# Selected Region: ap-southeast-2 (or current execution region)
REGION = os.environ.get("AWS_REGION", "ap-southeast-2")
bedrock_runtime = boto3.client("bedrock-runtime", region_name=REGION)

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
    "Access-Control-Allow-Methods": "OPTIONS,POST",
    "Content-Type": "application/json"
}

RANDOM_FACTS = [
    "The first computer bug was an actual moth found trapped between relays in the Harvard Mark II computer by Grace Hopper's team in 1947.",
    "Ada Lovelace published the first machine algorithm in 1843 for Charles Babbage's Analytical Engine, becoming the world's first computer programmer.",
    "Margaret Hamilton coined the term 'Software Engineering' while architecting the Apollo 11 Guidance Computer flight software, which ran on only 4KB of RAM.",
    "The concept of Vector Space Models for semantic information retrieval was pioneered in the 1960s by Gerard Salton at Cornell University.",
    "Claude Shannon proved in 1937 that electrical switches could evaluate Boolean algebra, forming the foundational bridge between mathematics and electronic computing."
]

SYSTEM_PROMPT = """You are the official AI Twin of Satyam, representing Satyam authentically to recruiters, engineering managers, and visitors.

CAPABILITIES & BEHAVIORAL GUIDELINES:
1. GREETINGS & CASUAL COURTESY: Warmly welcome greetings like "Hi", "Hii", "Hello", "Hey", "How are you?", "Who are you?", and "What can you do?". Introduce yourself as Satyam's digital twin and portfolio copilot, and invite the visitor to explore his projects and engineering background.
2. BEHAVIORAL & RECRUITER QUESTIONS: Confidently and professionally answer behavioral interview questions about Satyam:
   - Why hire Satyam / Strengths: Strong theoretical CS core (9.3 CGPA at VIT, 200+ LeetCode problems, custom HNSW & KD-Tree indexing from scratch in Java), end-to-end full-stack delivery (FastAPI, Next.js, Spring Boot), fast learning and real-world ML impact at Infotact Solutions.
   - Challenges & High Pressure: Deconstructs complex ambiguous issues systematically, benchmarks bottlenecks with telemetry rather than guessing (e.g. optimizing Synapse graph traversal), and communicates transparently under deadlines.
   - Engineering Philosophy & Style: Swiss Bauhaus "Form follows function" — clean, type-safe, maintainable code over unnecessary bloat; high observability; strong collaborative team spirit.
   - Career Goals & Availability: Seeking Software Engineering internships and full-time roles (2026/2027) in Backend, AI/ML, or Distributed Systems. Based in India (Vellore / Uttar Pradesh), fully open to Remote, Hybrid, and On-Site relocation globally.
3. GROUNDING & RESTRICTION CONSTRAINT (CRITICAL RULE — ZERO HALLUCINATIONS / OUT-OF-SCOPE FALLBACK):
   - You only answer questions regarding Satyam, his engineering projects, technical stack, education, experience, career goals, behavioral fit, and contact details.
   - If a user asks questions completely UNRELATED to Satyam or his engineering career (such as general world trivia, geography, cooking recipes, sports scores, or unrelated school homework), you MUST refuse politely and reply:
     "That query is outside Satyam's engineering portfolio, but here is an interesting computer science fact for you:
     [Insert one interesting computer science or AI history fact here]
     Feel free to ask about Satyam's vector engine Synapse, his full-stack platforms, or his academic background at VIT!"
   - Never invent facts, companies, or metrics beyond what is verified in the knowledge base below.

<satyam_knowledge_base>
PROFILE:
- Name: Satyam
- Headline: Software Engineer | AI/ML & Distributed Systems
- Education: Vellore Institute of Technology (VIT), 2023 — 2027 (Pre-final year student)
- Degree: B.Tech in Computer Science & Engineering (AI/ML)
- Academic Standing: Cumulative GPA 9.3 / 10.0
- Contact: Email: satyam12303@gmail.com | Phone: +91 8218385753 | Location: Vellore / Uttar Pradesh, India
- Profiles: LinkedIn: linkedin.com/in/satyam-dot-com | GitHub: github.com/Satyam-git-dotcom
- Career Goals: Actively seeking Software Engineering internships and Full-Time roles (2026/2027) in Backend Engineering, AI/ML Infrastructure, or Distributed Systems. Fully open to Remote, Hybrid, and On-Site relocation globally.
- Behavioral Traits: Rigorous problem solver (200+ LeetCode problems), low-level systems enthusiasm (custom HNSW implementation), telemetry-first debugging, collaborative communicator.

EXPERIENCE:
- Company: Infotact Solutions
- Role: Data Associate L1 (Intern)
- Timeline: Oct 2026 — Present
- Details: Collaborates with cross-functional teams to deploy statistical and machine learning models with Python, scikit-learn, and TensorFlow. Preprocesses structured and unstructured datasets, performs Exploratory Data Analysis (EDA) with Matplotlib and Seaborn, and documents experiments for reproducibility.

FLAGSHIP PROJECTS:
1. Synapse // Custom LLM/RAG Vector Engine
   - Stack: Java, Spring Boot, Ollama, HNSW, KD-Tree, REST APIs
   - Summary: Implemented custom HNSW (Hierarchical Navigable Small World) and KD-Tree vector search algorithms from scratch in Java & Spring Boot, achieving sub-millisecond query latency on benchmark endpoints. Architected an offline-first RAG chatbot with 14 REST endpoints and built a dual embedding pipeline supporting 16-dim demo vectors and 768-dim semantic embeddings via Ollama nomic-embed-text.
   - GitHub: https://github.com/Satyam-git-dotcom/MY-OWN-LLM

2. FasalWatch // Autonomous Climate Resilience Agent
   - Stack: Python, Strands Agents SDK, AWS DynamoDB, Amazon Translate, Amazon SNS, FastAPI
   - Summary: Autonomous multi-agent climate resilience system that analyzes 7-day microclimate forecasts, detects crop-stage thermal stress, synthesizes low-cost farm interventions, and delivers Grade-5 mother-tongue advisories over basic SMS before heatwaves strike.
   - GitHub: https://github.com/Satyam-git-dotcom/FasalWatch

3. EconoCausal // Dynamic Pricing & Causal AI Uplift
   - Stack: Python, EconML, Microsoft DoWhy, LightGBM, SciPy Linear Programming, FastAPI, React
   - Summary: Causal AI promotional allocation platform utilizing Double Machine Learning (DML) with Neyman orthogonality to estimate Individual Treatment Effects (ITE) and strip away selection bias. Uses SciPy Highs LP Prescriptive Optimizer to allocate budget across discount tiers ($0, $10, $20), preventing coupon bleeding on organic conversions.
   - GitHub: https://github.com/Satyam-git-dotcom/EconoCausal-Dynamic-Pricing

4. Kore Logic // AI Career & Learning Platform
   - Stack: Next.js 15, FastAPI, PostgreSQL, LangChain, ChromaDB, SQLAlchemy
   - Summary: Full-stack career acceleration platform with 19 REST API endpoints and a normalized PostgreSQL schema across 10 core entity tables. Features a LangChain & ChromaDB mentor chatbot achieving sub-200ms latency on semantic searches, and a cosine-similarity matching engine over skill embeddings.
   - GitHub: https://github.com/Satyam-git-dotcom/Kore-Logic

5. Aura Tender // AI Procurement Compliance Platform
   - Stack: Python, FastAPI, React, Gemini 1.5 Pro, RAG
   - Summary: AI procurement compliance evaluation platform automating multimodal document analysis, cutting manual review effort by over 80% per tender. Deployed a zero-shot Gemini 1.5 Pro extraction pipeline that cross-references bidder evidence against compliance checklists with human-in-the-loop flagging.
   - GitHub: https://github.com/Satyam-git-dotcom/Aura-Tender

SKILLS & PROFICIENCIES:
- Languages: Java, Python, JavaScript, SQL
- Core CS: Data Structures & Algorithms (200+ LeetCode problems solved), System Design, OOP, DBMS, Operating Systems
- Backend & Frameworks: Spring Boot, FastAPI, Node.js/Express, Next.js 15, React, REST APIs
- AI / Vector Search: RAG, Vector Search, HNSW, KD-Tree, LangChain, LangGraph, ChromaDB, Scikit-Learn, TensorFlow, NLTK
- Cloud & Tools: AWS Bedrock, Docker, Git, CI/CD, Postman
</satyam_knowledge_base>
"""

def lambda_handler(event, context):
    http_method = event.get("httpMethod") or event.get("requestContext", {}).get("http", {}).get("method", "")

    if http_method == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": CORS_HEADERS,
            "body": json.dumps({"status": "ok"})
        }

    if http_method != "POST":
        return {
            "statusCode": 405,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": f"Method {http_method} not allowed. Use POST."})
        }

    try:
        body = event.get("body", "{}")
        payload = json.loads(body) if isinstance(body, str) else (body or {})

        user_message = payload.get("message", "").strip()
        history = payload.get("history", [])

        if not user_message:
            return {
                "statusCode": 400,
                "headers": CORS_HEADERS,
                "body": json.dumps({"error": "Missing 'message' in request body."})
            }

        # Cap message length to prevent token abuse (Task 3 requirement)
        if len(user_message) > 1200:
            user_message = user_message[:1200]

        # Prepare messages array
        messages = []
        for turn in history[-6:]:
            role = "user" if turn.get("sender") in ["user", "YOU"] else "assistant"
            content = turn.get("text", "").strip()
            if content:
                messages.append({"role": role, "content": [{"text": content}]})

        messages.append({"role": "user", "content": [{"text": user_message}]})

        # Invoke Bedrock with primary model and automatic fallback
        active_model = DEFAULT_MODEL_ID
        try:
            response = bedrock_runtime.converse(
                modelId=active_model,
                messages=messages,
                system=[{"text": SYSTEM_PROMPT}],
                inferenceConfig={
                    "maxTokens": 600,
                    "temperature": 0.2,
                    "topP": 0.9
                }
            )
        except ClientError as e:
            err_code = e.response.get("Error", {}).get("Code", "")
            err_msg = e.response.get("Error", {}).get("Message", "")
            print(f"Primary model {active_model} returned {err_code}: {err_msg}. Switching to fallback...")
            active_model = FALLBACK_MODEL_ID
            response = bedrock_runtime.converse(
                modelId=active_model,
                messages=messages,
                system=[{"text": SYSTEM_PROMPT}],
                inferenceConfig={
                    "maxTokens": 600,
                    "temperature": 0.2,
                    "topP": 0.9
                }
            )

        output_text = response["output"]["message"]["content"][0]["text"]

        return {
            "statusCode": 200,
            "headers": CORS_HEADERS,
            "body": json.dumps({
                "response": output_text,
                "model": active_model
            })
        }

    except Exception as e:
        print(f"Error executing Bedrock inference: {str(e)}")
        # If model failure occurs, graceful fact fallback
        fallback_fact = random.choice(RANDOM_FACTS)
        return {
            "statusCode": 200,
            "headers": CORS_HEADERS,
            "body": json.dumps({
                "response": f"I had trouble connecting to my cloud neural runtime, but here is an authentic computer science fact for you:\n\n💡 {fallback_fact}\n\nFeel free to ask about Satyam's custom vector engine Synapse, his AI career platform Kore Logic, or his academics at VIT!",
                "fallback": True
            })
        }

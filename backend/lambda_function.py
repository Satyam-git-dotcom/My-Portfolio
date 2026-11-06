import json
import os
import boto3
from botocore.exceptions import ClientError

# Configuration
# Selected Region: ap-southeast-2
# Primary model: Claude 3.5 Sonnet v2 via APAC regional inference profile
DEFAULT_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "apac.anthropic.claude-3-5-sonnet-20241022-v2:0")
FALLBACK_MODEL_ID = os.environ.get("BEDROCK_FALLBACK_MODEL_ID", "apac.amazon.nova-micro-v1:0")

# Initialize Bedrock Runtime client in Lambda selected region (ap-southeast-2)
bedrock_runtime = boto3.client("bedrock-runtime", region_name="ap-southeast-2")

# Load System Prompt
SYSTEM_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
try:
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
except Exception:
    SYSTEM_PROMPT = "You are the AI Twin of Satyam, an AI Systems Engineer and student at VIT (9.3 CGPA)."

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
    "Access-Control-Allow-Methods": "OPTIONS,POST",
    "Content-Type": "application/json"
}

def lambda_handler(event, context):
    """
    AWS Lambda handler for Satyam's Portfolio AI Chatbot powered by Claude on Amazon Bedrock.
    Supports CORS preflight (OPTIONS) and chat completions (POST).
    """
    http_method = event.get("httpMethod") or event.get("requestContext", {}).get("http", {}).get("method", "")

    # Handle CORS Preflight
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
        if isinstance(body, str):
            payload = json.loads(body) if body else {}
        else:
            payload = body or {}

        user_message = payload.get("message", "").strip()
        history = payload.get("history", [])

        if not user_message:
            return {
                "statusCode": 400,
                "headers": CORS_HEADERS,
                "body": json.dumps({"error": "Missing 'message' field in request body."})
            }

        # Guardrail: Prevent excessive token abuse
        if len(user_message) > 1200:
            user_message = user_message[:1200]

        # Prepare Bedrock Converse API messages format
        messages = []
        for turn in history[-6:]:  # Keep recent 3 dialogue turns for context
            role = "user" if turn.get("sender") in ["user", "YOU"] else "assistant"
            content = turn.get("text", "").strip()
            if content:
                messages.append({
                    "role": role,
                    "content": [{"text": content}]
                })

        # Append current user query
        messages.append({
            "role": "user",
            "content": [{"text": user_message}]
        })

        # Attempt invocation with primary model (Claude 3.5 Sonnet)
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
            print(f"Primary model {active_model} returned {err_code}: {err_msg}. Executing fallback...")
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
        return {
            "statusCode": 500,
            "headers": CORS_HEADERS,
            "body": json.dumps({
                "error": "Internal AI server error",
                "details": str(e)
            })
        }

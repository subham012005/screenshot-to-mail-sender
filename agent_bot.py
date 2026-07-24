import os
import time
import threading
import requests
import mimetypes
import base64
import json
from typing import Optional, List
from email.message import EmailMessage

# Gmail API
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from http.server import BaseHTTPRequestHandler, HTTPServer

# Langchain / LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ================== ENV SETUP ==================
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
if not BOT_TOKEN or not CHAT_ID:
    print("Warning: BOT_TOKEN or CHAT_ID not set in .env file")

BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

# Path to resume, resolved relative to this script — works on Windows AND Render/Linux
RESUME_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resume.pdf')

# ================== TELEGRAM FUNCTIONS ==================
def send_telegram_message(text: str, retries: int = 3):
    url = f"{BASE_URL}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}
    for attempt in range(retries):
        try:
            res = requests.post(url, json=payload, timeout=30)
            data = res.json()
            if data.get("ok"):
                return data
            print("Telegram API error:", data)
        except requests.exceptions.RequestException as e:
            print(f"[SendMessage] Attempt {attempt + 1} failed:", e)
            time.sleep(5)
    return None

def download_telegram_file(file_id: str, dest_dir: str = "downloads") -> str | None:
    url = f"{BASE_URL}/getFile"
    try:
        res = requests.get(url, params={"file_id": file_id}, timeout=30)
        data = res.json()
        if not data.get("ok"):
            print("Telegram API error (getFile):", data)
            return None
            
        file_path = data["result"]["file_path"]
        download_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
        dl_res = requests.get(download_url, timeout=60)
        dl_res.raise_for_status()
        
        os.makedirs(dest_dir, exist_ok=True)
        filename = os.path.basename(file_path)
        if not filename:
            filename = f"file_{file_id}.unknown"
            
        local_path = os.path.join(dest_dir, filename)
        with open(local_path, "wb") as f:
            f.write(dl_res.content)
            
        return local_path
    except Exception as e:
        print(f"[DownloadFile] Error: {e}")
        return None

def fetch_telegram_messages(offset: int | None = None):
    url = f"{BASE_URL}/getUpdates"
    params = {"timeout": 50}
    if offset is not None:
        params["offset"] = offset
    try:
        res = requests.get(url, params=params, timeout=70)
        data = res.json()
        if not data.get("ok"):
            print("Telegram API error:", data)
            return []
        return data["result"]
    except requests.exceptions.ReadTimeout:
        return []
    except requests.exceptions.RequestException as e:
        print("[FetchUpdates] Network error:", e)
        time.sleep(10)
        return []

# Load resume at startup
resume_context = ""
try:
    with open("resume.txt", "r", encoding="utf-8") as f:
        resume_context = f.read()
except Exception:
    pass

# ================== HTML EMAIL TEMPLATE ==================
# Defined at module level (column 0) to avoid f-string indentation SyntaxErrors.
# {{BODY}} is replaced at send time via str.replace().
HTML_EMAIL_TEMPLATE = (
    "<!DOCTYPE html>"
    "<html lang='en'>"
    "<head>"
    "<meta charset='UTF-8'>"
    "<meta name='viewport' content='width=device-width, initial-scale=1.0'>"
    "</head>"
    "<body style='margin:0;padding:0;background-color:#f4f6f8;"
    "font-family:Arial,sans-serif;'>"
    "<table width='100%' cellpadding='0' cellspacing='0' "
    "style='background-color:#f4f6f8;padding:30px 0;'>"
    "<tr><td align='center'>"
    "<table width='620' cellpadding='0' cellspacing='0' "
    "style='background:#ffffff;border-radius:10px;overflow:hidden;"
    "box-shadow:0 2px 12px rgba(0,0,0,0.08);'>"
    "<tr><td style='background:linear-gradient(135deg,#1a1a2e 0%,#16213e 60%,#0f3460 100%);"
    "padding:28px 36px;'>"
    "<p style='margin:0;color:#ffffff;font-size:18px;font-weight:700;"
    "letter-spacing:0.5px;'>{{SUBJECT}}</p>"
    "</td></tr>"
    "<tr><td style='padding:36px 36px 32px;'>"
    "<div style='color:#1e293b;font-size:15px;line-height:1.8;'>{{BODY}}</div>"
    "</td></tr>"
    "</table></td></tr></table></body></html>"
)

# ================== EMAIL & AI TOOLS ==================

class SendEmailInput(BaseModel):
    to: str = Field(description="The email address of the primary recipient")
    subject: str = Field(description="The subject of the email")
    body: str = Field(description="The body content of the email")
    cc: Optional[str] = Field(None, description="Comma-separated email addresses for CC")
    bcc: Optional[str] = Field(None, description="Comma-separated email addresses for BCC")
    attachment_paths: Optional[List[str]] = Field(None, description="List of file paths for attachments")
    use_html: bool = Field(True, description="If True, send as a styled HTML email. If False, send as plain text.")

@tool("send_email", args_schema=SendEmailInput)
def send_email(to: str, subject: str, body: str, cc: Optional[str] = None, bcc: Optional[str] = None, attachment_paths: Optional[List[str]] = None, use_html: bool = True) -> str:
    """Sends an email using Gmail API (OAuth2) with optional attachments, CC, and BCC. use_html controls whether to send styled HTML or plain text."""
    SENDER_EMAIL = os.getenv("SENDER_EMAIL")
    if not SENDER_EMAIL:
        return "Error: SENDER_EMAIL environment variable must be set."

    # --- Load credentials from Render Secret File or local token.json ---
    # On Render: store token.json as a Secret File at path /etc/secrets/token.json
    # Locally: place token.json in the same directory as this script
    TOKEN_PATH = os.environ.get(
        "GMAIL_TOKEN_PATH",
        "/etc/secrets/token.json" if os.path.exists("/etc/secrets/token.json")
        else os.path.join(os.path.dirname(os.path.abspath(__file__)), "token.json")
    )
    CREDS_PATH = os.environ.get(
        "GMAIL_CREDS_PATH",
        "/etc/secrets/credentials.json" if os.path.exists("/etc/secrets/credentials.json")
        else os.path.join(os.path.dirname(os.path.abspath(__file__)), "credentials.json")
    )

    SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

    try:
        creds = None
        if os.path.exists(TOKEN_PATH):
            creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                # Persist refreshed token back
                with open(TOKEN_PATH, "w") as f:
                    f.write(creds.to_json())
            else:
                return ("Error: Gmail token missing or expired. "
                        "Run generate_token.py locally to create token.json, "
                        "then upload it as a Secret File on Render.")
    except Exception as e:
        return f"Error loading Gmail credentials: {str(e)}"

    # --- Normalize body: collapse hard mid-sentence line breaks into spaces ---
    # The AI wraps lines at ~70 chars using \n. Plain-text email clients render
    # those as actual line breaks, causing the narrow/cluttered look.
    # We rejoin lines within the same paragraph into one long line for plain text.
    # However, we preserve intra-paragraph newlines in the HTML version so that
    # multi-line blocks (e.g., the signature) render correctly with <br> tags.
    raw_paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    # Plain-text: collapse each paragraph's lines into a single long line
    normalized_paragraphs = [" ".join(line.strip() for line in p.splitlines() if line.strip()) for p in raw_paragraphs]
    body = "\n\n".join(normalized_paragraphs)

    # --- Optionally convert plain-text body to styled HTML ---
    html_body = None
    if use_html:
        # HTML: preserve intra-paragraph line breaks as <br> so multi-line
        # blocks like the signature are not squished into one line.
        html_paragraphs = "".join(
            f"<p>{'<br>'.join(line.strip() for line in para.splitlines() if line.strip())}</p>"
            for para in raw_paragraphs
        )
        html_body = (
            HTML_EMAIL_TEMPLATE
            .replace("{{SUBJECT}}", subject)
            .replace("{{BODY}}", html_paragraphs)
        )

    # --- Build the MIME message ---
    def _build_attachment_parts(outer_msg):
        for path in (attachment_paths or []):
            if not os.path.exists(path):
                return f"Error: Attachment not found at {path}"
            ctype, _ = mimetypes.guess_type(path)
            if ctype is None:
                ctype = "application/octet-stream"
            maintype, subtype = ctype.split("/", 1)
            part = MIMEBase(maintype, subtype)
            with open(path, "rb") as f:
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", "attachment", filename=os.path.basename(path))
            outer_msg.attach(part)
        return None

    if use_html:
        assert html_body is not None  # always set above when use_html=True
        if attachment_paths:
            msg = MIMEMultipart("mixed")
            alt = MIMEMultipart("alternative")
            alt.attach(MIMEText(body, "plain", "utf-8"))
            alt.attach(MIMEText(html_body, "html", "utf-8"))
            msg.attach(alt)
            err = _build_attachment_parts(msg)
            if err: return err
        else:
            msg = MIMEMultipart("alternative")
            msg.attach(MIMEText(body, "plain", "utf-8"))
            msg.attach(MIMEText(html_body, "html", "utf-8"))
    else:
        # Plain text only
        if attachment_paths:
            msg = MIMEMultipart("mixed")
            msg.attach(MIMEText(body, "plain", "utf-8"))
            err = _build_attachment_parts(msg)
            if err: return err
        else:
            msg = MIMEText(body, "plain", "utf-8")

    msg["Subject"] = subject
    msg["From"]    = SENDER_EMAIL
    msg["To"]      = to
    if cc:  msg["Cc"]  = cc
    if bcc: msg["Bcc"] = bcc

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()

    try:
        service = build("gmail", "v1", credentials=creds)
        service.users().messages().send(  # type: ignore[attr-defined]
            userId="me", body={"raw": raw}
        ).execute()
        return f"Email successfully sent to {to} via Gmail API."
    except Exception as e:
        return f"Failed to send email via Gmail API: {str(e)}"


class DraftEmailFromImagesInput(BaseModel):
    image_paths: List[str] = Field(description="List of local file paths to the images to analyze.")
    instructions: Optional[str] = Field(None, description="Any extra instructions or context provided by the user.")

@tool("draft_email_from_images", args_schema=DraftEmailFromImagesInput)
def draft_email_from_images(image_paths: List[str], instructions: Optional[str] = None) -> str:
    """Analyzes one or more images and generates a structured email draft based on their contents."""
    content_list = []
    
    for image_path in image_paths:
        if not os.path.exists(image_path):
            return f"Error: Image not found at {image_path}"
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                content_list.append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                )
        except Exception as e:
            return f"Error reading image: {str(e)}"
            
    prompt_text = (
        "Analyze these images. The user is a job seeker / freelancer. The images likely show a Job Description (JD), "
        "a freelance contract, or a lead posted by SOMEONE ELSE. Your task is to draft an email on behalf of the user "
        "APPLYING for this role or pitching their services to the person who posted it. Do NOT write the email as if the user "
        "is the one hiring or posting the job.\n\n"
        
        "CRITICAL - SKILL CHECK:\n"
        f"Here is the user's resume:\n---\n{resume_context}\n---\n"
        "First, evaluate if this job is completely unrelated to the user's skills. If it is a massive mismatch, "
        "do NOT draft an email. Instead, return exactly: 'SKILLS_MISMATCH: [explain the mismatch].'\n\n"
        
        "If the image is completely ambiguous, return exactly: 'CLARIFICATION_NEEDED: [explain what you are confused about]'.\n\n"
        
        "Otherwise, draft a completely finalized, ready-to-send email based on its content. Use the user's resume to fill in actual details, "
        "skills, and their name so there are absolutely no placeholders (like [Name] or [Company]). "
        "Extract the recipient email from the image. If there is no email visible, put 'UNKNOWN_EMAIL'.\n\n"
        "IMPORTANT: The email will be sent as plain text. Do NOT use Markdown link formatting (e.g., [LinkedIn](https...)). Instead, write out links fully like 'LinkedIn: https://...'\n\n"
        f"IMPORTANT: The user wants to attach their resume. The exact absolute path to their resume is: '{RESUME_PATH}'. You MUST include this exact path in the 'Attachments:' field of your drafted response.\n\n"
        f"Format your response exactly like this:\nTo: extracted_email@example.com\nCC: \nBCC: \nSubject: Your Subject Here\nContent: The full email body here\nAttachments: {RESUME_PATH}"
    )
    
    if instructions:
        prompt_text += f"\n\nUser instructions: {instructions}"
        
    content_list.insert(0, {"type": "text", "text": prompt_text})
    
    try:
        vision_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        message = HumanMessage(content=content_list)
        response = vision_llm.invoke([message])
        return str(response.content)
    except Exception as e:
        return f"Error during image analysis: {str(e)}"

# ================== LANGGRAPH AGENT SETUP ==================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [send_email, draft_email_from_images]

SYSTEM_PROMPT = f"""You are an email drafting assistant for a job seeker / freelancer.
When a user sends one or more images, use the `draft_email_from_images` tool to generate a draft.

If the tool returns a message starting with 'SKILLS_MISMATCH:', ask the user if they still want to apply despite the mismatch.
If the tool returns a message starting with 'CLARIFICATION_NEEDED:', ask the user to clarify their intent or the image contents before proceeding. Do NOT guess.

Otherwise, show the drafted email to the user exactly like this:
To: (the actual email address)
CC: (if any)
BCC: (if any)
Subject: (the actual subject line)
Content: (the full, finalized email body)
Attachments: (if any)

CRITICAL RULES:
1. NEVER use placeholders (like [Your Name], [Company Name], etc.). Use the user's resume context to fill in all details, or deduce them from the image.
2. DO NOT send the email yet. Ask the user for approval.
3. Once the user approves sending, ask: "Do you want to send this as a styled HTML email or plain text?" Wait for their answer.
   - If they say "html", "styled", or similar → set use_html=True when calling send_email.
   - If they say "plain", "simple", "text", or similar → set use_html=False when calling send_email.
4. Then call `send_email` with use_html set accordingly. MUST pass the attachment path from the draft into the `attachment_paths` argument.
5. If they ask for changes, update the draft, show it to them again, and ask for approval again.

Here is the user's resume and professional background. Use this context if you need to discuss their skills:
{resume_context}
"""

agent_executor = create_react_agent(llm, tools)
chat_history: list[SystemMessage | HumanMessage | AIMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

# ================== MAIN APP LOGIC ==================
def process_user_message(message_text: str):
    print(f"\n[Telegram] Processing message: {message_text}")
    try:
        chat_history.append(HumanMessage(content=message_text))
        result = agent_executor.invoke({"messages": chat_history})
        final_messages = result["messages"]
        response = final_messages[-1].content
        chat_history.append(AIMessage(content=response))
        
        if len(chat_history) > 20:
            chat_history.pop(1)
            chat_history.pop(1)
            
    except Exception as e:
        response = f"Sorry, I encountered an error: {e}"
        
    print(f"[Agent] Replied: {response}")
    send_telegram_message(response)


def poll_telegram():
    offset = None
    media_groups = {}
    print("Telegram polling started...")
    
    while True:
        try:
            updates = fetch_telegram_messages(offset)
            current_time = time.time()
            
            for update in updates:
                offset = update["update_id"] + 1
                
                if "message" in update:
                    msg = update["message"]
                    
                    if "photo" in msg or "document" in msg:
                        file_id = None
                        if "photo" in msg:
                            file_id = msg["photo"][-1]["file_id"]
                        else:
                            file_id = msg["document"]["file_id"]
                            
                        caption = msg.get("caption", "")
                        print(f"\n[Telegram] Received file/photo, downloading...")
                        local_path = download_telegram_file(file_id)
                        
                        media_group_id = msg.get("media_group_id")
                        
                        if media_group_id:
                            if media_group_id not in media_groups:
                                media_groups[media_group_id] = {"paths": [], "caption": "", "last_updated": current_time}
                            if local_path:
                                media_groups[media_group_id]["paths"].append(local_path)
                            if caption:
                                media_groups[media_group_id]["caption"] += caption + " "
                            media_groups[media_group_id]["last_updated"] = current_time
                        else:
                            # Not an album, process immediately
                            if local_path:
                                message_text = f"User sent an image/file. Saved locally at: {local_path}. Caption: {caption}. Please analyze this using draft_email_from_images."
                            else:
                                message_text = f"User sent a file, but I failed to download it. Caption: {caption}"
                            process_user_message(message_text)
                            
                    elif "text" in msg:
                        process_user_message(msg["text"])
                        
            # Check for completed media groups (no new parts for 2 seconds)
            completed_groups = []
            for mg_id, data in media_groups.items():
                if current_time - data["last_updated"] > 2.0:
                    completed_groups.append(mg_id)
                    
            for mg_id in completed_groups:
                data = media_groups.pop(mg_id)
                paths_str = str(data["paths"])
                caption = data["caption"].strip()
                message_text = f"User sent {len(data['paths'])} images in an album. Saved locally at: {paths_str}. Caption: {caption}. Please analyze all of them using draft_email_from_images."
                process_user_message(message_text)

            time.sleep(1)
        except Exception as e:
            print(f"Error polling telegram: {e}")
            time.sleep(5)

def keep_alive():
    """A simple HTTP server to satisfy Render's port binding requirement."""
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"Bot is alive!")
            
        def log_message(self, format, *args):
            # Suppress logs for the self-ping to avoid clutter
            pass

    port = int(os.environ.get('PORT', 8080))
    server = HTTPServer(('0.0.0.0', port), Handler)
    print(f"Starting keep-alive server on port {port}")
    server.serve_forever()

def self_ping():
    """Pings the external URL every 10 minutes to prevent sleep."""
    url = os.environ.get('RENDER_EXTERNAL_URL')
    if not url:
        print("RENDER_EXTERNAL_URL not set, self-ping disabled.")
        return
    while True:
        try:
            time.sleep(600) # 10 minutes
            res = requests.get(url)
            print(f"Self-ping to {url} status code: {res.status_code}")
        except Exception as e:
            print(f"Self-ping failed: {e}")

def main():
    # Start the keep-alive server
    threading.Thread(target=keep_alive, daemon=True).start()
    
    # Start the self-ping thread
    threading.Thread(target=self_ping, daemon=True).start()

    t = threading.Thread(target=poll_telegram, daemon=True)
    t.start()
    
    send_telegram_message("🤖 Unified Bot is awake and ready! Multi-image support enabled.")
    print("Terminal input ready. Type a message to send to Telegram (or 'exit' to quit):")
    
    while True:
        try:
            cmd = input()
            if cmd.strip().lower() == 'exit':
                break
            if cmd.strip():
                send_telegram_message(f"From Terminal: {cmd}")
        except EOFError:
            # If running as a background service (e.g. Render), stdin is closed.
            # Sleep forever to keep the main thread alive.
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()

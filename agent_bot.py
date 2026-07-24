import os
import time
import threading
import re
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

# Web search
from duckduckgo_search import DDGS

# ================== ENV SETUP ==================
try:
    dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    load_dotenv(dotenv_path)
except Exception as e:
    print(f"Warning: Failed to load .env file dynamically ({e}). Loading from environment directly.")


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

    # --- Normalize body: collapse hard mid-sentence line breaks into spaces while preserving list and signature structures ---
    def _normalize_para(p_text: str, is_l: bool) -> str:
        lines = [line.strip() for line in p_text.splitlines() if line.strip()]
        if not lines:
            return ""
            
        # Check if contains list items
        has_list = any(
            l.startswith("-") or 
            l.startswith("*") or 
            l.startswith("•") or 
            (l and l[0].isdigit() and "." in l.split()[0])
            for l in lines
        )
        if has_list:
            return "\n".join(lines)
            
        # Check if signature or contact info block
        avg_len = sum(len(l) for l in lines) / len(lines)
        is_cnt = any(
            "@" in l or 
            "http" in l or 
            any(c.isdigit() for c in l) and len(l) < 25 
            for l in lines
        )
        if is_l or avg_len < 45 or (is_cnt and len(lines) > 1):
            return "\n".join(lines)
        else:
            return " ".join(lines)

    raw_paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    normalized_paragraphs = []
    for idx, para in enumerate(raw_paragraphs):
        is_last_para = (idx == len(raw_paragraphs) - 1)
        normalized_paragraphs.append(_normalize_para(para, is_last_para))
        
    body = "\n\n".join(normalized_paragraphs)

    # --- Convert plain-text body to beautiful, styled HTML ---
    html_body = None
    if use_html:
        # Convert **bold** to <strong>bold</strong>
        processed_text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", body)
        
        # Convert markdown links [text](url) to anchor tags
        processed_text = re.sub(
            r"\[(.*?)\]\((.*?)\)", 
            r'<a href="\2" style="color: #0f3460; text-decoration: underline;">\1</a>', 
            processed_text
        )
        
        # Convert blocks to paragraphs and lists
        blocks = [b.strip() for b in processed_text.split("\n\n") if b.strip()]
        html_blocks = []
        
        for block in blocks:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if not lines:
                continue
                
            # Determine if the block contains list items
            has_list_item = any(
                line.startswith("-") or 
                line.startswith("*") or 
                line.startswith("•") or 
                (line and line[0].isdigit() and "." in line.split()[0])
                for line in lines
            )
            
            if not has_list_item:
                # Normal paragraph or greeting or signature.
                # A block should use <br> (keep line breaks) ONLY if:
                # - It is the very last block of the email (the signature)
                # - Or it looks like a contact block (contains email/phone/links/URLs and has short lines)
                # Otherwise, it is a regular text paragraph and MUST be joined with a space to utilize the full line length!
                is_last = (blocks.index(block) == len(blocks) - 1)
                is_contact = any(
                    "@" in l or 
                    "http" in l or 
                    any(c.isdigit() for c in l) and len(l) < 25 
                    for l in lines
                )
                
                if is_last or (is_contact and len(lines) > 1):
                    paragraph_content = "<br>".join(lines)
                else:
                    paragraph_content = " ".join(lines)
                    
                html_blocks.append(f"<p style='margin-top: 0; margin-bottom: 16px; line-height: 1.8;'>{paragraph_content}</p>")
            else:
                # Block has list items: convert items to standard <li> inside <ul>/<ol>
                html_list_items = []
                list_type = "ul"
                
                for line in lines:
                    match_bullet = re.match(r"^[-*•]\s+(.*)", line)
                    match_numbered = re.match(r"^(\d+)[.)]\s+(.*)", line)
                    
                    if match_bullet:
                        list_type = "ul"
                        html_list_items.append(f"<li style='margin-bottom: 8px;'>{match_bullet.group(1)}</li>")
                    elif match_numbered:
                        list_type = "ol"
                        html_list_items.append(f"<li style='margin-bottom: 8px;'>{match_numbered.group(2)}</li>")
                    else:
                        # Non-list item line inside list block
                        if html_list_items:
                            html_blocks.append(f"<{list_type} style='margin-top: 8px; margin-bottom: 16px; padding-left: 20px; color: #334155;'>{''.join(html_list_items)}</{list_type}>")
                            html_list_items = []
                        html_blocks.append(f"<p style='margin-top: 0; margin-bottom: 16px; line-height: 1.8;'>{line}</p>")
                        
                if html_list_items:
                    html_blocks.append(f"<{list_type} style='margin-top: 8px; margin-bottom: 16px; padding-left: 20px; color: #334155;'>{''.join(html_list_items)}</{list_type}>")
        
        html_paragraphs = "".join(html_blocks)
        html_body = (
            HTML_EMAIL_TEMPLATE
            .replace("{{SUBJECT}}", subject)
            .replace("{{BODY}}", html_paragraphs)
        )
        # Write debug files
        try:
            debug_dir = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(debug_dir, "last_debug_html.html"), "w", encoding="utf-8") as f:
                f.write(html_body)
            with open(os.path.join(debug_dir, "last_debug_plain.txt"), "w", encoding="utf-8") as f:
                f.write(body)
            print(f"[Debug] Wrote last_debug_html.html and last_debug_plain.txt in {debug_dir}")
        except Exception as e:
            print("[Debug] Error writing debug files:", e)



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

# ================== MEMORY TOOL ==================
# chat_history is declared here so clear_memory tool can reference it by name.
chat_history: list[SystemMessage | HumanMessage | AIMessage] = []

@tool("clear_memory")
def clear_memory() -> str:
    """Clears the entire conversation history and starts fresh.
    Call this whenever the user wants to forget, reset, clear, start over,
    wipe memory, begin again, or anything with similar intent."""
    global chat_history
    if chat_history:
        system_msg = chat_history[0]
        chat_history.clear()
        chat_history.append(system_msg)
    return "MEMORY_CLEARED"

# ================== WEB SEARCH TOOL ==================
@tool("web_search")
def web_search(query: str) -> str:
    """Search the web for recent, up-to-date information on any topic.

    Use this tool BEFORE drafting any proposal or outreach email to:
    - Research the latest trends, pain points, and innovations in the recipient's industry
    - Find the most effective current approaches for the type of proposal you're writing
    - Understand what makes proposals stand out RIGHT NOW (not 2 years ago)
    - Research the recipient's company/domain if mentioned

    Always run 2-3 targeted searches to get a well-rounded picture before writing.
    Prefer queries like: 'best practices 2025 [topic]', 'latest [industry] trends', '[topic] challenges 2025'.
    """
    try:
        results = []
        with DDGS() as ddgs:
            # First try: past month for freshest results
            hits = list(ddgs.text(query, max_results=5, timelimit="m"))
            if not hits:
                # Fallback: past year
                hits = list(ddgs.text(query, max_results=5, timelimit="y"))
            if not hits:
                hits = list(ddgs.text(query, max_results=5))

        for i, r in enumerate(hits, 1):
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            results.append(f"{i}. {title}\n   {body}\n   🔗 {href}")

        return "\n\n".join(results) if results else "No results found for this query."
    except Exception as e:
        return f"Search error: {str(e)}"


# ================== LANGGRAPH AGENT SETUP ==================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [send_email, draft_email_from_images, clear_memory, web_search]

SYSTEM_PROMPT = f"""You are an elite email strategist and ghostwriter for Subham Sharma, a freelancer/developer.
Your goal is NOT to write generic emails. Every email must feel handcrafted, specific, and high-converting.

=== YOUR WORKFLOW FOR PROPOSAL / OUTREACH EMAILS ===

STEP 1 — GATHER CONTEXT (ask questions FIRST if needed)
Before doing anything, check if you have enough context to write a highly specific email.
You NEED at minimum: the recipient's industry/domain, the service being proposed, and the recipient's email.
If ANY of these are missing or vague, ask the user targeted questions. Example:
  "To write the best possible proposal, I need a few details:
   1. Who is the recipient — what's their business/industry?
   2. What specific problem are you solving for them?
   3. Do you have their email address?"
DO NOT skip this step and write a generic email.

STEP 2 — RESEARCH (mandatory for proposals, strongly recommended for all outreach)
Use the `web_search` tool to find fresh, specific information:
- Run 2-3 searches on the recipient's industry/domain (e.g. "dental clinic automation challenges 2025", "best WhatsApp chatbot for healthcare 2025")
- Search for what makes great proposals in this niche RIGHT NOW
- Look for pain points, trends, statistics, and buzzwords that are current — not from 2 years ago
- If a search returns no results, try DIFFERENT keywords — never give up after one failed search.
The goal: your email should mention things they haven't heard a hundred times before.

STEP 3 — DRAFT (must sound like a real human, NOT an AI)
Using your research + user context + resume below, draft the email:
- NEVER use standard AI cliches like "I hope this email finds you well", "I hope you are doing well", "My name is Subham Sharma and...", or "I am reaching out to...".
- Open directly with a specific observation, hook, or casual greeting: e.g. "Hi [Name] -", "Hello [Name] -", or simply start with the point.
- Write in a direct, casual-professional, conversational tone. Write like a colleague or fellow business owner sending a quick, helpful note, not like a sales bot or corporate marketer.
- DO NOT use formal list structures with bolded bullet headers (e.g. "- **Feature Name:** Description"). Bulleted lists with bold titles scream "AI generated". Instead, use short, natural flowing sentences to describe features, or simple un-bolded bullet points if needed.
- Write each paragraph as a single continuous line (do NOT wrap lines or insert newlines at 70 characters). Only use single newlines for the signature block.
- Keep the entire email under 150 words. Be concise and high-impact.
- Close with a low-friction, natural question (e.g. "Worth a quick chat?", "Are you open to this?", "Let me know if you'd like to see a demo").


=== WHEN THE USER SENDS IMAGES ===
Use the `draft_email_from_images` tool first to extract context from the image.
Then follow the same STEP 2 → STEP 3 workflow above to enrich it with research.

=== RESUME / SENDER DETAILS (use these — never use placeholders) ===
{resume_context}

=== DRAFT FORMAT — always show the draft like this before sending ===
To: actual_email@example.com
Subject: Actual Subject Line Here
Content:
(full, finalized email body — no placeholders whatsoever)
Attachments: (resume path if relevant, else omit)

=== MEMORY MANAGEMENT ===
If the user wants to reset, clear, forget, start over, or anything similar → call `clear_memory` immediately.
After it returns, reply: "🧹 Memory cleared! Starting fresh."

=== CRITICAL RULES ===
1. ZERO PLACEHOLDERS. Never write bracketed placeholder text like [Recipient's Name], [Company Name], [Your Name], [Contact Info], etc.
   - If the recipient's name is unknown, greet with "Hi there,", "Dear Hiring Team,", or "Hello,". Never use a placeholder.
   - If the company name is unknown, say "your company" or "your business".
   - Use Subham's real details: Subham Sharma | +917988944185 | subham1401sh@gmail.com | https://linkedin.com/in/subham1401
2. ALWAYS ask clarifying questions if context is insufficient — never guess and write generic.
3. ALWAYS use web_search before drafting proposals. Generic emails get ignored.
4. After drafting, ask: "Should I send this?" — wait for approval.
5. Once approved, ask: "HTML (styled) or plain text?"
   - If the user specifies "plain" or "text" or "simple" → you MUST call send_email with use_html=False. This is a strict requirement. Do NOT use HTML.
   - If the user specifies "html", "styled", "fancy", "rich", or is unsure → call send_email with use_html=True.
   - If the user says "both" or "either" → call send_email with use_html=True (since HTML mode includes a plain-text fallback automatically).
   - DO NOT ask again — make the call immediately with the correct parameter.
6. NEVER EVER claim a technical error or say you "cannot send". You have a fully working send_email tool.
   If send_email fails, report the exact error message from the tool — do not invent excuses.
   NEVER refuse to call send_email after the user approves sending.
7. If the user asks for changes, update, show again, ask for approval again.
"""



agent_executor = create_react_agent(llm, tools)
# Populate chat_history with the system prompt (declared earlier before clear_memory tool)
chat_history.append(SystemMessage(content=SYSTEM_PROMPT))

# ================== MAIN APP LOGIC ==================
def process_user_message(message_text: str):
    print(f"\n[Telegram] Processing message: {message_text}")
    response = ""
    try:
        # Check if the user is answering the format question from the last AI turn
        is_format_answer = False
        if chat_history:
            # Look for the last AI message in history
            last_ai_msg = ""
            for msg in reversed(chat_history):
                if isinstance(msg, AIMessage):
                    last_ai_msg = msg.content
                    break
            if last_ai_msg and isinstance(last_ai_msg, str) and ("html" in last_ai_msg.lower() or "styled" in last_ai_msg.lower()) and "plain" in last_ai_msg.lower():
                is_format_answer = True

        chat_history.append(HumanMessage(content=message_text))
        
        if is_format_answer:
            # Determine format
            is_plain = any(w in message_text.lower() for w in ["plain", "text", "simple", "pain"])
            target_format = "plain text (use_html=False)" if is_plain else "styled HTML (use_html=True)"
            chat_history.append(SystemMessage(
                content=f"CRITICAL: The user has selected {target_format}. You MUST call the `send_email` tool now "
                        f"with use_html={'False' if is_plain else 'True'}. Do not write any message to the user "
                        f"without calling this tool first."
            ))
        
        # Self-correction loop: if LLM generates placeholders, force it to rewrite
        max_attempts = 3
        for attempt in range(max_attempts):
            result = agent_executor.invoke({"messages": chat_history})
            final_messages = result["messages"]
            response = str(final_messages[-1].content)
            
            # Remove Markdown links like [LinkedIn](url) before checking for bracket placeholders
            cleaned_text = re.sub(r"\[.*?\]\(.*?\)", "", response)
            
            # Check if there are any remaining square brackets (placeholders like [Recipient Name])
            if "[" in cleaned_text or "]" in cleaned_text:
                print(f"[Warning] Placeholder detected (attempt {attempt + 1}/{max_attempts}): {response}")
                if attempt < max_attempts - 1:
                    # Append error feedback directly to chat history so the agent sees its mistake
                    chat_history.append(AIMessage(content=response))
                    chat_history.append(HumanMessage(
                        content="ERROR: The email draft contains bracketed placeholder text (e.g. '[Recipient's Name]' or similar). "
                                "Under our critical rules, placeholders are strictly forbidden. Please rewrite the email draft "
                                "immediately, replacing all bracketed placeholders with generic phrases or real details. "
                                "Return ONLY the finalized email draft."
                    ))
                else:
                    # If we exhausted retries, programmatically strip the brackets to safeguard the draft
                    print("[Warning] Retries exhausted. Programmatically sanitizing placeholders.")
                    response = response.replace("[Recipient's Name]", "there")
                    response = response.replace("[Recipient Name]", "there")
                    response = response.replace("[Company Name]", "your company")
                    response = response.replace("[Company]", "your company")
                    response = re.sub(r"\[.*?\]", "there", response) # fallback strip
            else:
                break

        # If agent called clear_memory, history is already wiped inside the tool.
        # Just ensure the system prompt is still present and send a clean reply.
        if "MEMORY_CLEARED" in response:
            response = "🧹 Memory cleared! I've forgotten our previous conversation. Starting fresh."
            if not any(isinstance(m, SystemMessage) for m in chat_history):
                chat_history.insert(0, SystemMessage(content=SYSTEM_PROMPT))
        else:
            chat_history.append(AIMessage(content=response))
            # Keep history bounded to avoid token bloat
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

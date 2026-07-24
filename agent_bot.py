
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
import html
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

# ================== HTML EMAIL TEMPLATES ==================
# Defined at module level (column 0) to avoid f-string indentation SyntaxErrors.
# {{BODY}} is replaced at send time via str.replace().
NORMAL_EMAIL_TEMPLATE = (
    "<!DOCTYPE html>"
    "<html lang='en'>"
    "<head>"
    "<meta charset='UTF-8'>"
    "</head>"
    "<body style='font-family: Arial, sans-serif; font-size: 14px; color: #222222; line-height: 1.5; margin: 20px;'>"
    "{{BODY}}"
    "</body>"
    "</html>"
)

MODERN_EMAIL_TEMPLATE = (
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
    "<table width='100%' cellpadding='0' cellspacing='0' "
    "style='width:100%;max-width:1000px;background:#ffffff;"
    "border-radius:10px;overflow:hidden;"
    "box-shadow:0 2px 12px rgba(0,0,0,0.08);'>"
    "<tr><td style='background:linear-gradient(135deg,#1a1a2e 0%,#16213e 60%,#0f3460 100%);"
    "padding:28px 36px;'>"
    "<p style='margin:0;color:#ffffff;font-size:18px;font-weight:700;"
    "letter-spacing:0.5px;'>{{SUBJECT}}</p>"
    "</td></tr>"
    "<tr><td style='padding:24px;'>"
    "<div style='color:#1e293b;font-size:15px;line-height:1.8;'>{{BODY}}</div>"
    "</td></tr>"
    "</table></td></tr></table></body></html>"
)

# ================== EMAIL FORMATTING ENGINE ==================

def apply_inline_styles(text: str) -> str:
    # Escape plain-text HTML characters first
    processed = html.escape(text, quote=False)

    processed = re.sub(
        r"\*\*(.*?)\*\*",
        r"<strong>\1</strong>",
        processed
    )

    # Temporarily protect Markdown links
    protected_links = []

    def protect_markdown_link(match):
        link_html = (
            f'<a href="{match.group(2)}" '
            f'style="color:#0f3460;text-decoration:underline;">'
            f'{match.group(1)}</a>'
        )
        placeholder = f"__PROTECTED_LINK_{len(protected_links)}__"
        protected_links.append(link_html)
        return placeholder

    processed = re.sub(
        r"\[([^\]]+)\]\((https?://[^\s)]+)\)",
        protect_markdown_link,
        processed
    )

    # Convert remaining plain URLs only
    processed = re.sub(
        r'(?<!["\'])\bhttps?://[^\s<>]+',
        lambda match: (
            f'<a href="{match.group(0)}" '
            f'style="color:#0f3460;text-decoration:underline;">'
            f'{match.group(0)}</a>'
        ),
        processed
    )

    # Restore protected Markdown links
    for index, link_html in enumerate(protected_links):
        processed = processed.replace(
            f"__PROTECTED_LINK_{index}__",
            link_html
        )

    return processed

def is_list_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith(('-', '*', '•')):
        return True
    if re.match(r'^\d+[.)](?:\s+|$)', stripped):
        return True
    return False

def is_greeting_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if re.match(r'^(hi|hello|dear|hey)\b', stripped, re.IGNORECASE):
        if len(stripped) < 40 or stripped[-1] in [',', ':', '!']:
            return True
    return False

def find_signature_start_index(lines: list[str]) -> int:
    signoff_pattern = re.compile(
        r"^(best|best regards|warm regards|kind regards|regards|"
        r"sincerely|thanks|thank you|cheers|warmly|respectfully|"
        r"with appreciation)[,.]?$",
        re.IGNORECASE
    )

    # Prefer an explicit sign-off
    for i, line in enumerate(lines):
        if signoff_pattern.match(line.strip()):
            return i

    # Otherwise detect a compact contact block near the bottom
    contact_pattern = re.compile(
        r"(^subham(?: sharma)?$)|"
        r"([\w.\-+]+@[\w.\-]+\.\w+)|"
        r"(https?://|www\.|linkedin\.com|github\.com)|"
        r"(^\+?[\d\s\-()]{7,20}$)",
        re.IGNORECASE
    )

    first_contact_index = len(lines)
    found_contact = False

    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()

        if not stripped:
            if found_contact:
                continue
            continue

        if contact_pattern.search(stripped):
            first_contact_index = i
            found_contact = True
            continue

        if found_contact:
            break

    return first_contact_index

def format_html_list(lines: list[str]) -> str:
    if not lines:
        return ""
    # Decide list type based on first item
    first_stripped = lines[0].strip()
    is_ol = bool(re.match(r'^\d+[.)]', first_stripped))
    list_tag = "ol" if is_ol else "ul"
    
    html_items = []
    for line in lines:
        stripped = line.strip()
        # Strip the bullet/numbered marker
        if is_ol:
            content = re.sub(r'^\d+[.)]\s*', '', stripped)
        else:
            content = re.sub(r'^[-*•]\s*', '', stripped)
            
        content = apply_inline_styles(content)
        html_items.append(f"<li style='margin-bottom: 8px;'>{content}</li>")
        
    return f"<{list_tag} style='margin-top: 8px; margin-bottom: 16px; padding-left: 20px; color: #334155;'>{''.join(html_items)}</{list_tag}>"

def parse_and_format_email(body: str, subject: str, template_style: str = "normal") -> tuple[str, str]:
    template = MODERN_EMAIL_TEMPLATE if template_style == "modern" else NORMAL_EMAIL_TEMPLATE

    # Safety fallback if HTML is accidentally passed
    if re.search(r"<(p|br|div|span|a|ul|ol|li)\b", body, re.IGNORECASE):
        plain_body = re.sub(
            r"<br\s*/?>",
            "\n",
            body,
            flags=re.IGNORECASE
        )
        plain_body = re.sub(
            r"</(?:p|div|li)>",
            "\n\n",
            plain_body,
            flags=re.IGNORECASE
        )
        plain_body = re.sub(r"<[^>]+>", "", plain_body)
        plain_body = html.unescape(plain_body)
        plain_body = re.sub(r"\n{3,}", "\n\n", plain_body).strip()

        html_body = (
            template
            .replace("{{SUBJECT}}", html.escape(subject))
            .replace("{{BODY}}", body)
        )

        return plain_body, html_body

    # Split raw body into lines
    lines = body.splitlines()

    # Find signature block
    sig_start_idx = find_signature_start_index(lines)

    # Group lines into blocks
    blocks = []
    current_block = None

    for i, line in enumerate(lines):
        stripped = line.strip()

        if not stripped:
            line_type = "blank"
        elif i >= sig_start_idx:
            line_type = "signature"
        elif is_list_line(line):
            line_type = "list"
        elif is_greeting_line(line):
            line_type = "greeting"
        else:
            line_type = "normal"

        if current_block is None:
            current_block = {
                "type": line_type,
                "lines": []
            }

            if line_type != "blank":
                current_block["lines"].append(stripped)

        elif current_block["type"] == line_type:
            if line_type != "blank":
                current_block["lines"].append(stripped)

        else:
            blocks.append(current_block)

            current_block = {
                "type": line_type,
                "lines": []
            }

            if line_type != "blank":
                current_block["lines"].append(stripped)

    if current_block is not None:
        blocks.append(current_block)

    # Format plain-text body
    plain_parts = []

    for block in blocks:
        block_type = block["type"]
        block_lines = block["lines"]

        if block_type == "blank":
            continue

        if block_type == "normal":
            plain_parts.append(" ".join(block_lines))
        else:
            plain_parts.append("\n".join(block_lines))

    plain_body = "\n\n".join(plain_parts).strip()

    # Format HTML body
    html_blocks = []

    for block in blocks:
        block_type = block["type"]
        block_lines = block["lines"]

        if block_type == "blank":
            continue

        if block_type == "list":
            html_blocks.append(format_html_list(block_lines))
            continue

        if block_type == "normal":
            content = apply_inline_styles(
                " ".join(block_lines)
            )

        elif block_type == "signature":
            content = "<br>".join(
                apply_inline_styles(line)
                for line in block_lines
            )

        else:  # greeting
            content = apply_inline_styles(
                " ".join(block_lines)
            )

        html_blocks.append(
            "<p style='margin-top:0;"
            "margin-bottom:16px;"
            "line-height:1.8;'>"
            f"{content}"
            "</p>"
        )

    html_paragraphs = "".join(html_blocks)

    html_body = (
        template
        .replace("{{SUBJECT}}", html.escape(subject))
        .replace("{{BODY}}", html_paragraphs)
    )

    return plain_body, html_body
    
# ================== EMAIL & AI TOOLS ==================

def _send_email_core(to: str, subject: str, body: str, cc: Optional[str] = None, bcc: Optional[str] = None, attachment_paths: Optional[List[str]] = None, template_style: str = "normal") -> str:
    SENDER_EMAIL = os.getenv("SENDER_EMAIL")
    if not SENDER_EMAIL:
        return "Error: SENDER_EMAIL environment variable must be set."

    # --- Load credentials from Render Secret File or local token.json ---
    TOKEN_PATH = os.environ.get(
        "GMAIL_TOKEN_PATH",
        "/etc/secrets/token.json" if os.path.exists("/etc/secrets/token.json")
        else os.path.join(os.path.dirname(os.path.abspath(__file__)), "token.json")
    )

    SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

    try:
        creds = None
        if os.path.exists(TOKEN_PATH):
            creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                with open(TOKEN_PATH, "w") as f:
                    f.write(creds.to_json())
            else:
                return ("Error: Gmail token missing or expired. "
                        "Run generate_token.py locally to create token.json, "
                        "then upload it as a Secret File on Render.")
    except Exception as e:
        return f"Error loading Gmail credentials: {str(e)}"

    # Normalize accidental HTML from the LLM before formatting
    if re.search(r"<[^>]+>", body):
        body = re.sub(r"<br\s*/?>", "\n", body, flags=re.IGNORECASE)
        body = re.sub(r"</(?:p|div|li)>", "\n\n", body, flags=re.IGNORECASE)
        body = re.sub(r"<[^>]+>", "", body)
        body = html.unescape(body).strip()
    
    body, html_body = parse_and_format_email(body, subject, template_style)
    


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

    # Both styles are packaged as MIME multipart alternative (HTML + Plain text fallback)
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


class SendNormalEmailInput(BaseModel):
    to: str = Field(description="The email address of the primary recipient")
    subject: str = Field(description="The subject of the email")
    body: str = Field(description="The body content of the email")
    cc: Optional[str] = Field(None, description="Comma-separated email addresses for CC")
    bcc: Optional[str] = Field(None, description="Comma-separated email addresses for BCC")
    attachment_paths: Optional[List[str]] = Field(None, description="List of file paths for attachments")

@tool("send_normal_email", args_schema=SendNormalEmailInput)
def send_normal_email(to: str, subject: str, body: str, cc: Optional[str] = None, bcc: Optional[str] = None, attachment_paths: Optional[List[str]] = None) -> str:
    """Sends a clean, personal-style email (normal/plain text layout, looks like a normal typed letter, wraps dynamically to fit screen width)."""
    return _send_email_core(to, subject, body, cc, bcc, attachment_paths, template_style="normal")


class SendModernEmailInput(BaseModel):
    to: str = Field(description="The email address of the primary recipient")
    subject: str = Field(description="The subject of the email")
    body: str = Field(description="The body content of the email")
    cc: Optional[str] = Field(None, description="Comma-separated email addresses for CC")
    bcc: Optional[str] = Field(None, description="Comma-separated email addresses for BCC")
    attachment_paths: Optional[List[str]] = Field(None, description="List of file paths for attachments")

@tool("send_modern_email", args_schema=SendModernEmailInput)
def send_modern_email(to: str, subject: str, body: str, cc: Optional[str] = None, bcc: Optional[str] = None, attachment_paths: Optional[List[str]] = None) -> str:
    """Sends a styled, modern-style email (fancy newsletter/marketing card layout with background color, rounded corners, drop shadow, and gradient header banner)."""
    return _send_email_core(to, subject, body, cc, bcc, attachment_paths, template_style="modern")



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
tools = [send_normal_email, send_modern_email, draft_email_from_images, clear_memory, web_search]

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
- Write each paragraph and list item as a single continuous line. NEVER manually wrap lines or insert newlines at a fixed character limit (like 70 or 80 characters). Normal sentences within the same paragraph must be written consecutively without line breaks. Only use single newlines for list items and the signature/contact block.
- Keep the entire email under 150 words. Be concise and high-impact.
- Close with a low-friction, natural question (e.g. "Worth a quick chat?", "Are you open to this?", "Let me know if you'd like to see a demo").

IMPORTANT:
When calling the send_normal_email or send_modern_email tools, the body argument MUST be plain text only.

Never include HTML tags such as:
<p>
<br>
<div>
<span>
<a>
<ul>
<li>

Never include HTML inside the tool body argument. The Python formatter generates the HTML automatically behind the scenes.

Use only plain text with blank lines between paragraphs.

Example:

Hi there,

I wanted to...

Best,
Subham Sharma
+917988944185
subham1401sh@gmail.com
https://linkedin.com/in/subham1401

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
5. Once approved, ask: "Modern (fancy styled layout) or Normal (clean simple text layout)?"
   - If the user specifies "normal", "plain", "text", "simple", or "traditional" → call send_normal_email.
   - If the user specifies "modern", "html", "styled", "fancy", or "rich" → call send_modern_email.
   - If the user is unsure or says "both" or "either" → call send_normal_email.
   - DO NOT ask again — make the call immediately.
6. NEVER EVER claim a technical error or say you "cannot send". You have fully working send_normal_email and send_modern_email tools.
   If a tool fails, report the exact error message from the tool — do not invent excuses.
   NEVER refuse to call the sending tools after the user approves sending.
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
            if last_ai_msg and isinstance(last_ai_msg, str) and ("modern" in last_ai_msg.lower() or "normal" in last_ai_msg.lower() or "fancy" in last_ai_msg.lower() or "plain" in last_ai_msg.lower() or "styled" in last_ai_msg.lower()):
                is_format_answer = True

        chat_history.append(HumanMessage(content=message_text))
        
        if is_format_answer:
            # Determine format
            is_modern = any(w in message_text.lower() for w in ["modern", "styled", "fancy", "rich"])
            target_tool = "send_modern_email" if is_modern else "send_normal_email"
            chat_history.append(SystemMessage(
                content=f"CRITICAL: The user has selected the {'modern' if is_modern else 'normal'} format. You MUST call the `{target_tool}` tool now. "
                        f"Do not write any message to the user without calling this tool first."
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

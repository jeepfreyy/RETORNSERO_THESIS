"""
mailer.py — Sends password-reset OTP emails via Gmail SMTP (App Password).

Usage:
    from mailer import send_reset_code
    send_reset_code(recipient="user@example.com", code="482917")

All credentials are read from environment variables (see .env.example).
"""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def _build_html_body(code: str) -> str:
    """Render the emerald-on-dark HTML email matching the Barangay Sentinel palette."""
    return f"""\
<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#0f172a;font-family:'Segoe UI',Roboto,sans-serif;">
  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background:#0f172a;">
    <tr><td align="center" style="padding:40px 20px;">
      <table width="480" cellspacing="0" cellpadding="0"
             style="background:linear-gradient(135deg,#0f172a,#1e293b);
                    border:1px solid rgba(52,211,153,0.3);border-radius:16px;
                    box-shadow:0 0 40px rgba(16,185,129,0.1);">
        <!-- Header -->
        <tr><td style="padding:32px 32px 0;text-align:center;">
          <p style="margin:0;font-size:14px;letter-spacing:4px;color:#34d399;
                    font-weight:700;">BARANGAY SENTINEL</p>
          <p style="margin:8px 0 0;font-size:12px;color:#64748b;
                    letter-spacing:2px;">ACCESS RECOVERY SYSTEM</p>
        </td></tr>
        <!-- Divider -->
        <tr><td style="padding:20px 32px;">
          <hr style="border:none;height:1px;background:rgba(52,211,153,0.2);">
        </td></tr>
        <!-- Body -->
        <tr><td style="padding:0 32px;">
          <p style="margin:0 0 8px;font-size:15px;color:#cbd5e1;">
            Your verification code is:
          </p>
          <div style="background:rgba(16,185,129,0.08);border:1px solid rgba(52,211,153,0.3);
                      border-radius:12px;padding:20px;text-align:center;margin:16px 0;">
            <span style="font-size:36px;font-weight:700;letter-spacing:12px;color:#34d399;
                         font-family:'Courier New',monospace;">{code}</span>
          </div>
          <p style="margin:16px 0 0;font-size:13px;color:#94a3b8;">
            Enter this code in the Barangay Sentinel system to reset your access credentials.
            This code expires in <strong style="color:#f59e0b;">10 minutes</strong>.
          </p>
          <p style="margin:12px 0 0;font-size:13px;color:#94a3b8;">
            If you did not request this, you can safely ignore this email.
          </p>
        </td></tr>
        <!-- Footer -->
        <tr><td style="padding:24px 32px;">
          <hr style="border:none;height:1px;background:rgba(52,211,153,0.2);">
          <p style="margin:12px 0 0;font-size:11px;color:#475569;text-align:center;
                    letter-spacing:1px;">
            &#128274; 256-BIT ENCRYPTED &bull; BARANGAY SENTINEL SECURITY SYSTEM
          </p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""


def _build_plain_body(code: str) -> str:
    """Fallback plain-text version."""
    return (
        f"BARANGAY SENTINEL — ACCESS RECOVERY\n"
        f"------------------------------------\n\n"
        f"Your verification code is: {code}\n\n"
        f"Enter this code in the Barangay Sentinel system to reset your access credentials.\n"
        f"This code expires in 10 minutes.\n\n"
        f"If you did not request this, you can safely ignore this email.\n"
    )


def send_reset_code(recipient: str, code: str) -> None:
    """
    Send a password-reset OTP to *recipient* via SMTP.

    Raises smtplib.SMTPException (or subclass) on failure so callers can
    decide how to handle it.  **Never** log the code or email body.
    """
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "587"))
    username = os.environ["SMTP_USERNAME"]
    password = os.environ["SMTP_PASSWORD"]
    from_name = os.environ.get("MAIL_FROM_NAME", "Barangay Sentinel")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Barangay Sentinel \u2014 Your access recovery code"
    msg["From"] = f"{from_name} <{username}>"
    msg["To"] = recipient

    msg.attach(MIMEText(_build_plain_body(code), "plain", "utf-8"))
    msg.attach(MIMEText(_build_html_body(code), "html", "utf-8"))

    with smtplib.SMTP(host, port) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(username, password)
        server.sendmail(username, [recipient], msg.as_string())

"""
Task Registry — Code Review Environment
3 tasks: easy → medium → hard
Each task has a real code diff with seeded bugs/security issues.
"""

TASK_REGISTRY = {

    # -----------------------------------------------------------------------
    # EASY: Null pointer + obvious off-by-one in a simple Python function
    # -----------------------------------------------------------------------
    "easy_null_pointer": {
        "filename": "user_service.py",
        "language": "python",
        "context": (
            "PR #42: Add get_user_age() helper to the UserService class. "
            "Reviewer should check for correctness and basic safety."
        ),
        "diff": '''\
--- a/user_service.py
+++ b/user_service.py
@@ -1,15 +1,40 @@
 from datetime import date
+import logging
 
 class UserService:
     def __init__(self, db):
         self.db = db
+        self.logger = logging.getLogger(__name__)
 
     def get_user(self, user_id: int):
         return self.db.query("SELECT * FROM users WHERE id = ?", user_id)
 
+    def get_user_age(self, user_id: int) -> int:
+        """Return user age in years."""
+        user = self.get_user(user_id)
+        birth_date = user["birth_date"]                  # line 14
+        today = date.today()
+        age = today.year - birth_date.year               # line 16
+        if today.month < birth_date.month:
+            age -= 1
+        return age
+
+    def get_all_usernames(self) -> list:
+        """Return list of all usernames."""
+        results = self.db.query("SELECT username FROM users")
+        usernames = []
+        for i in range(len(results)):                    # line 26
+            usernames.append(results[i]["username"])
+        return usernames
+
+    def delete_user(self, user_id: int, admin_token: str):
+        """Delete a user — admin only."""
+        if admin_token == "secret123":                   # line 32
+            self.db.execute("DELETE FROM users WHERE id = ?", user_id)
+            self.logger.info(f"Deleted user {user_id}")
+        else:
+            raise PermissionError("Invalid admin token")
''',
        "seeded_issues": [
            {
                "id": 0,
                "type": "bug",
                "severity": "high",
                "line": 14,
                "description": "get_user() can return None if user_id doesn't exist, causing AttributeError on user['birth_date']",
                "keywords": {"none", "null", "nonetype", "attributeerror", "user", "exist", "check", "14", "birth_date"},
            },
            {
                "id": 1,
                "type": "security",
                "severity": "critical",
                "line": 32,
                "description": "Hardcoded admin token 'secret123' is a critical security vulnerability",
                "keywords": {"hardcoded", "secret", "token", "admin", "password", "credential", "plaintext", "32"},
            },
        ],
    },

    # -----------------------------------------------------------------------
    # MEDIUM: Flask route with SQL injection + insecure token + missing auth
    # -----------------------------------------------------------------------
    "medium_flask_security": {
        "filename": "auth_routes.py",
        "language": "python",
        "context": (
            "PR #87: New Flask authentication routes for a SaaS app. "
            "Security reviewer should check for OWASP Top 10 violations."
        ),
        "diff": '''\
--- a/auth_routes.py
+++ b/auth_routes.py
@@ -0,0 +1,68 @@
+from flask import Flask, request, jsonify, session
+import sqlite3
+import hashlib
+import jwt
+import os
+
+app = Flask(__name__)
+app.secret_key = "dev-secret-key-do-not-use"            # line 8
+
+DB_PATH = "users.db"
+
+def get_db():
+    return sqlite3.connect(DB_PATH)
+
+@app.route("/login", methods=["POST"])
+def login():
+    username = request.form.get("username")
+    password = request.form.get("password")
+
+    # Hash password and look up user
+    pw_hash = hashlib.md5(password.encode()).hexdigest()  # line 21
+
+    db = get_db()
+    cursor = db.cursor()
+    query = f"SELECT * FROM users WHERE username=\'{username}\' AND password_hash=\'{pw_hash}\'"  # line 24
+    cursor.execute(query)
+    user = cursor.fetchone()
+
+    if user:
+        token = jwt.encode(
+            {"user_id": user[0], "username": username},
+            app.secret_key,
+            algorithm="HS256"
+        )
+        session["user"] = username
+        return jsonify({"token": token})
+    return jsonify({"error": "Invalid credentials"}), 401
+
+@app.route("/admin/users", methods=["GET"])
+def list_users():                                         # line 38
+    db = get_db()
+    cursor = db.cursor()
+    cursor.execute("SELECT id, username, email FROM users")
+    users = cursor.fetchall()
+    return jsonify(users)
+
+@app.route("/profile", methods=["POST"])
+def update_profile():
+    user_id = request.form.get("user_id")
+    new_bio = request.form.get("bio")
+
+    db = get_db()
+    cursor = db.cursor()
+    query = f"UPDATE users SET bio=\'{new_bio}\' WHERE id={user_id}"  # line 51
+    cursor.execute(query)
+    db.commit()
+    return jsonify({"success": True})
+
+@app.route("/reset_password", methods=["POST"])
+def reset_password():
+    email = request.form.get("email")
+    token = os.urandom(4).hex()                          # line 59
+    # TODO: send token via email
+    return jsonify({"reset_token": token})               # line 61
''',
        "seeded_issues": [
            {
                "id": 0,
                "type": "security",
                "severity": "critical",
                "line": 24,
                "description": "SQL injection in login query — username and pw_hash concatenated directly into SQL string",
                "keywords": {"sql", "injection", "f-string", "concatenat", "parameteriz", "query", "login", "24", "username"},
            },
            {
                "id": 1,
                "type": "security",
                "severity": "critical",
                "line": 21,
                "description": "MD5 used for password hashing — MD5 is cryptographically broken, use bcrypt/argon2",
                "keywords": {"md5", "hash", "password", "bcrypt", "argon", "weak", "broken", "crypto", "21"},
            },
            {
                "id": 2,
                "type": "security",
                "severity": "high",
                "line": 38,
                "description": "Admin endpoint /admin/users has no authentication or authorization check",
                "keywords": {"auth", "authentication", "authorization", "admin", "unprotected", "missing", "38", "list_users"},
            },
            {
                "id": 3,
                "type": "security",
                "severity": "high",
                "line": 59,
                "description": "Password reset token is only 4 bytes (32-bit entropy) — trivially brute-forceable; use os.urandom(32)",
                "keywords": {"token", "entropy", "random", "4", "weak", "brute", "reset", "59", "urandom"},
            },
            {
                "id": 4,
                "type": "security",
                "severity": "critical",
                "line": 61,
                "description": "Reset token returned in HTTP response body — it should only be sent via email, not exposed in API response",
                "keywords": {"token", "response", "expose", "leak", "email", "body", "61", "return"},
            },
        ],
    },

    # -----------------------------------------------------------------------
    # HARD: Race condition in async Python + subtle logic bug + IDOR
    # -----------------------------------------------------------------------
    "hard_async_race": {
        "filename": "payment_processor.py",
        "language": "python",
        "context": (
            "PR #201: Async payment processing service. "
            "High-stakes financial code — reviewer must catch concurrency bugs, "
            "business logic errors, and authorization flaws. "
            "This code runs in production handling real money."
        ),
        "diff": '''\
--- a/payment_processor.py
+++ b/payment_processor.py
@@ -0,0 +1,95 @@
+import asyncio
+import logging
+from decimal import Decimal
+from typing import Optional
+
+logger = logging.getLogger(__name__)
+
+class PaymentProcessor:
+    def __init__(self, db, notifier):
+        self.db = db
+        self.notifier = notifier
+        self._processing = set()                         # line 12
+
+    async def process_payment(self, payment_id: int, user_id: int) -> dict:
+        """Process a pending payment."""
+        if payment_id in self._processing:
+            return {"error": "Already processing"}
+
+        self._processing.add(payment_id)                 # line 19
+        try:
+            payment = await self.db.get_payment(payment_id)
+
+            if payment is None:
+                return {"error": "Payment not found"}
+
+            # Verify payment belongs to user                line 26
+            if payment["user_id"] != user_id:
+                logger.warning(f"IDOR attempt: user {user_id} on payment {payment_id}")
+                return {"error": "Unauthorized"}
+
+            if payment["status"] != "pending":
+                return {"error": f"Payment already {payment['status']}"}
+
+            # Deduct from wallet
+            wallet = await self.db.get_wallet(payment["user_id"])
+            balance = Decimal(str(wallet["balance"]))
+            amount = Decimal(str(payment["amount"]))
+
+            if balance < amount:                          # line 39
+                return {"error": "Insufficient funds"}
+
+            # Update balance and mark paid                  line 42
+            new_balance = balance - amount
+            await self.db.update_wallet(payment["user_id"], float(new_balance))
+            await self.db.update_payment_status(payment_id, "completed")
+
+            await self.notifier.send(payment["user_id"], f"Payment {payment_id} completed")
+            return {"success": True, "balance": float(new_balance)}
+
+        finally:
+            self._processing.discard(payment_id)         # line 51
+
+    async def batch_refund(self, payment_ids: list, admin_user_id: int) -> dict:
+        """Refund multiple payments — admin only."""
+        admin = await self.db.get_user(admin_user_id)
+        if not admin.get("is_admin"):
+            return {"error": "Unauthorized"}
+
+        results = {}
+        total_refunded = Decimal("0")
+
+        for payment_id in payment_ids:
+            payment = await self.db.get_payment(payment_id)
+            if payment and payment["status"] == "completed":
+                amount = Decimal(str(payment["amount"]))
+                wallet = await self.db.get_wallet(payment["user_id"])
+                new_balance = Decimal(str(wallet["balance"])) + amount
+                await self.db.update_wallet(payment["user_id"], float(new_balance))
+                await self.db.update_payment_status(payment_id, "refunded")
+                total_refunded += amount
+                results[payment_id] = "refunded"
+
+        return {"refunded": len(results), "total_amount": float(total_refunded)}
+
+    async def get_payment_details(self, payment_id: int, requesting_user_id: int) -> Optional[dict]:
+        """Get full payment details including card info."""
+        payment = await self.db.get_payment(payment_id)  # line 79
+        return payment                                    # line 80
+
+    def calculate_fee(self, amount: float, is_international: bool) -> float:
+        """Calculate processing fee."""
+        if is_international:
+            fee = amount * 0.029 + 0.30
+        else:
+            fee = amount * 0.015 + 0.25
+        return round(fee, 2)
+
+    async def schedule_retry(self, payment_id: int, delay_seconds: int = 60):
+        """Retry a failed payment after delay."""
+        await asyncio.sleep(delay_seconds)
+        payment = await self.db.get_payment(payment_id)
+        if payment and payment["status"] == "failed":
+            await self.process_payment(payment_id, payment["user_id"])
''',
        "seeded_issues": [
            {
                "id": 0,
                "type": "bug",
                "severity": "critical",
                "line": 12,
                "description": "Race condition: _processing is a plain set shared across coroutines — two concurrent calls can both pass the check before either adds to the set. Use asyncio.Lock.",
                "keywords": {"race", "condition", "concurrent", "lock", "asyncio", "set", "atomic", "12", "processing"},
            },
            {
                "id": 1,
                "type": "bug",
                "severity": "critical",
                "line": 39,
                "description": "TOCTOU bug: balance is checked then updated in two separate DB calls with no transaction/lock — double-spend is possible under concurrency",
                "keywords": {"toctou", "transaction", "atomic", "double", "spend", "balance", "39", "check", "concurren"},
            },
            {
                "id": 2,
                "type": "security",
                "severity": "critical",
                "line": 79,
                "description": "IDOR in get_payment_details: requesting_user_id is accepted but never checked against payment owner — any user can fetch any payment's card info",
                "keywords": {"idor", "authorization", "auth", "check", "79", "80", "requesting", "owner", "user_id"},
            },
            {
                "id": 3,
                "type": "bug",
                "severity": "high",
                "line": 44,
                "description": "Decimal converted to float before DB write loses precision for financial amounts — store as Decimal or string",
                "keywords": {"float", "decimal", "precision", "financial", "money", "44", "convert", "loss"},
            },
        ],
    },
}

# 🔐 Security Policy

## 📅 Supported Versions

We currently support the latest version of the project.  
Security updates and fixes will only be provided for the latest code in the `main` branch.

| Version | Supported          |
|---------|--------------------|
| Latest  | ✅ Yes              |
| Older   | ❌ No               |

---

## 📣 Reporting a Vulnerability

If you discover a security vulnerability in this project, **please report it privately** to the maintainers instead of creating a public issue. This helps prevent abuse and gives us time to issue a fix.

### 📧 Contact

Please send an email to:

> **uditmerit@gmail.com**

Include the following in your report:
- Clear description of the vulnerability
- Steps to reproduce
- Affected files or code blocks
- Optional: Suggest a fix or patch

---

## 🔐 Security Best Practices Followed

This project follows these basic security principles:

- 🔒 **No hardcoded credentials** (e.g., API keys, passwords)
- 🚫 **Input validation** to prevent injection attacks
- 🧪 **Model prediction thresholds** are used to avoid misleading outcomes
- 🧹 Regular **dependency review** and `requirements.txt` version pinning
- 📁 Use of **`.gitignore`** to exclude sensitive directories (e.g., `models/` with large or private data)
- 🔍 Model and prediction logs are not exposed to the public

---

## 🚫 Known Security Issues

There are currently **no known security vulnerabilities** in this project.

---

## 🤝 Responsible Disclosure

We appreciate ethical and responsible disclosure of security vulnerabilities.  
Please allow us up to **72 hours** to respond to your report and provide a fix.

---

## 🔐 Third-Party Libraries Used

This project uses the following third-party libraries:
- Flask (web framework)
- scikit-learn (ML model)
- pandas (data processing)
- joblib (model serialization)

Please ensure you also review their individual security policies if you suspect issues from dependencies.

---

Thank you for helping us keep **Credit Card Approval Prediction** safe and secure! 🛡️

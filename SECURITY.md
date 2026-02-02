# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------- |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

The langchain-sarvam-integration team takes security bugs seriously. We appreciate your efforts to responsibly disclose your findings.

If you discover a security vulnerability, please send an email to [INSERT SECURITY EMAIL]. You will receive a response from the team within 48 hours. If the issue is confirmed, we will release a patch as soon as possible depending on complexity.

Please include the following information in your report:

- **Description** of the vulnerability
- **Steps to reproduce** the issue
- **Potential impact** of the vulnerability
- **Suggested fix** (if available)

### What to Expect

When you report a vulnerability:

1. **Acknowledgment**: We will send a confirmation that we received your report
2. **Assessment**: We will investigate and validate the vulnerability
3. **Resolution**: We will develop and test a fix
4. **Disclosure**: We will coordinate public disclosure with you

### Security Best Practices for Users

#### API Key Management

This package requires a Sarvam AI API key. Follow these best practices:

**✅ DO:**
- Store API keys in environment variables: `export SARVAM_API_KEY=your_key`
- Use `.env` files (add to `.gitignore`)
- Rotate keys periodically
- Use different keys for development and production

**❌ DON'T:**
- Commit API keys to version control
- Share keys in public code
- Log keys in error messages
- Embed keys in client-side code

#### Example Secure Configuration

```python
# Use environment variables
import os
from sarvam import SarvamChat

# Safe: reads from environment
chat = SarvamChat()  # Uses SARVAM_API_KEY env var

# Safe: explicit key (but use env vars in production)
chat = SarvamChat(api_key=os.getenv("SARVAM_API_KEY"))
```

#### .gitignore Example

Ensure your `.gitignore` includes:
```
.env
.env.local
*.key
credentials.json
```

## Security Features

This package includes the following security features:

- **Secure API Key Storage**: Uses `pydantic.SecretStr` for API keys
- **No Key Logging**: Keys are never logged in debug output
- **Input Validation**: Validates user inputs through Pydantic models
- **NullHandler Logging**: Safe for library use (doesn't spam logs)

## Dependency Security

We regularly update dependencies to address security vulnerabilities. Please:

- Keep your installation updated: `pip install --upgrade langchain-sarvam-integration`
- Review security advisories in `pyproject.toml`
- Report any security issues with dependencies

## Additional Resources

- [Sarvam AI Security Documentation](https://sarvam.ai/security)
- [LangChain Security Best Practices](https://python.langchain.com/docs/security/)
- [Python Security Guidelines](https://docs.python.org/3/security/index.html)

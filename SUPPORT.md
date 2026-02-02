# Support

## Getting Help

If you need help with langchain-sarvam-integration, here are the best ways to get support:

### Documentation

- 📖 [README.md](README.md) - Installation, usage, and examples
- 📋 [CHANGELOG.md](CHANGELOG.md) - Version history and changes

### Common Issues

#### API Key Problems

**Problem**: `ValueError: Sarvam API key must be provided`

**Solution**: Set your API key as an environment variable:
```bash
export SARVAM_API_KEY="your-api-key-here"
# Or on Windows:
set SARVAM_API_KEY=your-api-key-here
```

Or pass it directly:
```python
from sarvam import SarvamChat
chat = SarvamChat(api_key="your-api-key")
```

#### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'sarvam'`

**Solution**: Install the package:
```bash
pip install langchain-sarvam-integration
```

For development:
```bash
pip install -e .
```

#### JSON Parsing Errors

**Problem**: `ValueError: No valid JSON found in content`

**Solution**: This happens when Sarvam AI returns incomplete or malformed JSON. Try:
- Lower reasoning effort: `SarvamChat(reasoning_effort="low")`
- Lower temperature: `SarvamChat(temperature=0.3)`
- Simplify your prompt for more direct responses

### Getting Help from the Community

#### Ask a Question

1. **Search existing issues** - Your question may already be answered
2. **Read the documentation** - Check README and examples
3. **Open a question issue** - Use the [question template](https://github.com/your-username/langchain-sarvam-integration/issues/new?template=question.md)

#### Report a Bug

1. **Check if it's already reported** - Search existing issues
2. **Gather information**:
   - Python version (`python --version`)
   - Package version (`pip show langchain-sarvam-integration`)
   - Error message and stack trace
   - Minimal code example to reproduce
3. **Open a bug report** - Use the [bug report template](https://github.com/your-username/langchain-sarvam-integration/issues/new?template=bug_report.md)

#### Request a Feature

1. **Check if it's already requested** - Search existing issues
2. **Describe your use case** - What do you want to do?
3. **Provide examples** - How would you like it to work?
4. **Open a feature request** - Use the [feature request template](https://github.com/your-username/langchain-sarvam-integration/issues/new?template=feature_request.md)

### Professional Support

For enterprise or commercial support options, please contact:
[INSERT CONTACT INFORMATION]

### Contributing

Found a bug or want to add a feature? We welcome contributions!

- 📖 Read [CONTRIBUTING.md](CONTRIBUTING.md)
- 🔧 Check out the code
- 🧪 Run the tests
- 📤 Submit a pull request

### Resources

- [Sarvam AI Documentation](https://docs.sarvam.ai/)
- [Sarvam-M Model Docs](https://docs.sarvam.ai/api-reference-docs/getting-started/models/sarvam-m)
- [LangChain Documentation](https://python.langchain.com/)
- [LangChain GitHub Discussions](https://github.com/langchain-ai/langchain/discussions)

### Security Issues

⚠️ **Do not report security issues publicly.**

Please email security vulnerabilities to: [INSERT SECURITY EMAIL]

See [SECURITY.md](SECURITY.md) for more details.

### Response Time Expectations

- **Bug reports**: We aim to respond within 7 days
- **Feature requests**: We'll evaluate and respond within 14 days
- **Questions**: Community response times vary

### Code of Conduct

Please be respectful and constructive in all interactions. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

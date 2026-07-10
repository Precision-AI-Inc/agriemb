# Security Policy

## Supported Versions

Security fixes are provided for the latest released version of `precisionai-agriemb`.

Older versions are not guaranteed to receive patches. Please upgrade to the latest release before reporting a vulnerability where practical.

## Reporting a Vulnerability

Do not report security vulnerabilities through public GitHub issues, discussions, or pull requests.

Instead, report suspected vulnerabilities privately by emailing:

**michael@precision.ai**

Please include as much of the following as possible:

- A brief description of the issue
- Steps to reproduce, including any relevant input data
- A minimal proof of concept, if available
- The affected package version (`pip show precisionai-agriemb`)
- Python version and operating system
- Any known impact or realistic attack scenario

We will acknowledge receipt within 7 business days and follow up with any questions needed to investigate.

## Scope

Issues are considered security-relevant when they allow:

- Arbitrary code execution triggered by a crafted input (dataset file, embedding payload, etc.)
- Leakage of credentials (e.g. `PAI_EMBEDDINGS_API_KEY`) through logs, errors, or generated artifacts
- Supply-chain compromise of the package or its release artifacts

The following are generally out of scope:

- Issues that require pre-existing arbitrary code execution on the host
- Vulnerabilities only affecting unsupported versions
- Problems caused by intentionally unsafe use (e.g. committing a real `.env` file)
- Reports without enough information to reproduce or assess the issue
- Vulnerabilities in the sibling `agristress-utils` or `agrieval` projects — report
  those in their own repositories
- Vulnerabilities in the hosted PAI embedding API (https://embeddings.precision.ai/)
  itself — report those to the address above, as it is a separate deployment

## Coordinated Disclosure

Please allow the maintainers reasonable time to investigate and ship a fix before any public disclosure.

If a vulnerability is confirmed, we will:

1. Assess severity and the range of affected versions
2. Prepare and test a fix
3. Release a patched version through PyPI
4. Publish an advisory or release note as appropriate

## Security Updates

Security fixes are distributed through normal package channels. Upgrade with:

```bash
pip install --upgrade precisionai-agriemb
```

# Security Policy

## Supported Versions

Security fixes are provided for the latest released version of `pai-myproject`.

Older versions are not guaranteed to receive patches. Please upgrade to the latest release before reporting a vulnerability where practical.

## Reporting a Vulnerability

Do not report security vulnerabilities through public GitHub issues, discussions, or pull requests.

Instead, report suspected vulnerabilities privately by emailing:

**michael@precision.ai**

Please include as much of the following as possible:

- A brief description of the issue
- Steps to reproduce, including any relevant request payload or input data
- A minimal proof of concept, if available
- The affected package version (`pip show pai-myproject`)
- Python version and operating system
- Any known impact or realistic attack scenario

We will acknowledge receipt within 7 business days and follow up with any questions needed to investigate.

## Scope

Issues are considered security-relevant when they allow:

- Unauthorized access to data or results served through the API
- Arbitrary code execution triggered by a crafted input payload
- Privilege escalation through the service
- Supply-chain compromise of the package or its release artifacts
- Denial of service caused by malformed or adversarially large input where the service is expected to handle untrusted data

The following are generally out of scope:

- Issues that require pre-existing arbitrary code execution on the host
- Vulnerabilities only affecting unsupported versions
- Problems caused by intentionally unsafe deployment (e.g. exposing the API to the public internet without authentication)
- Reports without enough information to reproduce or assess the issue

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
pip install --upgrade pai-myproject
```

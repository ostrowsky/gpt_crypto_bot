# Coverage authority provisioning

Status: account/key provisioning only; not portfolio certification or promotion.
Date: 2026-10-02. TH-01 through TH-12 apply.

The explicitly approved GptBotCoverageAuthority is a standard local service
principal, never an administrator. Provisioning must first create and verify a
Windows restore point. Failure to create a new point aborts before accounts,
rights, directories or keys change. Administrator elevation is an OS requirement.
An existing account or authority directory is not silently reused or overwritten.

Private material lives under ProgramData/GptBotCoverageAuthority, outside the
operator-writable checkout. Protected ACLs and administrator ownership permit only
SYSTEM/Administrators full control and the exact authority SID read access to the
private key. Evaluator/trainer have no private access. Public RSA-3072 verification
material has a separate protected directory, evaluator read access, and no role
write access. Passwords are randomly generated in memory and saved only using
administrator-user DPAPI in an administrator-only directory. Nothing secret is
printed or committed. The signing principal has batch logon and explicit deny
interactive/remote-interactive logon; no unrestricted signing scheduled task is
installed. Signing requests must later validate provenance, policy parity and
experiment approval; possession of a key is never sufficient certification.

RSA uses SHA256/PKCS1 signatures. Existing HMAC certificates remain a legacy
protocol; provisioning does not hand the private key to the evaluator, enable
legacy shared-key signing, change verification gates, issue a certificate, enable
BUY or claim a closed loop. Public-key verification and a validated signer service
are separate required integration work before authority evidence can be consumed.

Public verification is now wired into scheduled evaluator/controller and both
production portfolio CLIs using `coverage-rsa-sha256-v1`. They read only the fixed
OS-protected public trust store and pinned raw key digest, never the old coverage
HMAC environment secret. Windows RSA SHA256/PKCS1 verification uses the native
provider, accepts only public RSA-3072 parameters, and fails closed on missing
key/pin, changed payload, malformed signature, provider failure or timeout.
Explicit byte-key function callers remain legacy test/migration interfaces; the
scheduled production path cannot choose that fallback from an input certificate.
No signer task, automatically certified provenance or production promotion is
implied. Existing protected scheduled source must be separately deployed after
restore; changing checkout files does not update that frozen source.

The installer records restore sequence, exact SID, public fingerprint, grants and
state PROVISIONED_NOT_CERTIFYING. Any failure after account creation removes only
the newly created account and its newly created directory after containment
verification. Restore point recovery remains available. Re-running an installed
authority requires explicit audited migration, not implicit key rotation.

Verification: PowerShell parse; focused structural tests ensure checkpoint precedes
mutation, account collision refusal, narrow ACLs, no admin group, deny interactive,
no certificates/activation, and fail-closed rollback. Deployment additionally
requires actual OS restore point and account/ACL verification. Synthetic tests do
not prove Windows deployment. Full Harness FAIL remains a deployment blocker for
models, not a reason to suppress account/key preparation.

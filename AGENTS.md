# Project agent rules

## Complete the engineering cycle

For every code or behavior change, finish the engineering cycle in the same
task unless the user explicitly asks not to commit or not to push:

1. Record or update the behavioral specification.
2. Add or update focused automated tests.
3. Run the relevant tests and `git diff --check`.
4. Review the final diff and stage only intended source/spec/test files.
5. Commit with a descriptive message.
6. Push the commit to the configured remote.
7. Report the specification, tests, commit, and push result to the user.

Do not commit runtime state, generated reports, credentials, positions, logs, or
trained model artifacts unless the user explicitly requests those artifacts.

Trading-policy hypotheses must be validated on the maximum available historical
period before production behavior is relaxed. Always propose and perform that
validation when evaluating a new hypothesis.

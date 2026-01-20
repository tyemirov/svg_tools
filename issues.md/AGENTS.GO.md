# AGENTS.GO.md

## Scope

Backend guidance for Go code. Follow AGENTS.md for repo-wide policies, documentation rules, and workflow expectations.

## Backend (Go Language)

### Core Principles

- Reuse existing code first; extend or adapt before writing new code.
- Generalize existing implementations instead of duplicating them.
- Favor data structures (maps, registries, tables) over branching logic.
- Use composition, interfaces, and method sets (“object-oriented Go”).
- Depend on interfaces; return concrete types.
- Group behavior on receiver types with cohesive methods.
- Inject all external effects (I/O, network, time, randomness, OS).
- No hidden globals for behavior.
- Treat inputs as immutable; return new values instead of mutating.
- Separate pure logic from effectful layers.
- Keep units small and composable.
- Minimal public API surface.
- Provide only the best solution — no alternatives.

---

### Deliverables (for automation)

- Only changed files.
- No diffs, snippets, or examples.
- Must compile cleanly.
- Must pass `go fmt ./... && go vet ./... && go test ./...`.

---

### Code Style

- No single-letter identifiers.
- Long, descriptive names for all identifiers.
- No inline comments.
- Only GoDoc for modules and exported identifiers.
- No repeated inline string literals — lift to constants.
- Return `error`; wrap with `%w` or `errors.Join`.
- No panics in library code.
- Use zap for logging; no `fmt.Println`.
- Prefer channels and contexts over shared mutable state.
- Guard critical sections explicitly.

---

### Project Structure

- `cmd/` for CLI entrypoints.
- `internal/` for private packages.
- `pkg/` for reusable libraries.
- No package cycles.
- Respect existing layout and naming.

---

### Configuration & CLI

- Use Viper + Cobra.
- Flags optional when provided via config/env.
- Validate config in `PreRunE`.
- Read secrets from environment.

---

### Dependencies (Approved)

- Core: `spf13/viper`, `spf13/cobra`, `uber/zap`.
- HTTP: `gin-gonic/gin`, `gin-contrib/cors`.
- Data: `gorm.io/gorm`, `gorm.io/driver/postgres`, `jackc/pgx/v5`.
- Auth/Validation: `golang-jwt/jwt/v5`, `go-playground/validator/v10`.
- Testing: `stretchr/testify`.
- Optional: `joho/godotenv`, `prometheus/client_golang`, `robfig/cron/v3`.
- Prefer standard library whenever possible.

---

### Testing

- **Integration tests only**: Unit tests are prohibited. All tests MUST be black-box integration tests that exercise the real deployed code path.
- Tests must go through real entry points (HTTP handlers, CLI commands, public APIs).
- No testing components in isolation—if it doesn't run the actual code path, it's worthless.
- Use `t.TempDir()` for temporary dirs; no filesystem pollution.
- Table-driven tests for coverage of edge cases.
- 100% coverage driven by integration/end-to-end scenarios.

---

### Web/UI

- Use Gin for routing.
- Middleware for CORS, auth, logging.
- Vanilla CSS; no Bootstrap.
- Header fixed top; footer fixed bottom using CSS utilities.

---

### Performance & Reliability

- Measure before optimizing.
- Favor clarity first, optimize after.
- Use maps and indexes for hot paths.
- Always propagate `context.Context`.
- Backoff/retry as data-driven config.

---

### Security

- Secrets from env.
- Never log secrets or PII.
- Validate all inputs.
- Principle of least privilege.
- CSP-friendly ES modules. Allowed third-party scripts: Google Analytics snippet, Google Identity Services, Loopaware widget. When CSP is enabled, inline scripts must be limited to GA config or guarded by nonce/hash.

#### CSP Template (optional; use when enabling CSP)

- HTTP header (preferred):
  - `Content-Security-Policy: default-src 'self'; script-src 'self' https://cdn.jsdelivr.net https://accounts.google.com https://www.googletagmanager.com https://loopaware.mprlab.com 'nonce-<nonce-value>'; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; img-src 'self' data: blob:; connect-src 'self' https://llm-proxy.mprlab.com http://localhost:8080; font-src 'self' data:; frame-src https://accounts.google.com; base-uri 'self'; form-action 'self';`
- Meta tag (static hosting):
  - `<meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self' https://cdn.jsdelivr.net https://accounts.google.com https://www.googletagmanager.com https://loopaware.mprlab.com 'unsafe-inline'; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; img-src 'self' data: blob:; connect-src 'self' https://llm-proxy.mprlab.com http://localhost:8080; font-src 'self' data:; frame-src https://accounts.google.com; base-uri 'self'; form-action 'self';">`
- Replace `connect-src` endpoints when running against different backends or proxies. Prefer nonces over `'unsafe-inline'` where a server can inject them.
- When using a local LLM proxy on a non-default port (e.g., `http://localhost:8081`), include it in `connect-src`.

### Containerization

- Follow `AGENTS.DOCKER.md` for Dockerfile requirements and runtime expectations.
- Container builds must default to multi-stage Dockerfiles with root as the sole user.
- Set `CGO_ENABLED=0` (`CGOENABLED=false`) for Go binaries built in Docker. Enable CGO only when absolutely necessary and document the rationale.

### Assistant Workflow

- Read repo and scan existing code.
- Plan reuse and extension.
- Replace branching with data tables where appropriate.
- Implement minimal, cohesive types.
- Inject dependencies.
- Prove with table-driven tests.

---

### Review Checklist

- [ ] Reused/extended existing code.
- [ ] Replaced branching with data structures where appropriate.
- [ ] Minimal, cohesive public API.
- [ ] All side effects injected.
- [ ] No single-letter identifiers.
- [ ] Constants used for repeated strings.
- [ ] zap logging; contextual errors.
- [ ] Config via Viper; validated in `PreRunE`.
- [ ] Table-driven tests; no filesystem pollution.
- [ ] `go fmt`, `go vet`, `go test ./...` pass.

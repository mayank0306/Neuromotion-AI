# Architecture overview

NeuroMotion is currently a modular FastAPI application with a Next.js web client. PostgreSQL is the system of record, Redis is an optional cache, and MediaPipe provides the initial pose-estimation implementation.

```text
apps/web  ->  /api/v1 FastAPI backend  ->  PostgreSQL
                              |               Redis (optional)
                              v
                       MediaPipe pose estimator
```

The API is organized by domain (`auth`, `users`, `movement`, `analysis`, and `insights`). New capabilities should be added as domain modules rather than as cross-cutting route handlers.

The empty `services/` directories are future extraction boundaries, not running microservices. The repository deliberately remains a monolith until deployment or team ownership requires separation.

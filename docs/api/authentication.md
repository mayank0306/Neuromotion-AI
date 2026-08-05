# Authentication API

All authenticated endpoints require `Authorization: Bearer <access_token>`.

- `POST /api/v1/auth/register` creates a user and returns a `user` and `token` object.
- `POST /api/v1/auth/login` returns an access and refresh token.
- `POST /api/v1/auth/refresh` exchanges a refresh token for a new token pair.
- `GET /api/v1/users/me` returns the authenticated user profile.

Access tokens are short-lived. Do not store refresh tokens in browser local storage in a production application; use an httpOnly secure cookie or a dedicated secure credential store.

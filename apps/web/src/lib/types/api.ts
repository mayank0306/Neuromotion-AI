export type Token = { access_token: string; refresh_token: string; token_type: string; expires_in: number };

export type LoginRequest = { email: string; password: string };

export type User = { id: string; email: string; first_name: string; last_name: string; is_active: boolean; is_email_verified: boolean; created_at: string };

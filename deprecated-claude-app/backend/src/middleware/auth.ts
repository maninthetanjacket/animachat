import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';

export interface AuthRequest extends Request {
  userId?: string;
  body: any;
  params: any;
}

// BREAKING CHANGE: JWT_SECRET is now required. The server will refuse to start
// without it. Previously it fell back to a hardcoded default that was visible in
// source code, allowing anyone to forge auth tokens. Ensure JWT_SECRET is set in
// all deployment environments (staging, production) before deploying this change.
function getJwtSecret(): string {
  const secret = process.env.JWT_SECRET;
  if (!secret) {
    throw new Error('JWT_SECRET environment variable is required. Set it to a strong random secret (32+ characters).');
  }
  return secret;
}

export function assertJwtSecretConfigured(): void {
  getJwtSecret();
}

export function authenticateToken(req: AuthRequest, res: Response, next: NextFunction) {
  const jwtSecret = getJwtSecret();
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    return res.status(401).json({ error: 'Access token required' });
  }

  jwt.verify(token, jwtSecret, (err, decoded) => {
    if (err) {
      return res.status(403).json({ error: 'Invalid or expired token' });
    }

    req.userId = (decoded as any).userId;
    next();
  });
}

export function generateToken(userId: string): string {
  return jwt.sign({ userId }, getJwtSecret(), { expiresIn: '7d' });
}

export function verifyToken(token: string): { userId: string } | null {
  try {
    return jwt.verify(token, getJwtSecret()) as { userId: string };
  } catch {
    return null;
  }
}

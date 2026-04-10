import { existsSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';

const currentFilePath = fileURLToPath(import.meta.url);
const currentDir = path.dirname(currentFilePath);
const backendDir = path.resolve(currentDir, '..');

const baseEnvPath = path.join(backendDir, '.env');
if (existsSync(baseEnvPath)) {
  dotenv.config({ path: baseEnvPath });
}

const localEnvPath = path.join(backendDir, '.env.local');
if (existsSync(localEnvPath)) {
  dotenv.config({ path: localEnvPath, override: true });
}

const PATTERNS: Array<{ kind: string; re: RegExp }> = [
  { kind: 'aws', re: /AKIA[0-9A-Z]{16}/g },
  { kind: 'github_pat', re: /ghp_[A-Za-z0-9]{36,}/g },
  { kind: 'github_pat', re: /github_pat_[A-Za-z0-9_]{80,}/g },
  { kind: 'openai', re: /sk-[A-Za-z0-9]{20,}/g },
  { kind: 'anthropic', re: /sk-ant-[A-Za-z0-9-]{40,}/g },
  {
    kind: 'private_key',
    re: /-----BEGIN [A-Z ]+PRIVATE KEY-----[\s\S]+?-----END [A-Z ]+PRIVATE KEY-----/g,
  },
];

export interface RedactionResult {
  content: string;
  count: number;
}

export function redact(content: string): RedactionResult {
  let count = 0;
  let out = content;
  for (const { kind, re } of PATTERNS) {
    out = out.replace(re, () => {
      count++;
      return `[REDACTED:${kind}]`;
    });
  }
  return { content: out, count };
}

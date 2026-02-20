---
providers:
  openrouter:
    base_url: "https://openrouter.ai/api/v1"
    api_key: "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"
    models: ["anthropic/claude-sonnet-4", "google/gemini-2.0-flash-001", "deepseek/deepseek-r1", "openai/gpt-4o"]

review:
  max_concurrent_calls: 3
  timeout_seconds: 120
  cost_warning_threshold: 5.00

pivot:
  threshold_percent: 20
  compare_against: "last_reviewed_version"
---

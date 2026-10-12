---
'@mastra/memory': patch
'@mastra/core': patch
---

Added `observation.providerContextTokens`, an opt-in occupancy cap that can trigger Observational Memory when last-step provider usage plus messages added since would fill the model's context window. Omit it to keep current behaviour.

```ts
observationalMemory: {
  observation: { messageTokens: 30_000, providerContextTokens: 100_000 },
}
```

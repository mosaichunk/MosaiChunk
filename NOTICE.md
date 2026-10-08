# Third-party components

Backbone code is fetched at fixed commits rather than copied into this repository.

- [RAVEN](https://github.com/mvp-ai-lab/RAVEN): CC BY-NC 4.0. `third_party/patches/raven.patch` modifies FSDP placement configuration.
- [LingBot-World-v2](https://github.com/robbyant/lingbot-world-v2): CC BY-NC-SA 4.0 (LICENSE.txt). `third_party/patches/lingbot.patch` adds external memory to its causal attention path.
- [Pi3](https://github.com/yyfz/Pi3): BSD 3-Clause. Pi3X reconstructs camera poses for I2V evaluation.
- MiniMax-H3, the RAVEN streaming adapter, LingBot, DL3DV, and Sekai retain their respective source terms. A router checkpoint does not include a backbone license or access to source datasets.

The download scripts preserve repository identities and revisions. Upstream license files remain in the downloaded checkouts.

# Survey of LLM Agents and Tool Use

## TL;DR
- LLM agents combine language models with external tools to overcome limitations in action and knowledge, perceiving environments, reasoning about goals, and executing actions via tool use [1][2][3].
- Core paradigms include prompting as plug-and-play, supervised tool learning, and reward-driven tool policy learning, enabling progressive internalization of tool-use behavior [4][5][6].
- Recent advances introduce real-time monitoring (OnTrack), confidence calibration (MARGIN), and tool merging/retrieval (ToolScope) to improve reliability and scalability [7][8][9][10].
- Industry surveys show growing production adoption (51% of respondents) with emphasis on observability, integration, and tool reliability [11][12].

## Background
LLM agents are defined as systems that perceive environments, reason about goals, and execute actions via tool use, addressing LLMs' inability to directly employ external instruments such as calculators or code interpreters [1][2]. They fall under the category of Learning agents among the five classic agent types (Simple Reflex, Model-based Reflex, Goal-based, Utility-based, Learning) [1]. Foundational work establishes a unified taxonomy covering tool use, planning, and feedback learning, defining universal LLM-profiled roles: policy models, evaluators, and dynamic models [2][5]. Early surveys highlight tool utilization as a critical aspect of action execution, involving tool-use decisions and tool selection, enabling precise calculation, up-to-date information, and code generation [3].

## Agent Architectures and Paradigms
The primary paradigms for building LLM agents with tool use are prompting as plug-and-play, supervised tool learning, and reward-driven tool policy learning [4]. Prompting guides a frozen model to use tools through prompts and in-context interaction without weight updates [4]. Supervised tool learning uses labeled or synthetic supervision to internalize robust tool-use behavior into model parameters [4]. Reward-driven tool policy learning optimizes long-horizon interaction through reinforcement and reward feedback [4]. Beyond prompting, architectures vary: some embed LLMs as copilots operating a host application under step-by-step user confirmation, reflecting a router-worker design [13]; others adopt self-evolving agents that continually update reusable state including model parameters, memories, tool definitions, skills, and workflows from experience [14]. These architectural choices influence safety, as evolving parameters can turn past experiences into future causes [14].

## Tool Management and Selection
Real-world toolsets often contain redundant tools with overlapping names and descriptions, introducing ambiguity and reducing selection accuracy for LLM agents [10][9]. LLMs face strict input context limits that prevent efficient consideration of large toolsets [10][9]. ToolScope addresses these challenges with a ToolScopeMerger that automatically audits and fixes tool merges to reduce redundancy, and a ToolScopeRetriever that ranks and selects only the most relevant tools for each query [10][9]. Additionally, researchers note that the same executable action can be exposed through many functionally equivalent tool definitions, leading to schema bias if agents do not behave consistently across them [15].

## Monitoring, Safety, and Intervention
As LLM agents are deployed in applications like trip planners, stock trading, and IT incident triage, autonomous operation with minimal safeguards can lead to cost and safety issues from irreversible actions [7]. Existing safeguard approaches either add cost and latency per step or are too late (post-hoc log evaluation) [7]. OnTrack proposes a streaming monitoring mechanism using streaming structure-aware optimal transport to compare agent steps in real time, enabling timely intervention [7]. In multi-agent coordination, MARGIN (Multi-Agent Runtime Grading via Incremental Normalisation) learns model-specific confidence corrections from observed answer outcomes without retraining, improving calibration across heterogeneous foundation models [9]. Safety in self-evolving agents is a growing concern because once experience becomes reusable state, past events become future causes, potentially amplifying harmful influences [14].

## Evaluation, Benchmarks, and Applications
Evaluation of LLM agents benefits from specialized benchmarks; for black-box optimization, integrating mathematically rigorous tools shows great potential, though existing studies use diverse domains and configurations, hindering comparability [8]. Industry surveys indicate that 51% of professionals are using agents in production today, with mid-sized companies leading adoption at 63%, and 78% having active plans to implement agents soon [11]. Top use cases include research and summarization (58%) and streamlining tasks for personal productivity or assistance (53.5%) [11]. The Model Context Protocol (MCP) enables instant integrations without additional programming, facilitating plug-and-deploy agents [12]. Tool use and function-calling reliability are rated as the most important benchmarks for predicting action reliability [12].

## Trends and Open Problems
Over the last two years, real-time monitoring and confidence calibration have emerged to improve agent reliability and safety [7][9]. Tool management techniques like merging and context-aware filtering aim to handle large, redundant toolsets efficiently [10][9]. Despite growing adoption, key challenges remain: ensuring consistent behavior across functionally equivalent tool definitions (mitigating schema bias) [15]; balancing autonomy with safeguards to prevent costly or unsafe actions [7]; scaling tool selection under strict context limits [10][9]; and developing standardized benchmarks for fair comparison across agent designs [8]. Additionally, the safety implications of self-evolving agents, where learned parameters can propagate past experiences as future causes, require further study [14]. Open problems also include creating universal tool-use capabilities rather than task-specific solutions [2] and enhancing multi-agent coordination through calibrated confidence sharing [9].

## References
[1] Exploring Large Language Model based Intelligent Agents: Definitions, Methods, and Prospects. web. https://arxiv.org/html/2401.03428 (n.d.)
[2] A Review of Prominent Paradigms for LLM-Based Agents: Tool Use (Including RAG), Planning, and Feedback Learning. web. https://arxiv.org/html/2406.05804 (n.d.)
[3] Large Language Model Agent: A Survey on Methodology .... web. https://arxiv.org/abs/2503.21460 (n.d.)
[4] Agentic Tool Use in Large Language Models: A Survey. web. https://arxiv.org/html/2604.00835 (2026-06-29)
[5] A Review of Prominent Paradigms for LLM-Based Agents. web. https://aclanthology.org/2025.coling-main.652/ (n.d.)
[6] LLM-Based Agents for Tool Learning: A Survey. web. https://link.springer.com/article/10.1007/s41019-025-00296-9 (2025-06-26)
[7] OnTrack: Real-Time Monitoring and Intervention in LLM Agent Trajectories via Streaming Structure-Aware Optimal Transport. arxiv. https://arxiv.org/abs/2610.12375 (2026-10-08)
[8] A Closer Look at Agentic BBO: Benchmarking LLM Agents for Black-Box Optimization. arxiv. https://arxiv.org/abs/2610.12183 (2026-10-08)
[9] MARGIN: Runtime Confidence Calibration for Multi-Agent Foundation Model Coordination. hf-daily. https://huggingface.co/papers/2605.22949 (2026-10-08)
[10] ToolScope: Enhancing LLM Agent Tool Use through Tool Merging and Context-Aware Filtering. hf-search. https://huggingface.co/papers/2510.20036 (2026-05-08)
[11] LangChain State of AI Agents Report: 2024 Trends. web. https://www.langchain.com/stateofaiagents (2026-06-12)
[12] G2 InsightReport: AI Agents 2025. web. https://learn.g2.com/hubfs/G2-InsightReport-AIAgents2025.pdf?hsLang=en (2025-08-01)
[13] Forms of LLM-Integrated Applications from LLM-Chats to Autonomous AI Agent System. arxiv. https://arxiv.org/abs/2610.11899 (2026-10-08)
[14] Safety in Self-Evolving Agents: A Survey. arxiv. https://arxiv.org/abs/2610.00093 (2026-09-08)
[15] Action-Space Shaping for LLM Agents: Measuring and Mitigating Tool-Schema Bias. arxiv. https://arxiv.org/abs/2609.34971 (2026-09-28)

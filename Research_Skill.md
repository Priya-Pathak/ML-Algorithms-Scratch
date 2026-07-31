# OpenCode AI/ML Research Companion Skill

This skill automates weekly tracking, summarization, beginner-friendly filtering, and contribution planning for AI/ML research papers.

## ⚙️ Configuration
```yaml
id: weekly-ml-research-companion
name: Weekly ML Research Companion
description: Fetches, summarizes, filters beginner-friendly AI/ML papers, generates a 1-week implementation plan, and emails the report.
trigger:
  cron: "0 9 * * 1" # Runs every Monday at 9:00 AM UTC
inputs:
  user_email: "your.email@example.com"
  experience_level: "beginner"
```

## 🚀 Execution Workflow

### 1. Fetching Latest Trends
The skill queries open-access repositories and trending repositories over the past 7 days:
*   **API Sources**: [arXiv API](https://arxiv.org/help/api) (`cat:cs.AI`, `cat:cs.LG`), [Hugging Face Papers](https://huggingface.co/papers), and [Papers with Code](https://paperswithcode.com/).
*   **Filter Criteria**: Sort by submission date (past week) and high citation/social traction indicators.

### 2. LLM Summarization Prompt
```text
You are an expert AI research assistant. Review the attached titles and abstracts of the top 5 trending AI/ML papers from the past week. 
Provide a 3-sentence executive summary for each paper covering:
1. Core problem addressed.
2. Proposed methodology.
3. Primary performance results.
```

### 3. Beginner-Friendly Selection Matrix
The skill evaluates fetched papers against the following technical constraint matrix to select the single best paper for a beginner:
*   **Code Availability**: Must have an official, documented GitHub repository.
*   **Framework**: Built using standard high-level libraries (`PyTorch`, `Hugging Face Transformers`, or `Scikit-Learn`).
*   **Hardware Requirements**: Can be trained or fine-tuned on a single free T4 GPU tier (e.g., Google Colab).
*   **Dataset Access**: Uses a publicly accessible, small-to-medium benchmark dataset.

---

## 📅 1-Week Open-Source Contribution Plan

This plan guides you through setting up, understanding, and making your first open-source contribution based on the selected beginner paper.

### Day 1: Environment & Setup
*   **Task**: Fork the official repository of the selected paper.
*   **Action**: Clone it locally and set up a virtual environment (`venv` or `conda`).
*   **Milestone**: Run the installation script (`pip install -r requirements.txt`) and verify zero errors.

### Day 2: Reproduce Baseline
*   **Task**: Run the official inference script or mini-training notebook.
*   **Action**: Download the recommended toy dataset or pre-trained weights.
*   **Milestone**: Successfully generate one baseline output (e.g., a prediction, an image, or text completion).

### Day 3: Code Architecture Deep-Dive
*   **Task**: Trace the core technical execution path.
*   **Action**: Insert breakpoints or print statements inside the main model definition file.
*   **Milestone**: Map out how data flows from raw input to the final loss function/evaluation layer.

### Day 4: Documentation & Bug Hunting
*   **Task**: Audit repository usability and look for open issues.
*   **Action**: Check the repository's GitHub Issue tracker for tags like `good first issue`, `documentation`, or `typo`.
*   **Milestone**: Identify a missing setup edge case in the `README.md` or a typo in code comments.

### Day 5: Implement Enhancement
*   **Task**: Write your modification or improvement.
*   **Action**: Implement missing type hints, add detailed docstrings, or create a simple unit test for an untested utility function.
*   **Milestone**: Verify your changes locally without breaking existing baseline tests.

### Day 6: Upstream Preparation
*   **Task**: Format your codebase to match the repository style guidelines.
*   **Action**: Run style checking tools (`black`, `flake8`, or `isort`) if used by the project.
*   **Milestone**: Commit your changes to a clean, descriptive feature branch (e.g., `git checkout -b fix/docs-setup-guide`).

### Day 7: Submit Pull Request (PR)
*   **Task**: Open a Pull Request to the original open-source repository.
*   **Action**: Write a structured PR description detailing what you changed, why you changed it, and how you tested it.
*   **Milestone**: Submit the PR and link it back to any relevant open issue you resolved.

---

## 📧 Automated Email Notification Template

The skill uses an internal SMTP/Mail service module to send the compiled markdown report directly to the configured `user_email`.

```text
Subject: 🚀 Weekly AI/ML Research Insights & Your Open-Source Roadmap

Hello,

Here is your automated weekly AI/ML research briefing compiled by OpenCode.

=========================================
1. WEEKLY TRENDS & SUMMARIES
=========================================
[Dynamically injected summaries of top 5 trending papers]

=========================================
2. RECOMMENDED BEGINNER IMPLEMENTATION
=========================================
Selected Paper: [Paper Title]
Why it fits: [Reasoning based on technical matrix parameters]

=========================================
3. YOUR 1-WEEK OPEN-SOURCE PLAN
=========================================
- Monday (Day 1): Fork repo & configure environment.
- Tuesday (Day 2): Reproduce baseline inference.
- Wednesday (Day 3): Trace code architecture.
- Thursday (Day 4): Identify good-first-issues/documentation gaps.
- Friday (Day 5): Implement code fixes, comments, or tests.
- Saturday (Day 6): Format code & apply style linters.
- Sunday (Day 7): Submit your Pull Request.

Keep coding,
OpenCode Automation Bot
```

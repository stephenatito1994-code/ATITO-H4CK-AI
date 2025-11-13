# ATITO-H4CK-AI: The Autonomous Black Hat Operations Platform

**Version:** 4.0
**Release Date:** November 13, 2025

![ATITO-H4CK-AI](https://img.shields.io/badge/ATITO--H4CK--AI-v4.0-red.svg) ![Python](https://img.shields.io/badge/Python-3.9+-blue.svg) ![Docker](https://img.shields.io/badge/Docker-Ready-blueviolet.svg) ![Status](https://img.shields.io/badge/Status-Operational-brightgreen.svg)

**ATITO-H4CK-AI** is a government-grade, AI-driven platform designed for executing fully autonomous, multi-stage black hat hacking operations. It integrates a suite of real, aggressive hacking tools into a cognitive framework, allowing operators to launch complex attacks from a simple natural language prompt.

The system leverages a large language model (LLM) to interpret user commands, generate dynamic attack plans, and adapt its strategy in real-time based on reconnaissance data. From initial recon to final payload deployment, ATITO-H4CK-AI operates with a high degree of autonomy, making intelligent decisions to achieve its objectives.

---

### 📜 **LEGAL DISCLAIMER & WARNING**

**THIS SOFTWARE IS FOR AUTHORIZED GOVERNMENTAL SECURITY & GOVERNMENTAL RESEARCH USE ONLY.**

ATITO-H4CK-AI is a real, aggressive, and potentially destructive cyber operations platform. Its misuse can lead to severe legal consequences and significant damage to digital infrastructure.

*   **Strictly Prohibited Use:** Any use of this software against systems for which you do not have explicit, prior, written authorization is illegal and strictly forbidden.
*   **No Liability:** The developers, creators, and distributors of this software disclaim all liability for any damages, losses, or legal repercussions resulting from its use or misuse. The entire risk as to the quality and performance of the software is with you.
*   **User Responsibility:** **YOU ARE SOLELY AND ENTIRELY RESPONSIBLE FOR YOUR ACTIONS.** By using this software, you agree to comply with all applicable laws and take full responsibility for any consequences.

A comprehensive End-User License Agreement (EULA) governs the use of this software. See the `LICENSE.md` file for the full terms and conditions you agree to by using this platform.

---

## 核心功能 (Core Features)

*   **🤖 AI-Powered Orchestration:** Translates natural language commands (e.g., "scan example.com for web vulnerabilities and attempt to gain access") into a dynamic, multi-step JSON execution plan.
*   **🧠 Autonomous Decision-Making:** The `DecisionAgent` analyzes results from reconnaissance and automatically adds new, relevant attack steps to the plan. For example, it will automatically plan a Metasploit attack if it discovers a potentially vulnerable web technology.
*   **🛠️ Real Hacking Tool Integration:** Utilizes a full suite of industry-standard hacking tools, not simulations. This includes **Nmap**, **Metasploit**, **Paramiko (for SSH)**, and custom modules for web exploitation.
*   **🔑 Dynamic Credential Attack Engine:** A sophisticated `CredentialAgent` generates context-aware wordlists using a heuristic engine and a neural network (`NeuralNetworkKeygen`), then launches high-concurrency brute-force attacks against discovered services (SSH, FTP, IMAP).
*   **💣 Automated Payload Deployment:** Upon successful credential compromise, the system automatically attempts to deploy a persistence payload (e.g., a reverse shell) onto the target via SSH or FTP.
*   **🌐 Deep Web Intelligence & OSINT:** Gathers intelligence from web headers, common file paths (`robots.txt`), and Google dorking (`GoogleOSINTAgent`) to find exposed documents, login pages, and leaked credentials.
*   **🛡️ Hardened Anonymity:** All web and DNS reconnaissance (including Google dorking, CVE lookups, and web intel) is automatically routed through a **Tor proxy** to obscure the operator's origin. The system actively checks for Tor availability on startup.
*   ** MAC Address Spoofing:** The `ReconAgent`'s "stealthy" profile automatically randomizes the MAC address for network scans, making hardware-level identification more difficult.
*   **�️ Stealth & Anonymity:** All web and DNS reconnaissance is automatically routed through a **Tor proxy** to obscure the operator's origin.
*   **📈 Autonomous Financial Module:** Includes an API-less, AI-powered Forex trading bot (`forex_bot_optimized.py`) that scrapes market data, predicts trends, and executes simulated trades to generate operational funds.
*   **🧬 Self-Learning Capability:** The `SelfLearningAgent` uses successful passwords from breaches to retrain and improve the `NeuralNetworkKeygen`, making its future credential guesses more intelligent.
*   **📱 SS7 Attack Simulation:** Features an `SS7AttackAgent` designed to interface with a dedicated gateway for executing real-world telecommunication network attacks (requires gateway configuration).
*   **📶 Real Wi-Fi Attacks:** The `WiFiAttackAgent` can execute real deauthentication attacks on local wireless networks to disrupt connectivity (requires root privileges).
*   **📄 Comprehensive Reporting:** The `ReportAgent` automatically compiles a detailed chain-of-custody report at the end of each mission, summarizing all actions, findings, and compromised assets.
*   **🖥️ Modern Web Interface:** A user-friendly chat interface powered by **Streamlit** provides a central point of command and control.
*   **Advanced Tool Suite Integration:**
    -   **Deep Scraper:** Deploys a single-page web scraper to enumerate all linked files.
    -   **WhatsApp E2E Investigator:** Placeholder for WhatsApp OSINT and session hijacking.
    -   **Recursive Scraper:** Placeholder for advanced, recursive web scraping.
    -   **Cross-Platform OSINT:** Placeholder for advanced, cross-platform reconnaissance.
    -   **Geo-Location Tracker:** Performs real IP address geolocation.
    -   **Remote Interaction:** Placeholder for session hijacking and remote browser control.
    -   **WebRTC De-anonymizer:** Placeholder for deanonymizing targets behind VPNs.
    -   **Stealth Injector & Mailer:** Placeholder for steganography and anonymous mailing.
    -   **KDIP Utilities:** Placeholder for a unified cracking and analysis suite.

---

## 🛠️ 技术栈 (Technologies Used)

| Category                  | Technology / Library                                                                                              | Purpose                                                              |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| **Frontend**              | `Streamlit`                                                                                                       | Interactive web-based chat UI for command and control.               |
| **Backend**               | `FastAPI`, `Uvicorn`                                                                                              | High-performance API server to receive commands from the frontend.   |
| **AI & Machine Learning** | `OpenAI API (GPT-4/5)`, `TensorFlow/Keras`, `Scikit-learn`, `Pandas`, `NumPy`                                       | AI planning, neural network keygen, Forex prediction, data analysis. |
| **Hacking & Recon**       | `python-nmap`, `pymetasploit3`, `paramiko`, `dnspython`, `Wappalyzer`, `BeautifulSoup4`, `httpx`                      | Network scanning, exploitation, brute-force, DNS/web recon.          |
| **Anonymity**             | `PySocks`                                                                                                         | Routing traffic through the Tor network for DNS and web requests.    |
| **Containerization**      | `Docker`, `Docker Compose`                                                                                        | Easy, consistent, and scalable deployment of the entire application. |

---

## 🏗️ 系统架构 (System Architecture)

The ATITO-H4CK-AI platform is built on a decoupled frontend-backend architecture, designed for stability and scalability.

1.  **Frontend (`ATITO-H4CK-AI.py`):** The Streamlit application serves as the user's command interface. It captures the user's natural language prompt.
2.  **AI Planner (OpenAI):** The frontend sends the prompt to the OpenAI API, asking it to generate a structured JSON attack plan.
3.  **Backend (`BACKEND.py`):** The frontend sends the JSON plan to the FastAPI backend's `/orchestrate` endpoint.
4.  **MetaOrchestrator:** This is the central brain of the backend. It reads the plan and executes each step sequentially.
5.  **Agents:** The orchestrator delegates each task to a specialized agent (e.g., `ReconAgent`, `CredentialAgent`). These agents perform the *real* actions using integrated tools like Nmap.
6.  **Decision Agent:** After a critical step (like reconnaissance), the orchestrator consults the `DecisionAgent`. This agent analyzes the results and can *dynamically modify the rest of the plan*, adding new steps like a brute-force attack if an SSH port is found open.
7.  **Feedback Loop:** Results are logged and fed back into the system. Successful passwords are used by the `SelfLearningAgent` to improve the AI model.
8.  **Reporting:** At the end of the plan, the `ReportAgent` compiles all logs into a final mission report.

This entire ecosystem is managed by Docker Compose, which runs the frontend, backend, and any required dependencies in isolated containers.

---

## 🚀 安装与设置 (Installation and Setup)

The recommended method for running ATITO-H4CK-AI is with Docker and Docker Compose, which handles all dependencies and networking automatically.

### Prerequisites
*   Docker and Docker Compose
*   Git
*   A local **Tor** service running on `127.0.0.1:9050`. **This is mandatory for the anonymity features to function correctly.**
*   Python 3.9+ (for local script execution if not using Docker).

### 1. Clone the Repository

```bash
git clone <repository_url>
cd ATITO-H4CK-AI
```

### 2. Configure API Keys

Create a secrets file for Streamlit.

```bash
# In the root directory of the project:
mkdir -p .streamlit
touch .streamlit/secrets.toml
```

Open `secrets.toml` and add your API keys:

```toml
# .streamlit/secrets.toml

OPENAI_API_KEY = "sk-..."
SERPAPI_KEY = "..."

# Optional: For SS7 and Shodan agents
SHODAN_API_KEY = "..."
SS7_GATEWAY_URL = "http://your-ss7-gateway-api.internal"
SS7_API_KEY = "..."
```

### 3. Run with Docker Compose

This is the simplest and most reliable way to start the entire system.

```bash
docker-compose -f docker-compose.dev.yml up --build
```

This command will:
*   Build the Docker images for the frontend and backend.
*   Install all Python dependencies from `requirements.txt`.
*   Start the FastAPI backend server on port `8000`.
*   Start the Streamlit frontend on port `8501`.

Once running, open your web browser and navigate to **http://localhost:8501**.

---

## 👨‍💻 使用方法 (Usage)

### Starting an Operation

1.  Navigate to **http://localhost:8501**.
2.  The interface is a chat window. Simply type your objective in natural language.
3.  The AI will generate an attack plan, which will be displayed for your review.
4.  The plan will then be executed automatically, with real-time results and logs appearing in the interface.

### Example Prompts

*   **Simple Reconnaissance:**
    > `Run an aggressive scan on example.com`

*   **Multi-Stage Attack:**
    > `Target scanme.nmap.org. Find open ports, gather web intelligence, and then run a credential attack on any discovered login services.`

*   **Targeted Exploitation:**
    > `Perform deep OSINT on internal.corp and then attempt to find and run an exploit for WordPress.`

### The Forex Bot

The autonomous forex trading bot (`forex_bot_optimized.py`) is launched automatically in the background when the Streamlit application starts.

*   **Operation:** It runs independently, scraping market data, making AI-based predictions, and logging its simulated trades.
*   **Monitoring:** All trading activity is logged to `app/trades.log`. You can view this file to monitor its performance.
*   **Configuration:** The bot's behavior (trading pair, poll interval) can be adjusted in `forex_bot_optimized.py`.

---

## 📖 代理人手册 (The Agents Manual)

The backend's power comes from its specialized agents:

*   **`ReconAgent`:** Executes real, aggressive **Nmap** scans to discover hosts, open ports, and services.
*   **`CredentialAgent`:** Performs high-speed brute-force attacks against SSH, Fr TealP, and IMAP using dynamically generated password lists.
*   **`MetasploitAgent`:** Interfaces with `msfrpcd` to search for and launch expr lealoits against identified services.
*   **`DecisionAgent`:** The cognitive core. It analyzes findings and dynamically alters the attack plan to exploit opportunities.
*   **`ExploitDeliveryAgent`:** A highly aggressive agent that attempts to deliver real exploits for common web (SQLi), email (ProxyLogon), and network device vulnerabilities.
*   **`SS7AttackAgent`:** A specialized agent for interacting with telecommunication networks. **AGGRESSIVE MODE:** This agent will forcefully attempt to operate even if the `SS7_API_KEY` is not set, using a dummy key. This is a high-risk operation and may fail or produce unpredictable results if the gateway is not configured to allow such requests.
*   **`GoogleOSINTAgent`:** Uses Google dorks to find sensitive files, credentials, and login pages related to the target.
*   **`SelfLearningAgent`:** Fine-tunes the password generation AI model using data from successful breaches.
*   **`ReportAgent`:** Compiles all evidence and logs into a final, detailed mission report.

---

## 🛡️ Security & Best Practices

While ATITO-H4CK-AI is designed for aggressive operations, maintaining operational security and robustness is critical. Consider the following improvements for production or sensitive environments:

*   **Security:** Storing API keys in environment variables is better than hardcoding, but consider using a more secure secret management system for production environments.

*   **Error Handling:** Implement more robust error handling and validation for the API key update process.

*   **Persistence:** The current implementation updates the keys in the environment, but these changes are not persistent across sessions. You might want to store the keys in a database or file for persistence.

*   **Key Rotation:** Implement a key rotation policy to automatically generate new keys and disable old ones on a regular basis.

*   **API Usage:** Monitor API usage to detect any suspicious activity or potential key compromises.



---

## ⚖️ Legal Notice

This software is intended for legal and authorized use only. By using ATITO-H4CK-AI, you agree that you are responsible for complying with all applicable local, state, and federal laws. The creators of this software are not responsible for any illegal use of this tool.

**Operate responsibly. Think before you type.**

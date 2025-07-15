# Ethereum-Address-Entity-Classifier-Label-Pipeline-Google-Cloud-_-Gemini-AI-
This project automates the extraction, enrichment, and classification of high-activity Ethereum addresses using on-chain data, open-source label datasets, and state-of-the-art AI models.
## Features:

On-Chain Data Mining: Extracts the most active Ethereum addresses from public BigQuery datasets with custom time windows.

Label Aggregation: Downloads, merges, and updates open-source Ethereum address labels (e.g., from Etherscan-labels) into Google Cloud Storage and BigQuery.

Address-Label Join: Cross-references top addresses with known labels, highlighting new, unlabeled, high-impact addresses.

AI-Powered Classification: Uses Gemini AI to suggest entity types (exchange, personal wallet, DeFi, bot, etc.) for new or unlabeled addresses, with explanations based on on-chain activity, Etherscan, and web snippets.

Robust Cloud Pipeline: Fully managed on Google Cloud—handles caching, checkpointing, batch processing, error recovery, and scalable data storage.

End-to-End Automation: From raw address extraction to final, AI-annotated label datasets ready for blockchain analytics, AML, or research.

## Stack:

Google Cloud Platform: BigQuery, Cloud Storage

Python: Pandas, Requests, BeautifulSoup

AI: Gemini Generative Model (Google)

## Use Cases:

Blockchain analytics and entity mapping

Crypto compliance & AML

DeFi research

Automated updating of Ethereum address labels

import os
import json
import requests
import pandas as pd
import time
from io import BytesIO
from google.cloud import storage, bigquery
from bs4 import BeautifulSoup

import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

# Environment settings
GCP_PROJECT = os.environ.get("GCP_PROJECT", "crypto-prediction-app-465116")
BQ_DATASET = os.environ.get("BQ_DATASET", "crypto_labels_now2")
BQ_TABLE_LABELS = os.environ.get("BQ_TABLE_LABELS", "etherscan_labels")
GCS_BUCKET = os.environ.get("GCS_BUCKET", "crypto-data-bucket-xyz")
LABELS_BLOB = os.environ.get("LABELS_BLOB", "labels/raw/combinedAllLabels.json")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
FULL_REFRESH = os.environ.get("FULL_REFRESH", "false").lower() == "true"

# New variables
START_DATE = os.environ.get("START_DATE")   # "2023-01-01"
END_DATE   = os.environ.get("END_DATE")     # "2023-02-01"

LABELS_URL = "https://raw.githubusercontent.com/brianleect/etherscan-labels/main/data/etherscan/combined/combinedAllLabels.json"

bq_client = bigquery.Client(project=GCP_PROJECT)
storage_client = storage.Client(project=GCP_PROJECT)

def save_to_gcs(df, filename):
    bucket = storage_client.bucket(GCS_BUCKET)
    blob = bucket.blob(filename)
    out = df.to_csv(index=False).encode()
    blob.upload_from_string(out)
    print(f"File saved to GCS: {filename}")

def save_results_to_bigquery(df, table_name):
    job = bq_client.load_table_from_dataframe(
        df,
        f"{GCP_PROJECT}.{BQ_DATASET}.{table_name}",
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    )
    job.result()
    print(f"Results saved to BigQuery table: {table_name}")

def extract_top_addresses(start_date=None, end_date=None, min_txs=5000):
    cache_file = "labels/tmp/active_addresses.csv"
    bucket = storage_client.bucket(GCS_BUCKET)
    blob = bucket.blob(cache_file)
    if not FULL_REFRESH and blob.exists():
        print("Loading addresses from cache file in GCS...")
        df = pd.read_csv(BytesIO(blob.download_as_bytes()))
        print(f"Loaded {len(df)} addresses from cache.")
        return df
    print("Extracting top active addresses from BigQuery...")
    if start_date and end_date:
        sql = f"""
            SELECT LOWER(to_address) as address, COUNT(*) as tx_count
            FROM `bigquery-public-data.crypto_ethereum.token_transfers`
            WHERE block_timestamp BETWEEN TIMESTAMP('{start_date}') AND TIMESTAMP('{end_date}')
            GROUP BY address
            HAVING tx_count > {min_txs}
            ORDER BY tx_count DESC
        """
    else:
        sql = f"""
            SELECT LOWER(to_address) as address, COUNT(*) as tx_count
            FROM `bigquery-public-data.crypto_ethereum.token_transfers`
            WHERE block_timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
            GROUP BY address
            HAVING tx_count > {min_txs}
            ORDER BY tx_count DESC
        """
    df = bq_client.query(sql).to_dataframe()
    save_to_gcs(df, cache_file)
    print(f"Extracted {len(df)} addresses and saved to cache.")
    return df

def download_and_upload_labels():
    print("Downloading labels from GitHub...")
    r = requests.get(LABELS_URL)
    assert r.status_code == 200, "Failed to download from GitHub"
    print("Uploading labels to GCS...")
    bucket = storage_client.bucket(GCS_BUCKET)
    blob = bucket.blob(LABELS_BLOB)
    blob.upload_from_string(r.content)
    print("Labels file uploaded to GCS.")
    return r.json()

def upload_labels_to_bigquery(labels_json):
    if not FULL_REFRESH:
        table_ref = f"{GCP_PROJECT}.{BQ_DATASET}.{BQ_TABLE_LABELS}"
        try:
            table = bq_client.get_table(table_ref)
            print("Labels table already exists in BigQuery. Skipping upload.")
            return None
        except Exception:
            pass
    print("Uploading labels as BigQuery table...")
    data = []
    if isinstance(labels_json, dict):
        for address, info in labels_json.items():
            data.append({
                "address": address.lower(),
                "label": ",".join(info.get("labels", [])) if "labels" in info else "",
                "category": info.get("name", ""),
                "source": "etherscan-github"
            })
    elif isinstance(labels_json, list):
        for rec in labels_json:
            if isinstance(rec, dict):
                data.append({
                    "address": rec.get("address", "").lower(),
                    "label": rec.get("label", ""),
                    "category": rec.get("category", ""),
                    "source": rec.get("source", "")
                })
    else:
        print("Labels file format not supported!")
        return None
    df = pd.DataFrame(data)
    job = bq_client.load_table_from_dataframe(
        df,
        f"{GCP_PROJECT}.{BQ_DATASET}.{BQ_TABLE_LABELS}",
        job_config=bigquery.LoadJobConfig(
            write_disposition="WRITE_TRUNCATE"
        ),
    )
    job.result()
    print("Labels table uploaded successfully.")
    return df

def join_and_classify(active_addresses_df):
    labeled_file = "labels/results/addresses_labeled.csv"
    new_file = "labels/results/addresses_new_to_review.csv"
    bucket = storage_client.bucket(GCS_BUCKET)
    blob_labeled = bucket.blob(labeled_file)
    blob_new = bucket.blob(new_file)

    if not FULL_REFRESH and blob_labeled.exists() and blob_new.exists():
        print("Loading join results from GCS...")
        joined_df = pd.read_csv(BytesIO(blob_labeled.download_as_bytes()))
        new_addresses = pd.read_csv(BytesIO(blob_new.download_as_bytes()))
        return joined_df, new_addresses

    print("Joining active addresses with labels table...")
    addresses_tbl = f"{GCP_PROJECT}.{BQ_DATASET}.{BQ_TABLE_LABELS}"
    temp_table = f"{BQ_DATASET}.temp_active_addresses"
    bq_client.load_table_from_dataframe(
        active_addresses_df, f"{GCP_PROJECT}.{temp_table}",
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    ).result()
    sql = f"""
        SELECT a.address, a.tx_count, l.label, l.category, l.source
        FROM `{GCP_PROJECT}.{temp_table}` AS a
        LEFT JOIN `{addresses_tbl}` AS l
        ON a.address = l.address
    """
    df = bq_client.query(sql).to_dataframe()
    new_addresses = df[df['label'].isna()].copy()
    print(f"New addresses for review: {len(new_addresses)}")
    save_to_gcs(df, labeled_file)
    save_to_gcs(new_addresses, new_file)
    return df, new_addresses

def get_etherscan_label(address):
    url = f"https://etherscan.io/address/{address}"
    try:
        r = requests.get(url, timeout=8)
        soup = BeautifulSoup(r.text, "html.parser")
        label_tag = soup.find("span", {"class": "u-label"})
        if label_tag:
            return label_tag.get_text(strip=True)
        headline = soup.find("span", {"class": "text-truncate"})
        if headline:
            return headline.get_text(strip=True)
        overview = soup.find("div", {"id": "ContentPlaceHolder1_divSummary"})
        if overview:
            return overview.get_text(strip=True)
    except Exception as e:
        return None
    return None

def get_web_snippet(address):
    url = f"https://duckduckgo.com/html/?q={address}+ethereum"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        soup = BeautifulSoup(r.text, "html.parser")
        results = soup.find_all("a", {"class": "result__a"})
        if results:
            return results[0].get_text(strip=True)
    except Exception as e:
        return None
    return None

def is_data_changed(new_df, final_results_blob):
   
    if not final_results_blob.exists():
        return True, set(new_df["address"])
    old_df = pd.read_csv(BytesIO(final_results_blob.download_as_bytes()))
    old_addresses = set(old_df["address"])
    new_addresses = set(new_df["address"])
    addresses_to_add = new_addresses - old_addresses
    return len(addresses_to_add) > 0, addresses_to_add

def classify_with_gemini_super_batch(new_addresses_df, batch_size=25, sleep_between=3):
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if GEMINI_API_KEY is None:
        raise Exception("Please set GEMINI_API_KEY environment variable.")

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')

    final_file = "labels/results/label_all_1.csv"
    checkpoint_file = "labels/tmp/gemini_resume_checkpoint.json"
    bq_table_results = "gemini_classified_results"
    bucket = storage_client.bucket(GCS_BUCKET)
    blob_final = bucket.blob(final_file)
    blob_checkpoint = bucket.blob(checkpoint_file)

    total = len(new_addresses_df)
    batches = (total // batch_size) + (1 if total % batch_size else 0)
    all_results = []

    # Resume from previous results if available
    if blob_final.exists():
        df_prev = pd.read_csv(BytesIO(blob_final.download_as_bytes()))
        all_results = df_prev.to_dict("records")

    # Resume from last completed batch if checkpoint found
    start_batch = 0
    if blob_checkpoint.exists():
        try:
            checkpoint = json.loads(blob_checkpoint.download_as_string().decode("utf-8"))
            start_batch = checkpoint.get("last_completed_batch", 0) + 1
            print(f"Resuming from batch {start_batch+1} ...")
        except Exception:
            start_batch = 0

    for i in range(start_batch, batches):
        batch_df = new_addresses_df.iloc[i*batch_size : (i+1)*batch_size]
        results = []
        for idx, row in batch_df.iterrows():
            address = row['address']
            tx_count = row['tx_count']

            etherscan_desc = get_etherscan_label(address)
            web_snippet = get_web_snippet(address)

            prompt = f"""
                       Address: {address}
                       Transactions in the last month: {tx_count}
                       Etherscan info: {etherscan_desc or 'Not available'}
                       Web search snippet: {web_snippet or 'Not available'}

                       Based on this information and your expertise, what is the most likely classification for this address (exchange, personal wallet, bot, DeFi, ... etc)? If there is not enough evidence, say "Unknown".

                       Format:
                       label: [classification]
                       explanation: [brief explanation]
                       """
            try:
                response = model.generate_content(prompt)
                label, explanation = "", ""
                content = response.text

                for line in content.splitlines():
                    if line.lower().startswith("label:"):
                        label = line.split(":", 1)[1].strip()
                    elif line.lower().startswith("explanation:"):
                        explanation = line.split(":", 1)[1].strip()

                results.append({
                    "address": address,
                    "tx_count": tx_count,
                    "etherscan_desc": etherscan_desc,
                    "web_snippet": web_snippet,
                    "label_gemini": label,
                    "explanation": explanation
                })
                print(f"{address[:8]}... => {label} | {explanation}")
                time.sleep(sleep_between)
            except Exception as e:
                print(f"Error with address {address}: {e}")
                results.append({
                    "address": address,
                    "tx_count": tx_count,
                    "etherscan_desc": etherscan_desc,
                    "web_snippet": web_snippet,
                    "label_gemini": "Error",
                    "explanation": str(e)
                })

        all_results.extend(results)

        # Save and update final results after each batch
        df = pd.DataFrame(all_results)
        save_to_gcs(df, final_file)
        save_results_to_bigquery(df, bq_table_results)
        print(f"Final results updated in GCS and BigQuery after batch {i+1}/{batches}")

        # Save checkpoint for the last completed batch
        checkpoint = {"last_completed_batch": i}
        blob_checkpoint.upload_from_string(json.dumps(checkpoint))
        print(f"Batch {i+1}/{batches} finished and checkpoint updated.")

    # After finishing all batches: delete checkpoint file
    blob_checkpoint.delete()
    print(f"All final results saved in GCS and BigQuery: {final_file}, {bq_table_results}")
    return df

# =========== Main Pipeline ===========
def main():
    addresses_df = extract_top_addresses(
        start_date=START_DATE,
        end_date=END_DATE,
        min_txs=5000
    )
    final_results_blob = storage_client.bucket(GCS_BUCKET).blob("labels/results/label_all_1.csv")
    changed, addresses_to_update = is_data_changed(addresses_df, final_results_blob)

    if not changed:
        print("No new addresses to classify. No further action needed.")
        return

    print(f"{len(addresses_to_update)} new addresses to classify.")
    addresses_to_update_df = addresses_df[addresses_df["address"].isin(addresses_to_update)].copy()

    # Download/upload labels to GCS (if file not found or on refresh)
    labels_blob = storage_client.bucket(GCS_BUCKET).blob(LABELS_BLOB)
    if FULL_REFRESH or not labels_blob.exists():
        labels_json = download_and_upload_labels()
    else:
        print("Labels file already exists in GCS.")
        raw_labels = labels_blob.download_as_string().decode('utf-8')
        try:
            labels_json = json.loads(raw_labels)
            if isinstance(labels_json, dict) and "data" in labels_json:
                labels_json = labels_json["data"]
            if isinstance(labels_json, list) and all(isinstance(x, str) for x in labels_json):
                labels_json = [json.loads(x) for x in labels_json]
        except Exception as e:
            print(f"Failed to parse labels file as JSON: {e}")
            labels_json = {}

    # Upload labels as BigQuery table
    upload_labels_to_bigquery(labels_json)

    # Join addresses with labels and extract new addresses
    joined_df, new_addresses = join_and_classify(addresses_df)

    # Classify only new addresses!
    if not addresses_to_update_df.empty:
        print("Sending new addresses to Gemini API in batches...")
        classified_df = classify_with_gemini_super_batch(addresses_to_update_df, batch_size=25, sleep_between=3)
    else:
        print("No new addresses to classify.")

    print("All done! Results are available in GCS and BigQuery.")

if __name__ == "__main__":
    main()

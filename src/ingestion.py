import urllib.request
import xml.etree.ElementTree as ET
import csv
import time
import os
import ssl
from pathlib import Path

# 15 Highly specific and overlapping categories
CATEGORIES = [
    "cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.NE", "cs.RO",
    "stat.ML", "math.PR", "math.ST", "physics.comp-ph",
    "astro-ph.GA", "astro-ph.HE", "astro-ph.CO",
    "q-bio.NC", "q-bio.QM"
]

TARGET_TOTAL = 50000
MAX_PAPERS_PER_CAT = TARGET_TOTAL // len(CATEGORIES) + 1  # 3334 papers per class
BATCH_SIZE = 500
WAIT_TIME = 3.0    # Required API sleep interval
MAX_RETRIES = 5    # Resilient exponential backoff 

def fetch_arxiv_papers(category, start, max_results):
    """Memory-efficient streaming XML generator over arXiv API."""
    url = f"http://export.arxiv.org/api/query?search_query=cat:{category}&start={start}&max_results={max_results}"
    req = urllib.request.Request(url)
    
    # Bypass Strict SSL validation constraints on disparate systems
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, context=ctx) as response:
                context_iter = ET.iterparse(response, events=("end",))
                records = []
                for event, elem in context_iter:
                    if elem.tag.endswith('entry'):
                        title_elem = elem.find('{http://www.w3.org/2005/Atom}title')
                        summary_elem = elem.find('{http://www.w3.org/2005/Atom}summary')
                        
                        if title_elem is not None and summary_elem is not None:
                            title = title_elem.text.replace('\n', ' ').strip()
                            summary = summary_elem.text.replace('\n', ' ').strip()
                            records.append({'title': title, 'abstract': summary, 'category': category})
                        
                        # Crucial Memory Management: Flush parsed XML nodes immediately
                        elem.clear() 
                return records
                
        except Exception as e:
            wait = 10 * (attempt + 1)
            print(f"    [!] Network error pulling {category}. Retrying in {wait}s...")
            time.sleep(wait)
            
    print(f"    [X] FATAL: Failed to fetch {category} after {MAX_RETRIES} attempts.")
    return []

def main():
    root_dir = Path(__file__).parent.parent
    data_dir = root_dir / 'data' / 'raw'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = data_dir / 'arxiv_50k.csv'
    
    print(f"[*] Starting Ingestion for {TARGET_TOTAL} papers.")
    
    # Default pipeline code designed to generate the entire dataset purely.
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['title', 'abstract', 'category'])
        writer.writeheader()
            
        for cat in CATEGORIES:
            papers_for_cat = 0
            print(f"\n[+] Fetching category: {cat}")
            
            while papers_for_cat < MAX_PAPERS_PER_CAT:
                to_fetch = min(BATCH_SIZE, MAX_PAPERS_PER_CAT - papers_for_cat)
                
                print(f"    -> Querying batch: start={papers_for_cat}, size={to_fetch}...")
                records = fetch_arxiv_papers(cat, papers_for_cat, to_fetch)
                
                if not records:
                    print(f"    -> API Exception. Yielding 0 records. Moving on.")
                    break
                    
                for record in records:
                    writer.writerow(record)
                    
                papers_for_cat += len(records)
                print(f"    -> Stored {papers_for_cat}/{MAX_PAPERS_PER_CAT} records.")
                time.sleep(WAIT_TIME)
                
    print(f"\n[DONE] Streaming Ingestion Complete!")

if __name__ == "__main__":
    main()

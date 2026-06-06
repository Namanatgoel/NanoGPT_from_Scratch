import os
import urllib.request
import xml.etree.ElementTree as ET
import re

def fetch_arxiv_domain_corpus(output_path, max_results=150):
    print(f"[INFO] Initializing ArXiv extraction for {max_results} papers...")
    
    # Target categories: Machine Learning (cs.LG), Artificial Intelligence (cs.AI)
    # Target domains: Generative AI, Reinforcement Learning, VAEs
    query = 'cat:cs.LG+AND+(%22Reinforcement+Learning%22+OR+%22Generative+AI%22+OR+%22Variational+Autoencoder%22)'
    url = f'http://export.arxiv.org/api/query?search_query={query}&max_results={max_results}'
    
    try:
        response = urllib.request.urlopen(url)
        xml_data = response.read()
    except Exception as e:
        print(f"[ERROR] Failed to fetch data from ArXiv API: {e}")
        print("[FALLBACK] API failure. Proceed immediately to Option B (Hugging Face Corpora).")
        return False

    root = ET.fromstring(xml_data)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    
    corpus_entries = []
    
    for entry in root.findall('atom:entry', ns):
        title = entry.find('atom:title', ns).text.strip()
        summary = entry.find('atom:summary', ns).text.strip()
        
        # Clean inline linebreaks from API text delivery
        title = re.sub(r'\s+', ' ', title)
        summary = re.sub(r'\s+', ' ', summary)
        
        corpus_entries.append(f"Title: {title}\nAbstract: {summary}\n")

    if not corpus_entries:
        print("[ERROR] Zero entries recovered from the API query.")
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(corpus_entries))
        
    print(f"[SUCCESS] Domain corpus compiled at: {output_path}")
    print(f"[METRIC] Total documents extracted: {len(corpus_entries)}")
    return True

if __name__ == "__main__":
    target_file = os.path.join("project", "data", "arxiv_domain_raw.txt")
    success = fetch_arxiv_domain_corpus(target_file, max_results=150)
    if not success:
        exit(1)
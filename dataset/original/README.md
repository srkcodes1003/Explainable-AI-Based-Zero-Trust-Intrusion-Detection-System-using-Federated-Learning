# CIC-IDS2017 Dataset Directory

Place the uncompressed CSV files from the **CIC-IDS2017 (MachineLearningCSV)** dataset directly in this folder (`dataset/original/`).

### Official Source
- **Website:** [Canadian Institute for Cybersecurity (UNB)](https://www.unb.ca/cic/datasets/ids-2017.html)
- **Archive File:** `MachineLearningCSV.zip` or `GeneratedLabelledFlows.zip`

### Expected CSV Files
- `Tuesday-WorkingHours.pcap_ISCX.csv` (FTP-Patator, SSH-Patator)
- `Wednesday-workingHours.pcap_ISCX.csv` (DoS / Heartbleed)
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` (Web Attacks: Brute Force, XSS, SQL Injection)
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` (Infiltration)
- `Friday-WorkingHours-Morning.pcap_ISCX.csv` (Botnet)
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` (PortScan)
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` (DDoS)

### Note on Dynamic Discovery
The exploration and preprocessing scripts in this project **dynamically discover** any `.csv` files in this directory. 
If the file names differ or if only a subset of files is present, the pipeline automatically detects them.


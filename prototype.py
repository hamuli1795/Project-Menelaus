import pandas as pd
import json
import os
from datetime import datetime

# --- CONFIGURATION & FILE NAMES ---

EXCEL_FILE = "12_week_prep.xlsx"
CSV_FILE = "Routines.csv"
JSON_LOG = "vitals_history.json"

def bootstrap_files():
    """Creates dummy files so the script runs on the first try in Replit."""
    if not os.path.exists(EXCEL_FILE):
        df = pd.DataFrame({"Date": [datetime.now().date()], "RPE": [8], "Lift": ["Squat"]})
        df.to_excel(EXCEL_FILE, index=False)
        print(f"Created template: {EXCEL_FILE}")

    if not os.path.exists(CSV_FILE):
        df = pd.DataFrame({"Date": [datetime.now().date()], "Pain_Scale": [2], "Scan_Efficiency": [6]})
        df.to_csv(CSV_FILE, index=False)
        print(f"Created template: {CSV_FILE}")

def get_external_data():
    """Extracts the latest 'Telemetry' from your spreadsheets."""
    try:
        # Pulling from Excel (Powerlifting)
        pl_df = pd.read_excel(EXCEL_FILE)
        latest_rpe = pl_df.iloc[-1]["RPE"]

        # Pulling from CSV (Hoops/Recovery)
        bb_df = pd.read_csv(CSV_FILE)
        latest_pain = bb_df.iloc[-1]["Pain_Scale"]
        latest_scan = bb_df.iloc[-1]["Scan_Efficiency"]
        
        return latest_rpe, latest_pain, latest_scan
    except Exception as e:
        print(f"⚠️ Sync Error: {e}. Using default safety values.")
        return 5, 0, 5  # Default safe values

def calculate_protocol(rpe, pain, scan):
    """The 'Egoist' Logic: Decides your daily Haki."""
    protocol = {"Intensity": "OPTIMAL", "Focus": "Skill Work", "Recovery": "Standard"}

    # Hardware Safety (Injury Firewall)
    if pain >= 7:
        protocol.update({"Intensity": "LOCKOUT", "Focus": "Film Study", "Recovery": "Ice/Sleep"})
    elif pain >= 4 or rpe >= 9:
        protocol.update({"Intensity": "LOW IMPACT", "Focus": "Soccer Scanning/Free Throws", "Recovery": "Isometrics"})
    
    # Skill Patch (Rugby/Soccer)
    if scan < 7:
        protocol["Focus"] += " + Xavi Scanning Drills"
    
    return protocol

def main():
    bootstrap_files()
    rpe, pain, scan = get_external_data()
    protocol = calculate_protocol(rpe, pain, scan)

    print("\n" + "="*40)
    print(f"🚀 VITALS ENGINE REPORT: {datetime.now().date()}")
    print(f"📊 DATA SYNC: Squat RPE: {rpe} | Knee Pain: {pain} | Scan IQ: {scan}")
    print("-" * 40)
    print(f"🔥 TODAY'S INTENSITY: {protocol['Intensity']}")
    print(f"🎯 FOCUS: {protocol['Focus']}")
    print(f"🛠️ RECOVERY: {protocol['Recovery']}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()

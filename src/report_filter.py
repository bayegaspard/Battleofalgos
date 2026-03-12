IMPORTANT_KEYS = [
    "size",
    "type",
    "submit_name",
    "sha256",
    "av_detect",
    "vx_family",
    "threat_score",
    "verdict",
    "file_metadata",
    "processes",
    "mitre_attcks",
    "network_mode",
    "signatures"
]

def filter_report(report):

    filtered = {}

    for key in IMPORTANT_KEYS:
        if key in report:
            filtered[key] = report[key]

    return filtered

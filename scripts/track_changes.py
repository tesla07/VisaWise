"""Track changes between scraping runs."""
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def compare_runs(old_dir, new_dir):
    """Compare two scraping runs and identify changes."""
    old_dir = Path(old_dir)
    new_dir = Path(new_dir)
    
    old_files = {f.name: f for f in old_dir.glob("*.json")}
    new_files = {f.name: f for f in new_dir.glob("*.json")}
    
    changes = {
        "added": [],
        "removed": [],
        "modified": [],
        "unchanged": []
    }
    
    # Find added files
    for name in new_files:
        if name not in old_files:
            changes["added"].append(name)
    
    # Find removed files
    for name in old_files:
        if name not in new_files:
            changes["removed"].append(name)
    
    # Find modified files
    for name in old_files:
        if name in new_files:
            old_data = json.loads(old_files[name].read_text(encoding='utf-8'))
            new_data = json.loads(new_files[name].read_text(encoding='utf-8'))
            
            # Compare content
            old_text = " ".join(chunk.get("text", "") for chunk in old_data)
            new_text = " ".join(chunk.get("text", "") for chunk in new_data)
            
            if old_text != new_text:
                changes["modified"].append({
                    "file": name,
                    "old_chunks": len(old_data),
                    "new_chunks": len(new_data),
                    "text_diff": len(new_text) - len(old_text)
                })
            else:
                changes["unchanged"].append(name)
    
    return changes

def generate_change_report(old_dir="data/processed", new_dir="data/runs/latest"):
    """Generate a human-readable change report."""
    changes = compare_runs(old_dir, new_dir)
    
    print("=" * 80)
    print("USCIS DATA CHANGE REPORT")
    print(f"Comparing: {old_dir} → {new_dir}")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)
    
    print(f"\n📊 SUMMARY")
    print(f"  New pages: {len(changes['added'])}")
    print(f"  Removed pages: {len(changes['removed'])}")
    print(f"  Modified pages: {len(changes['modified'])}")
    print(f"  Unchanged pages: {len(changes['unchanged'])}")
    
    if changes["added"]:
        print(f"\n✨ NEW PAGES ({len(changes['added'])})")
        for name in changes["added"][:10]:
            print(f"  + {name}")
        if len(changes["added"]) > 10:
            print(f"  ... and {len(changes['added']) - 10} more")
    
    if changes["removed"]:
        print(f"\n🗑️  REMOVED PAGES ({len(changes['removed'])})")
        for name in changes["removed"][:10]:
            print(f"  - {name}")
        if len(changes["removed"]) > 10:
            print(f"  ... and {len(changes['removed']) - 10} more")
    
    if changes["modified"]:
        print(f"\n📝 MODIFIED PAGES ({len(changes['modified'])})")
        for item in changes["modified"][:10]:
            print(f"  ~ {item['file']}")
            print(f"    Chunks: {item['old_chunks']} → {item['new_chunks']}")
            print(f"    Text change: {item['text_diff']:+d} chars")
        if len(changes["modified"]) > 10:
            print(f"  ... and {len(changes['modified']) - 10} more")
    
    print("\n" + "=" * 80)
    
    # Save detailed report
    report_file = Path("data/change_reports") / f"changes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_file.parent.mkdir(exist_ok=True)
    report_file.write_text(json.dumps(changes, indent=2, default=str))
    print(f"\nDetailed report saved to: {report_file}")
    
    return changes

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        generate_change_report(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python track_changes.py <old_dir> <new_dir>")
        print("Example: python track_changes.py data/processed data/runs/2025-12-01")


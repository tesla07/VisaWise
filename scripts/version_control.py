"""Version control system for scraped USCIS data."""
import json
import shutil
from pathlib import Path
from datetime import datetime
import hashlib

class DataVersionControl:
    """Manage versions of scraped immigration data."""
    
    def __init__(self, base_dir="data"):
        self.base_dir = Path(base_dir)
        self.processed_dir = self.base_dir / "processed"
        self.versions_dir = self.base_dir / "versions"
        self.metadata_file = self.versions_dir / "version_metadata.json"
        
        # Create directories
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        
    def create_version(self, description="", tags=None):
        """Create a new version snapshot of current data."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_name = f"v_{timestamp}"
        version_dir = self.versions_dir / version_name
        
        print(f"Creating version: {version_name}")
        
        # Copy current data to version directory
        if self.processed_dir.exists():
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy files individually to handle long paths on Windows
            for json_file in self.processed_dir.glob("*.json"):
                try:
                    shutil.copy2(json_file, version_dir / json_file.name)
                except Exception as e:
                    print(f"  ⚠️  Skipping {json_file.name}: {e}")
            
            # Calculate stats
            file_count = len(list(version_dir.glob("*.json")))
            total_size = sum(f.stat().st_size for f in version_dir.glob("*.json"))
            
            # Create version metadata
            metadata = {
                "version": version_name,
                "timestamp": timestamp,
                "datetime": datetime.now().isoformat(),
                "description": description,
                "tags": tags or [],
                "file_count": file_count,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "checksum": self._calculate_checksum(version_dir)
            }
            
            # Save version-specific metadata
            version_meta_file = version_dir / "version_info.json"
            version_meta_file.write_text(json.dumps(metadata, indent=2))
            
            # Update global metadata
            self._update_global_metadata(metadata)
            
            print(f"✓ Version created: {version_name}")
            print(f"  Files: {file_count}")
            print(f"  Size: {metadata['total_size_mb']} MB")
            print(f"  Location: {version_dir}")
            
            return version_name, metadata
        else:
            print(f"✗ No data found in {self.processed_dir}")
            return None, None
    
    def list_versions(self):
        """List all available versions."""
        if not self.metadata_file.exists():
            print("No versions found.")
            return []
        
        metadata = json.loads(self.metadata_file.read_text())
        versions = metadata.get("versions", [])
        
        print("=" * 80)
        print("AVAILABLE DATA VERSIONS")
        print("=" * 80)
        
        for v in reversed(versions):  # Show newest first
            print(f"\n📦 {v['version']}")
            print(f"   Date: {v['datetime']}")
            print(f"   Files: {v['file_count']}")
            print(f"   Size: {v['total_size_mb']} MB")
            if v.get('description'):
                print(f"   Description: {v['description']}")
            if v.get('tags'):
                print(f"   Tags: {', '.join(v['tags'])}")
        
        print("\n" + "=" * 80)
        return versions
    
    def restore_version(self, version_name):
        """Restore a specific version to processed directory."""
        version_dir = self.versions_dir / version_name
        
        if not version_dir.exists():
            print(f"✗ Version {version_name} not found")
            return False
        
        # Backup current data first
        backup_name = f"backup_before_{version_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.create_version(description=f"Backup before restoring {version_name}", tags=["backup"])
        
        # Clear processed directory
        if self.processed_dir.exists():
            shutil.rmtree(self.processed_dir)
        
        # Restore version
        shutil.copytree(version_dir, self.processed_dir)
        
        print(f"✓ Restored version: {version_name}")
        print(f"  Location: {self.processed_dir}")
        
        return True
    
    def compare_versions(self, version1, version2):
        """Compare two versions and show differences."""
        v1_dir = self.versions_dir / version1
        v2_dir = self.versions_dir / version2
        
        if not v1_dir.exists() or not v2_dir.exists():
            print("✗ One or both versions not found")
            return None
        
        v1_files = {f.name for f in v1_dir.glob("*.json")}
        v2_files = {f.name for f in v2_dir.glob("*.json")}
        
        added = v2_files - v1_files
        removed = v1_files - v2_files
        common = v1_files & v2_files
        
        print("=" * 80)
        print(f"COMPARING VERSIONS: {version1} → {version2}")
        print("=" * 80)
        
        print(f"\n📊 SUMMARY")
        print(f"  Added files: {len(added)}")
        print(f"  Removed files: {len(removed)}")
        print(f"  Common files: {len(common)}")
        
        if added:
            print(f"\n✨ ADDED FILES ({len(added)})")
            for f in sorted(list(added)[:10]):
                print(f"  + {f}")
            if len(added) > 10:
                print(f"  ... and {len(added) - 10} more")
        
        if removed:
            print(f"\n🗑️  REMOVED FILES ({len(removed)})")
            for f in sorted(list(removed)[:10]):
                print(f"  - {f}")
            if len(removed) > 10:
                print(f"  ... and {len(removed) - 10} more")
        
        print("\n" + "=" * 80)
        
        return {
            "added": list(added),
            "removed": list(removed),
            "common": list(common)
        }
    
    def delete_version(self, version_name, confirm=False):
        """Delete a specific version."""
        version_dir = self.versions_dir / version_name
        
        if not version_dir.exists():
            print(f"✗ Version {version_name} not found")
            return False
        
        if not confirm:
            print(f"⚠️  This will permanently delete version: {version_name}")
            print(f"   Run with confirm=True to proceed")
            return False
        
        shutil.rmtree(version_dir)
        
        # Update metadata
        if self.metadata_file.exists():
            metadata = json.loads(self.metadata_file.read_text())
            metadata["versions"] = [v for v in metadata["versions"] if v["version"] != version_name]
            self.metadata_file.write_text(json.dumps(metadata, indent=2))
        
        print(f"✓ Deleted version: {version_name}")
        return True
    
    def cleanup_old_versions(self, keep_last=6):
        """Keep only the N most recent versions."""
        if not self.metadata_file.exists():
            print("No versions to clean up")
            return
        
        metadata = json.loads(self.metadata_file.read_text())
        versions = metadata.get("versions", [])
        
        if len(versions) <= keep_last:
            print(f"Only {len(versions)} versions exist, keeping all")
            return
        
        # Sort by timestamp
        versions_sorted = sorted(versions, key=lambda x: x["timestamp"], reverse=True)
        
        to_delete = versions_sorted[keep_last:]
        
        print(f"Keeping {keep_last} most recent versions")
        print(f"Deleting {len(to_delete)} old versions:")
        
        for v in to_delete:
            print(f"  - {v['version']} ({v['datetime']})")
            self.delete_version(v["version"], confirm=True)
        
        print(f"✓ Cleanup complete")
    
    def _calculate_checksum(self, directory):
        """Calculate checksum for all files in directory."""
        hasher = hashlib.sha256()
        
        for json_file in sorted(directory.glob("*.json")):
            hasher.update(json_file.read_bytes())
        
        return hasher.hexdigest()
    
    def _update_global_metadata(self, version_metadata):
        """Update global version metadata file."""
        if self.metadata_file.exists():
            metadata = json.loads(self.metadata_file.read_text())
        else:
            metadata = {"versions": []}
        
        metadata["versions"].append(version_metadata)
        metadata["last_updated"] = datetime.now().isoformat()
        metadata["total_versions"] = len(metadata["versions"])
        
        self.metadata_file.write_text(json.dumps(metadata, indent=2))


def main():
    """CLI interface for version control."""
    import sys
    
    vc = DataVersionControl()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python version_control.py create [description] [tags]")
        print("  python version_control.py list")
        print("  python version_control.py restore <version_name>")
        print("  python version_control.py compare <version1> <version2>")
        print("  python version_control.py cleanup [keep_last]")
        return
    
    command = sys.argv[1]
    
    if command == "create":
        description = sys.argv[2] if len(sys.argv) > 2 else ""
        tags = sys.argv[3:] if len(sys.argv) > 3 else []
        vc.create_version(description, tags)
    
    elif command == "list":
        vc.list_versions()
    
    elif command == "restore":
        if len(sys.argv) < 3:
            print("✗ Please specify version name")
            return
        vc.restore_version(sys.argv[2])
    
    elif command == "compare":
        if len(sys.argv) < 4:
            print("✗ Please specify two version names")
            return
        vc.compare_versions(sys.argv[2], sys.argv[3])
    
    elif command == "cleanup":
        keep = int(sys.argv[2]) if len(sys.argv) > 2 else 6
        vc.cleanup_old_versions(keep)
    
    else:
        print(f"✗ Unknown command: {command}")


if __name__ == "__main__":
    main()


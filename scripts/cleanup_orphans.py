"""Script to clean up orphaned detection records."""
import os
import sys

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_dir, 'src'))

# Change to project directory for relative path resolution
os.chdir(project_dir)

from anpr.database.manager import DatabaseManager
from anpr.database.utils import cleanup_orphaned_detections

def main():
    # Use default database (will use data/anpr.db)
    db = DatabaseManager()
    
    with db.get_session() as session:
        # First do a dry run
        count, ids = cleanup_orphaned_detections(session, dry_run=True)
        print(f"Found {count} orphaned detection records")
        
        if count > 0:
            # Now actually delete
            count, ids = cleanup_orphaned_detections(session, dry_run=False)
            print(f"Deleted {count} orphaned detection records")
            print(f"Deleted IDs: {ids[:20]}..." if len(ids) > 20 else f"Deleted IDs: {ids}")
        
        # Show remaining count
        from anpr.database.models import Detection
        remaining = session.query(Detection).count()
        print(f"Remaining valid detections: {remaining}")

if __name__ == "__main__":
    main()

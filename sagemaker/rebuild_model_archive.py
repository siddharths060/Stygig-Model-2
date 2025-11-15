#!/usr/bin/env python3
"""
Quick fix: Rebuild model.tar.gz with missing stygig source code

This script:
1. Downloads the existing model.tar.gz from S3
2. Extracts it
3. Adds the missing src/stygig directory
4. Adds the missing config/ directory
5. Rebuilds and uploads the fixed model.tar.gz
"""

import os
import sys
import shutil
import tarfile
import tempfile
import subprocess
from pathlib import Path

def run_command(cmd):
    """Run shell command and return output."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise RuntimeError(f"Command failed: {cmd}")
    return result.stdout

def main():
    # Configuration
    bucket = "stygig-ml-s3"
    model_s3_uri = "s3://stygig-ml-s3/model-artifacts/stygig-training-1762502640/output/model.tar.gz"
    output_s3_uri = "s3://stygig-ml-s3/model-artifacts/stygig-async-fixed/model.tar.gz"
    
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src"
    config_dir = project_root / "config"
    
    print("=" * 70)
    print("REBUILD MODEL ARCHIVE - Add Missing Source Code")
    print("=" * 70)
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        work_dir = temp_path / "model_work"
        work_dir.mkdir()
        
        print("\n[1/6] Downloading existing model.tar.gz from S3...")
        model_archive = temp_path / "model.tar.gz"
        run_command(f"aws s3 cp {model_s3_uri} {model_archive}")
        
        print("\n[2/6] Extracting model.tar.gz...")
        with tarfile.open(model_archive, 'r:gz') as tar:
            tar.extractall(work_dir)
        
        # List current contents
        print("\nCurrent contents:")
        for item in work_dir.rglob('*'):
            if item.is_file():
                print(f"  - {item.relative_to(work_dir)}")
        
        print("\n[3/6] Adding missing src/stygig directory...")
        target_src_dir = work_dir / "src"
        target_src_dir.mkdir(parents=True, exist_ok=True)
        
        if src_dir.exists():
            shutil.copytree(src_dir / "stygig", target_src_dir / "stygig", dirs_exist_ok=True)
            print(f"  ✓ Copied {src_dir / 'stygig'} -> {target_src_dir / 'stygig'}")
        else:
            print(f"  ✗ Error: Source directory not found: {src_dir}")
            sys.exit(1)
        
        print("\n[4/6] Adding config directory...")
        if config_dir.exists():
            shutil.copytree(config_dir, work_dir / "config", dirs_exist_ok=True)
            print(f"  ✓ Copied {config_dir} -> {work_dir / 'config'}")
        else:
            print(f"  ! Warning: Config directory not found: {config_dir}")
        
        # List new contents
        print("\nUpdated contents:")
        for item in work_dir.rglob('*'):
            if item.is_file():
                print(f"  - {item.relative_to(work_dir)}")
        
        print("\n[5/6] Creating new model.tar.gz...")
        new_archive = temp_path / "model_fixed.tar.gz"
        
        with tarfile.open(new_archive, 'w:gz') as tar:
            for item in work_dir.iterdir():
                tar.add(item, arcname=item.name)
        
        file_size_mb = new_archive.stat().st_size / (1024 * 1024)
        print(f"  ✓ Created archive: {file_size_mb:.1f} MB")
        
        print("\n[6/6] Uploading fixed model.tar.gz to S3...")
        run_command(f"aws s3 cp {new_archive} {output_s3_uri}")
        
        print("\n" + "=" * 70)
        print("✅ SUCCESS!")
        print("=" * 70)
        print(f"\nFixed model uploaded to: {output_s3_uri}")
        print("\nNext steps:")
        print("1. Deploy new endpoint with this fixed model:")
        print(f"   python sagemaker/deploy_async_endpoint.py \\")
        print(f"       --model-data {output_s3_uri} \\")
        print(f"       --sns-topic-arn arn:aws:sns:ap-south-1:732414292744:SNSMLTopic \\")
        print(f"       --endpoint-name stygig-async-fixed-test")
        print("\n2. Or update existing endpoint:")
        print(f"   aws sagemaker update-endpoint \\")
        print(f"       --endpoint-name stygig-async-endpoint-20251115-101546 \\")
        print(f"       --endpoint-config-name <new-config-name>")

if __name__ == "__main__":
    main()

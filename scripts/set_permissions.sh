#!/bin/bash

################################################################################
# StyGig - Set Script Permissions
################################################################################
#
# This script makes all shell scripts in the scripts/ directory executable.
# Run this after cloning the repository or updating scripts.
#
# Usage:
#   ./scripts/set_permissions.sh
#
################################################################################

echo "Making shell scripts executable..."
echo ""

# Change to project root
cd "$(dirname "$0")/.."

# Counter for files updated
COUNT=0

# Make scripts in scripts/ directory executable
if [ -d "scripts" ]; then
    for script in scripts/*.sh; do
        if [ -f "$script" ]; then
            chmod +x "$script"
            echo "✅ $script"
            COUNT=$((COUNT + 1))
        fi
    done
fi

# Make scripts in scripts/testing/ directory executable
if [ -d "scripts/testing" ]; then
    for script in scripts/testing/*.sh; do
        if [ -f "$script" ]; then
            chmod +x "$script"
            echo "✅ $script"
            COUNT=$((COUNT + 1))
        fi
    done
fi

echo ""
echo "✅ Done! Made $COUNT script(s) executable."
echo ""
echo "You can now run:"
echo "  ./scripts/run_pipeline.sh          # Full ML pipeline"
echo "  ./scripts/deploy_model.sh          # Deploy model only"
echo "  ./scripts/testing/verify_structure.sh   # Verify project structure"


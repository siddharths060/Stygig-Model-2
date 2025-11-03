#!/bin/bash

# Make all shell scripts executable
# Run this after cloning the repository

echo "Making shell scripts executable..."

chmod +x run_project.sh
chmod +x verify_structure.sh
chmod +x make_executable.sh

echo "✅ Done! All scripts are now executable."
echo ""
echo "You can now run:"
echo "  ./verify_structure.sh"
echo "  ./run_project.sh"

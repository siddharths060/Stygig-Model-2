#!/bin/bash

################################################################################
# StyGig Project Structure Verification Script
################################################################################
#
# Verifies that the refactored project structure is correct and all
# required files and directories are in place.
#
# Usage:
#   ./scripts/testing/verify_structure.sh
#
################################################################################

echo "════════════════════════════════════════════════════════════════════════════════"
echo "   StyGig Project Structure Verification"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Change to project root
cd "$(dirname "$0")/../.."

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

total_checks=0
passed_checks=0

check_file() {
    total_checks=$((total_checks + 1))
    if [ -f "$1" ]; then
        echo -e "${GREEN}✅${NC} $1"
        passed_checks=$((passed_checks + 1))
        return 0
    else
        echo -e "${RED}❌${NC} $1"
        return 1
    fi
}

check_dir() {
    total_checks=$((total_checks + 1))
    if [ -d "$1" ]; then
        echo -e "${GREEN}✅${NC} $1/"
        passed_checks=$((passed_checks + 1))
        return 0
    else
        echo -e "${RED}❌${NC} $1/"
        return 1
    fi
}

echo "📁 Checking Core Directory Structure"
echo "────────────────────────────────────────────────────────────────────────────────"
check_dir "src"
check_dir "src/stygig"
check_dir "src/stygig/core"
check_dir "src/stygig/core/rules"
check_dir "src/stygig/api"
check_dir "src/stygig/utils"
check_dir "sagemaker"
check_dir "config"
check_dir "outputs"
check_dir "outfits_dataset"
echo ""

echo "📁 Checking New Scripts Directory Structure"
echo "────────────────────────────────────────────────────────────────────────────────"
check_dir "scripts"
check_dir "scripts/testing"
echo ""

echo "📄 Checking Core Package Files"
echo "────────────────────────────────────────────────────────────────────────────────"
check_file "src/stygig/__init__.py"
check_file "src/stygig/core/__init__.py"
check_file "src/stygig/core/recommendation_engine.py"
check_file "src/stygig/core/color_logic.py"
check_file "src/stygig/core/gender_logic.py"
check_file "src/stygig/core/rules/__init__.py"
check_file "src/stygig/core/rules/category_compatibility.py"
check_file "src/stygig/api/__init__.py"
check_file "src/stygig/utils/__init__.py"
echo ""

echo "📄 Checking Config Files"
echo "────────────────────────────────────────────────────────────────────────────────"
check_file "config/__init__.py"
check_file "config/settings.py"
check_file "config/recommendation_config.py"
echo ""

echo "📄 Checking SageMaker Files"
echo "────────────────────────────────────────────────────────────────────────────────"
check_file "sagemaker/train.py"
check_file "sagemaker/inference.py"
check_file "sagemaker/run_sagemaker_pipeline.py"
check_file "sagemaker/requirements.txt"
echo ""

echo "📄 Checking New Scripts (Consolidated)"
echo "────────────────────────────────────────────────────────────────────────────────"
check_file "scripts/run_pipeline.sh"
check_file "scripts/deploy_model.sh"
check_file "scripts/manage_endpoints.py"
check_file "scripts/set_permissions.sh"
echo ""

echo "📄 Checking Testing Scripts"
echo "────────────────────────────────────────────────────────────────────────────────"
check_file "scripts/testing/test_endpoint.py"
check_file "scripts/testing/integration_test.py"
check_file "scripts/testing/local_train_test.py"
check_file "scripts/testing/verify_structure.sh"
echo ""

echo "📄 Checking Root Files"
echo "────────────────────────────────────────────────────────────────────────────────"
check_file "README.md"
check_file ".gitignore"
check_file "requirements_local.txt"
echo ""

echo "🔍 Checking Placeholders"
echo "────────────────────────────────────────────────────────────────────────────────"
check_file "outputs/.gitkeep"
check_file "outfits_dataset/.gitkeep"
echo ""

# Summary
echo "════════════════════════════════════════════════════════════════════════════════"
echo "   Verification Summary"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Total Checks: $total_checks"
echo -e "${GREEN}Passed: $passed_checks${NC}"
failed_checks=$((total_checks - passed_checks))
if [ $failed_checks -gt 0 ]; then
    echo -e "${RED}Failed: $failed_checks${NC}"
else
    echo -e "${GREEN}All checks passed!${NC}"
fi
echo ""

if [ $failed_checks -eq 0 ]; then
    echo -e "${GREEN}✅ Project structure is correct!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Review README.md for usage instructions"
    echo "  2. Run: ./scripts/run_pipeline.sh"
    echo "  3. Or deploy existing model: ./scripts/deploy_model.sh --model-uri s3://..."
    echo "  4. Manage endpoints: python scripts/manage_endpoints.py list"
    exit 0
else
    echo -e "${RED}❌ Some files or directories are missing${NC}"
    echo ""
    echo "Please check the script.md for the expected structure"
    exit 1
fi

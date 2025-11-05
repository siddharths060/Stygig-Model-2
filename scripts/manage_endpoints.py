#!/usr/bin/env python3
"""
StyGig Endpoint Management Utility

This script helps manage AWS SageMaker endpoints for the StyGig project.
It provides functionality to list, delete, and clean up endpoints.

Usage:
    python scripts/manage_endpoints.py list                    # List all StyGig endpoints
    python scripts/manage_endpoints.py delete --endpoint-name NAME   # Delete specific endpoint
    python scripts/manage_endpoints.py delete-all              # Delete all StyGig endpoints
    python scripts/manage_endpoints.py info --endpoint-name NAME     # Get endpoint details

Examples:
    # List all endpoints
    python scripts/manage_endpoints.py list

    # Delete a specific endpoint
    python scripts/manage_endpoints.py delete --endpoint-name stygig-endpoint-20251103-062336

    # Delete all StyGig endpoints (with confirmation)
    python scripts/manage_endpoints.py delete-all

    # Get detailed info about an endpoint
    python scripts/manage_endpoints.py info --endpoint-name stygig-endpoint-20251103-062336
"""

import argparse
import sys
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional

import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EndpointManager:
    """Manages SageMaker endpoints for the StyGig project."""
    
    def __init__(self, region: str = 'us-east-1', prefix: str = 'stygig'):
        """
        Initialize endpoint manager.
        
        Args:
            region: AWS region
            prefix: Endpoint name prefix to filter (default: 'stygig')
        """
        self.region = region
        self.prefix = prefix.lower()
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)
    
    def list_endpoints(self, status_filter: Optional[str] = None) -> List[Dict]:
        """
        List all StyGig endpoints.
        
        Args:
            status_filter: Filter by status (InService, Creating, Failed, etc.)
        
        Returns:
            List of endpoint dictionaries
        """
        try:
            logger.info(f"Listing endpoints in region: {self.region}")
            
            # List all endpoints
            paginator = self.sagemaker_client.get_paginator('list_endpoints')
            page_iterator = paginator.paginate(
                SortBy='CreationTime',
                SortOrder='Descending'
            )
            
            endpoints = []
            for page in page_iterator:
                for endpoint in page['Endpoints']:
                    # Filter by prefix
                    if self.prefix and not endpoint['EndpointName'].lower().startswith(self.prefix):
                        continue
                    
                    # Filter by status if provided
                    if status_filter and endpoint['EndpointStatus'] != status_filter:
                        continue
                    
                    endpoints.append(endpoint)
            
            logger.info(f"Found {len(endpoints)} endpoint(s) matching prefix '{self.prefix}'")
            return endpoints
            
        except ClientError as e:
            logger.error(f"Failed to list endpoints: {e}")
            return []
    
    def get_endpoint_details(self, endpoint_name: str) -> Optional[Dict]:
        """
        Get detailed information about a specific endpoint.
        
        Args:
            endpoint_name: Name of the endpoint
        
        Returns:
            Dictionary with endpoint details or None if not found
        """
        try:
            logger.info(f"Fetching details for endpoint: {endpoint_name}")
            
            response = self.sagemaker_client.describe_endpoint(
                EndpointName=endpoint_name
            )
            
            # Get endpoint config details
            config_name = response['EndpointConfigName']
            config_response = self.sagemaker_client.describe_endpoint_config(
                EndpointConfigName=config_name
            )
            
            details = {
                'endpoint_name': response['EndpointName'],
                'endpoint_arn': response['EndpointArn'],
                'status': response['EndpointStatus'],
                'creation_time': response['CreationTime'],
                'last_modified_time': response['LastModifiedTime'],
                'config_name': config_name,
                'production_variants': config_response['ProductionVariants']
            }
            
            # Add failure reason if exists
            if 'FailureReason' in response:
                details['failure_reason'] = response['FailureReason']
            
            return details
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ValidationException':
                logger.error(f"Endpoint not found: {endpoint_name}")
            else:
                logger.error(f"Failed to get endpoint details: {e}")
            return None
    
    def delete_endpoint(self, endpoint_name: str, delete_config: bool = True) -> bool:
        """
        Delete a specific endpoint.
        
        Args:
            endpoint_name: Name of the endpoint to delete
            delete_config: Also delete the endpoint configuration
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get endpoint config name before deletion
            config_name = None
            if delete_config:
                try:
                    response = self.sagemaker_client.describe_endpoint(
                        EndpointName=endpoint_name
                    )
                    config_name = response.get('EndpointConfigName')
                except ClientError:
                    pass
            
            # Delete endpoint
            logger.info(f"Deleting endpoint: {endpoint_name}")
            self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            logger.info(f"✅ Endpoint '{endpoint_name}' deleted successfully")
            
            # Delete endpoint configuration if requested
            if delete_config and config_name:
                try:
                    logger.info(f"Deleting endpoint config: {config_name}")
                    self.sagemaker_client.delete_endpoint_config(
                        EndpointConfigName=config_name
                    )
                    logger.info(f"✅ Endpoint config '{config_name}' deleted successfully")
                except ClientError as e:
                    logger.warning(f"Could not delete endpoint config: {e}")
            
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ValidationException':
                logger.error(f"Endpoint not found: {endpoint_name}")
            else:
                logger.error(f"Failed to delete endpoint: {e}")
            return False
    
    def delete_all_endpoints(self, confirm: bool = False) -> int:
        """
        Delete all StyGig endpoints.
        
        Args:
            confirm: If True, skip confirmation prompt
        
        Returns:
            Number of endpoints deleted
        """
        endpoints = self.list_endpoints()
        
        if not endpoints:
            logger.info("No endpoints found to delete")
            return 0
        
        print("\n" + "="*80)
        print("⚠️  WARNING: About to delete the following endpoints:")
        print("="*80)
        
        for i, endpoint in enumerate(endpoints, 1):
            print(f"{i}. {endpoint['EndpointName']}")
            print(f"   Status: {endpoint['EndpointStatus']}")
            print(f"   Created: {endpoint['CreationTime']}")
            print()
        
        print(f"Total: {len(endpoints)} endpoint(s)")
        print("="*80)
        
        # Confirm deletion
        if not confirm:
            response = input("\nAre you sure you want to delete ALL these endpoints? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("Deletion cancelled")
                return 0
        
        # Delete endpoints
        deleted_count = 0
        for endpoint in endpoints:
            endpoint_name = endpoint['EndpointName']
            if self.delete_endpoint(endpoint_name, delete_config=True):
                deleted_count += 1
        
        logger.info(f"\n✅ Deleted {deleted_count}/{len(endpoints)} endpoint(s)")
        return deleted_count
    
    def print_endpoints_table(self, endpoints: List[Dict]):
        """Print endpoints in a formatted table."""
        if not endpoints:
            print("\n📭 No endpoints found")
            return
        
        print("\n" + "="*120)
        print(f"{'Endpoint Name':<45} {'Status':<15} {'Instance Type':<15} {'Created':<25}")
        print("="*120)
        
        for endpoint in endpoints:
            name = endpoint['EndpointName']
            status = endpoint['EndpointStatus']
            created = endpoint['CreationTime'].strftime('%Y-%m-%d %H:%M:%S')
            
            # Get instance type from endpoint config
            try:
                config_name = endpoint.get('EndpointConfigName', '')
                if config_name:
                    config = self.sagemaker_client.describe_endpoint_config(
                        EndpointConfigName=config_name
                    )
                    variants = config.get('ProductionVariants', [])
                    instance_type = variants[0]['InstanceType'] if variants else 'Unknown'
                else:
                    instance_type = 'Unknown'
            except Exception:
                instance_type = 'Unknown'
            
            # Color code status
            status_color = ""
            if status == 'InService':
                status_color = '\033[0;32m'  # Green
            elif status == 'Failed':
                status_color = '\033[0;31m'  # Red
            elif status in ['Creating', 'Updating']:
                status_color = '\033[1;33m'  # Yellow
            
            print(f"{name:<45} {status_color}{status:<15}\033[0m {instance_type:<15} {created:<25}")
        
        print("="*120)
        print(f"\nTotal: {len(endpoints)} endpoint(s)\n")


def main():
    """Main entry point for the endpoint manager."""
    parser = argparse.ArgumentParser(
        description='StyGig Endpoint Management Utility',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all StyGig endpoints
  python scripts/manage_endpoints.py list

  # List only active endpoints
  python scripts/manage_endpoints.py list --status InService

  # Get details about specific endpoint
  python scripts/manage_endpoints.py info --endpoint-name stygig-endpoint-20251103-062336

  # Delete a specific endpoint
  python scripts/manage_endpoints.py delete --endpoint-name stygig-endpoint-20251103-062336

  # Delete all StyGig endpoints
  python scripts/manage_endpoints.py delete-all
        """
    )
    
    parser.add_argument(
        'action',
        choices=['list', 'info', 'delete', 'delete-all'],
        help='Action to perform'
    )
    
    parser.add_argument(
        '--endpoint-name',
        type=str,
        help='Specific endpoint name (required for info/delete)'
    )
    
    parser.add_argument(
        '--region',
        type=str,
        default='us-east-1',
        help='AWS region (default: us-east-1)'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='stygig',
        help='Endpoint name prefix to filter (default: stygig)'
    )
    
    parser.add_argument(
        '--status',
        type=str,
        choices=['InService', 'Creating', 'Updating', 'Failed', 'Deleting'],
        help='Filter by endpoint status'
    )
    
    parser.add_argument(
        '--yes',
        action='store_true',
        help='Skip confirmation prompts'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output in JSON format'
    )
    
    args = parser.parse_args()
    
    # Validate required arguments
    if args.action in ['info', 'delete'] and not args.endpoint_name:
        parser.error(f"--endpoint-name is required for '{args.action}' action")
    
    # Initialize manager
    manager = EndpointManager(region=args.region, prefix=args.prefix)
    
    # Execute action
    try:
        if args.action == 'list':
            endpoints = manager.list_endpoints(status_filter=args.status)
            
            if args.json:
                # Convert datetime objects to strings for JSON serialization
                for endpoint in endpoints:
                    endpoint['CreationTime'] = endpoint['CreationTime'].isoformat()
                    endpoint['LastModifiedTime'] = endpoint['LastModifiedTime'].isoformat()
                print(json.dumps(endpoints, indent=2))
            else:
                manager.print_endpoints_table(endpoints)
        
        elif args.action == 'info':
            details = manager.get_endpoint_details(args.endpoint_name)
            
            if details:
                if args.json:
                    # Convert datetime objects to strings
                    details['creation_time'] = details['creation_time'].isoformat()
                    details['last_modified_time'] = details['last_modified_time'].isoformat()
                    print(json.dumps(details, indent=2))
                else:
                    print("\n" + "="*80)
                    print(f"Endpoint Details: {args.endpoint_name}")
                    print("="*80)
                    print(f"Status: {details['status']}")
                    print(f"ARN: {details['endpoint_arn']}")
                    print(f"Created: {details['creation_time']}")
                    print(f"Last Modified: {details['last_modified_time']}")
                    print(f"Config Name: {details['config_name']}")
                    
                    if 'failure_reason' in details:
                        print(f"Failure Reason: {details['failure_reason']}")
                    
                    print("\nProduction Variants:")
                    for variant in details['production_variants']:
                        print(f"  - Instance Type: {variant['InstanceType']}")
                        print(f"    Instance Count: {variant['InitialInstanceCount']}")
                        print(f"    Variant Name: {variant['VariantName']}")
                    
                    print("="*80 + "\n")
            else:
                sys.exit(1)
        
        elif args.action == 'delete':
            success = manager.delete_endpoint(args.endpoint_name, delete_config=True)
            sys.exit(0 if success else 1)
        
        elif args.action == 'delete-all':
            deleted_count = manager.delete_all_endpoints(confirm=args.yes)
            
            if deleted_count > 0:
                print(f"\n💰 Cost Savings: Deleting {deleted_count} endpoint(s) will save costs!")
                print("   Endpoints are billed per hour while running.")
            
            sys.exit(0)
    
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

"""
Command Line Interface for Glaucoma Corpus Builder
Interactive CLI for managing glaucoma research corpus building
"""

import os
import sys
import argparse
import logging
from typing import List, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corpus import GlaucomaCorpusBuilder, CorpusBuilderConfig, APIAuthManager
from utils.logger import get_logger

logger = get_logger(__name__)

def setup_logging(verbose: bool = False):
    """Set up logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('glaucoma_corpus_builder.log')
        ]
    )

def create_config_from_args(args) -> CorpusBuilderConfig:
    """Create corpus builder configuration from command line arguments"""
    return CorpusBuilderConfig(
        enabled_sources=args.sources,
        max_results_per_source=args.max_results,
        search_years_back=args.years,
        min_relevance_score=args.min_relevance,
        min_quality_score=args.min_quality,
        remove_duplicates=not args.keep_duplicates,
        extract_entities=not args.no_entities,
        calculate_statistics=not args.no_stats,
        output_directory=args.output_dir,
        export_formats=args.formats,
        include_metadata=not args.no_metadata,
        max_concurrent_requests=args.concurrent,
        request_delay=args.delay,
        timeout_seconds=args.timeout
    )

def cmd_build_corpus(args):
    """Build glaucoma research corpus"""
    logger.info("🏗️ Starting Glaucoma Corpus Builder")
    
    try:
        # Create configuration
        config = create_config_from_args(args)
        
        # Initialize builder
        with GlaucomaCorpusBuilder(config) as builder:
            
            # Build corpus
            result = builder.build_corpus(
                custom_search_terms=args.custom_terms,
                subtopics=args.subtopics
            )
            
            # Display results
            if result.success:
                print(f"\n✅ Corpus building completed successfully!")
                print(f"📊 Total documents: {result.total_documents}")
                print(f"🔄 Processed documents: {result.processed_documents}")
                print(f"✨ Final corpus: {result.filtered_documents} high-quality documents")
                print(f"⏱️ Processing time: {result.processing_time:.1f} seconds")
                print(f"📁 Output files: {len(result.output_files)}")
                
                for file_path in result.output_files:
                    print(f"   - {file_path}")
                
                if result.recommendations:
                    print(f"\n💡 Recommendations:")
                    for rec in result.recommendations:
                        print(f"   - {rec}")
                        
            else:
                print(f"\n❌ Corpus building failed")
                error_summary = result.error_summary
                if error_summary.get('total_errors', 0) > 0:
                    print(f"🚨 Errors encountered: {error_summary['total_errors']}")
                    for category, count in error_summary.get('by_category', {}).items():
                        print(f"   - {category}: {count}")
                        
    except Exception as e:
        logger.error(f"Failed to build corpus: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)

def cmd_setup_credentials(args):
    """Set up API credentials"""
    print("🔐 Setting up API credentials for Glaucoma Corpus Builder")
    
    auth_manager = APIAuthManager()
    available_apis = auth_manager.list_available_apis()
    
    print(f"\n📋 Available APIs:")
    for api in available_apis:
        status = "✅ Configured" if api['available'] else "❌ Not configured"
        print(f"   {api['name']}: {status}")
        print(f"      Description: {api['description']}")
        if not api['available'] and 'env_key' in api:
            print(f"      Environment variable: {api['env_key']}")
        print()
    
    # Interactive credential setup
    if args.interactive:
        for api in available_apis:
            if not api['available']:
                print(f"\n🔧 Configure {api['name']}?")
                response = input("Enter API key (or press Enter to skip): ").strip()
                if response:
                    try:
                        auth_manager.add_credential(
                            api_name=api['name'],
                            api_key=response,
                            description=api.get('description', '')
                        )
                        print(f"✅ Added credentials for {api['name']}")
                    except Exception as e:
                        print(f"❌ Failed to add credentials: {e}")

def cmd_test_apis(args):
    """Test API connections"""
    print("🧪 Testing API connections...")
    
    auth_manager = APIAuthManager()
    available_apis = auth_manager.list_available_apis()
    
    print(f"\n📊 API Test Results:")
    for api in available_apis:
        if api['available']:
            print(f"🔍 Testing {api['name']}...")
            try:
                success = auth_manager.test_credential(api['name'])
                status = "✅ PASS" if success else "❌ FAIL"
                print(f"   {api['name']}: {status}")
            except Exception as e:
                print(f"   {api['name']}: ❌ ERROR - {e}")
        else:
            print(f"   {api['name']}: ⚠️ NOT CONFIGURED")

def cmd_health_check(args):
    """Perform health check"""
    print("🩺 Performing health check...")
    
    try:
        with GlaucomaCorpusBuilder() as builder:
            health_status = builder.get_health_status()
            
            overall_health = health_status['overall_health']
            status_icon = "✅" if overall_health else "❌"
            print(f"\n{status_icon} Overall Health: {'HEALTHY' if overall_health else 'UNHEALTHY'}")
            
            print(f"\n📊 Client Status:")
            for client_name, status in health_status['client_status'].items():
                client_healthy = status.get('healthy', False)
                client_icon = "✅" if client_healthy else "❌"
                print(f"   {client_icon} {client_name}: {'HEALTHY' if client_healthy else 'UNHEALTHY'}")
                
                if 'error' in status:
                    print(f"      Error: {status['error']}")
                else:
                    print(f"      Requests: {status.get('request_count', 0)}")
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Glaucoma Corpus Builder - Comprehensive medical research data extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build basic corpus
  python glaucoma_corpus_cli.py build

  # Build with custom parameters
  python glaucoma_corpus_cli.py build --max-results 500 --years 3 --sources pubmed clinical_trials

  # Setup API credentials interactively
  python glaucoma_corpus_cli.py setup --interactive

  # Test all API connections
  python glaucoma_corpus_cli.py test

  # Check system health
  python glaucoma_corpus_cli.py health
        """
    )
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build glaucoma research corpus')
    build_parser.add_argument('--sources', nargs='*', 
                             default=['pubmed', 'clinical_trials', 'europepmc'],
                             choices=['pubmed', 'clinical_trials', 'openfda', 'who_gho', 'europepmc'],
                             help='Data sources to use')
    build_parser.add_argument('--max-results', type=int, default=1000,
                             help='Maximum results per source')
    build_parser.add_argument('--years', type=int, default=5,
                             help='Years back to search')
    build_parser.add_argument('--min-relevance', type=float, default=15.0,
                             help='Minimum relevance score')
    build_parser.add_argument('--min-quality', type=float, default=0.5,
                             help='Minimum quality score')
    build_parser.add_argument('--output-dir', default='./data/glaucoma_corpus',
                             help='Output directory')
    build_parser.add_argument('--formats', nargs='*', default=['json'],
                             choices=['json'],
                             help='Export formats')
    build_parser.add_argument('--concurrent', type=int, default=5,
                             help='Maximum concurrent requests')
    build_parser.add_argument('--delay', type=float, default=0.5,
                             help='Delay between requests (seconds)')
    build_parser.add_argument('--timeout', type=int, default=30,
                             help='Request timeout (seconds)')
    build_parser.add_argument('--custom-terms', nargs='*',
                             help='Additional search terms')
    build_parser.add_argument('--subtopics', nargs='*',
                             choices=['primary', 'secondary', 'treatments', 'diagnostics'],
                             help='Glaucoma subtopics to focus on')
    build_parser.add_argument('--keep-duplicates', action='store_true',
                             help='Keep duplicate documents')
    build_parser.add_argument('--no-entities', action='store_true',
                             help='Skip entity extraction')
    build_parser.add_argument('--no-stats', action='store_true',
                             help='Skip statistics generation')
    build_parser.add_argument('--no-metadata', action='store_true',
                             help='Skip metadata in export')
    build_parser.set_defaults(func=cmd_build_corpus)
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Set up API credentials')
    setup_parser.add_argument('--interactive', '-i', action='store_true',
                             help='Interactive credential setup')
    setup_parser.set_defaults(func=cmd_setup_credentials)
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test API connections')
    test_parser.set_defaults(func=cmd_test_apis)
    
    # Health command
    health_parser = subparsers.add_parser('health', help='Check system health')
    health_parser.set_defaults(func=cmd_health_check)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Execute command
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
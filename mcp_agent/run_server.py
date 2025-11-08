#!/usr/bin/env python3
"""
Standalone script to run the Fall Detection MCP Server.
This script can be run directly without package installation.
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_server import FallDetectionMCPServer, main

if __name__ == "__main__":
    main()


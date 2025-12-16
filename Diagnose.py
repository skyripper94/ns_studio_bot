#!/usr/bin/env python3
"""
Полная диагностика системы удаления текста
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

def check_system():
    """Check all system components"""
    
    print("=" * 60)
    print("🔍 SYSTEM DIAGNOSTICS")
    print("=" * 60)
    
    errors = []
    warnings = []
    
    # 1. Check environment variables
    print("\n📋 CHECKING ENVIRONMENT VARIABLES:")
    
    tokens = {
        'REPLICATE_API_TOKEN': os.getenv('REPLICATE_API_TOKEN'),
        'GOOGLE_VISION_API_KEY': os.getenv('GOOGLE_VISION_API_KEY'),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'TELEGRAM_TOKEN': os.getenv('TELEGRAM_TOKEN')
    }
    
    for name, value in tokens.items():
        if value:
            print(f"  ✅ {name}: {value[:20]}...")
        else:
            print(f"  ❌ {name}: NOT SET")
            if name == 'REPLICATE_API_TOKEN':
                errors.append(f"{name} is CRITICAL for text removal!")
            else:
                warnings.append(f"{name} is missing")
    
    # 2. Check Python packages
    print("\n📦 CHECKING PYTHON PACKAGES:")
    
    packages = [
        'cv2',
        'PIL',
        'numpy',
        'replicate',
        'openai',
        'requests',
        'telegram'
    ]
    
    for package in packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}: NOT INSTALLED")
            errors.append(f"Package {package} not installed")
    
    # 3. Check Replicate connection
    if tokens['REPLICATE_API_TOKEN']:
        print("\n🌐 CHECKING REPLICATE CONNECTION:")
        try:
            import replicate
            client = replicate.Client(api_token=tokens['REPLICATE_API_TOKEN'])
            
            # Try to get model info
            model = client.models.get("black-forest-labs/flux-kontext-pro")
            print(f"  ✅ Connected to Replicate")
            print(f"  ✅ FLUX Kontext Pro available")
        except Exception as e:
            print(f"  ❌ Replicate connection failed: {e}")
            errors.append("Cannot connect to Replicate API")
    
    # 4. Check font file
    print("\n🔤 CHECKING FONT FILE:")
    font_path = '/app/fonts/WaffleSoft.otf'
    if os.path.exists(font_path):
        print(f"  ✅ Font found: {font_path}")
    else:
        print(f"  ⚠️ Font not found: {font_path}")
        warnings.append("Font file missing (will use fallback)")
    
    # 5. Check temp directories
    print("\n📁 CHECKING DIRECTORIES:")
    dirs = ['/tmp', '/tmp/bot_images']
    for dir_path in dirs:
        if os.path.exists(dir_path) and os.access(dir_path, os.W_OK):
            print(f"  ✅ {dir_path}: writable")
        else:
            print(f"  ❌ {dir_path}: not writable")
            warnings.append(f"Directory {dir_path} not writable")
    
    # 6. Check Google Vision API (if key present)
    if tokens['GOOGLE_VISION_API_KEY']:
        print("\n🔍 CHECKING GOOGLE VISION API:")
        try:
            import requests
            url = f"https://vision.googleapis.com/v1/images:annotate?key={tokens['GOOGLE_VISION_API_KEY']}"
            # Test with empty request
            response = requests.post(url, json={"requests": []}, timeout=5)
            if response.status_code == 200:
                print("  ✅ Google Vision API accessible")
            else:
                print(f"  ⚠️ Google Vision API returned: {response.status_code}")
                warnings.append("Google Vision API may not be configured correctly")
        except Exception as e:
            print(f"  ❌ Cannot connect to Google Vision: {e}")
            warnings.append("Google Vision API connection failed")
    
    # 7. Check OpenAI API (if key present)
    if tokens['OPENAI_API_KEY']:
        print("\n🤖 CHECKING OPENAI API:")
        try:
            import openai
            openai.api_key = tokens['OPENAI_API_KEY']
            # Test with models list
            models = openai.Model.list()
            print("  ✅ OpenAI API accessible")
        except Exception as e:
            print(f"  ⚠️ OpenAI API issue: {e}")
            warnings.append("OpenAI API may not be configured correctly")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DIAGNOSTIC SUMMARY:")
    print("=" * 60)
    
    if errors:
        print("\n🚨 CRITICAL ERRORS (must fix):")
        for error in errors:
            print(f"  ❌ {error}")
    
    if warnings:
        print("\n⚠️ WARNINGS (should fix):")
        for warning in warnings:
            print(f"  ⚠️ {warning}")
    
    if not errors:
        print("\n✅ SYSTEM READY!")
        print("All critical components are working.")
    else:
        print("\n❌ SYSTEM NOT READY!")
        print("Fix critical errors before running the bot.")
        
    print("\n" + "=" * 60)
    
    # Specific FLUX advice
    if not tokens['REPLICATE_API_TOKEN']:
        print("\n🔴 CRITICAL FOR TEXT REMOVAL:")
        print("REPLICATE_API_TOKEN is REQUIRED for FLUX to work!")
        print("Without it, text will NOT be removed properly.")
        print("\nGet your token from:")
        print("https://replicate.com/account/api-tokens")
        print("\nAdd to .env file:")
        print("REPLICATE_API_TOKEN=r8_xxxxxxxxxxxxx")
    
    return len(errors) == 0


if __name__ == "__main__":
    success = check_system()
    sys.exit(0 if success else 1)

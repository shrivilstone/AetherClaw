import time
import os
import sys

# Ensure relative imports resolve if run independently
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.pipeline import RAGPipeline

try:
    from wa_automate_python import WhatsApp
    WA_AVAILABLE = True
except ImportError:
    WA_AVAILABLE = False


def start_whatsapp_agent():
    print("⚡ Initializing AetherClaw Brain...")
    pipeline = RAGPipeline()
    
    print("\nStarting WhatsApp Web Automation Bridge...")
    print("------------------------------------------")
    print("REQUIRED: Please manually scan the QR Code that appears in the browser window.")
    print("------------------------------------------\n")
    
    wa = WhatsApp()
    print("WhatsApp Connected! AetherClaw is now listening for messages...")
    
    while True:
        try:
            # Poll for new unread messages in chats
            unread_messages = wa.get_unread_messages()
            for message in unread_messages:
                sender_id = message['sender']['id']
                content = message['body']
                print(f"[WhatsApp] Received from {sender_id}: {content}")
                
                # Query AetherClaw Pipeline
                response = pipeline.ask(content)
                print(f"[AetherClaw] Replying: {response}")
                
                # Send back to WhatsApp
                wa.send_message(sender_id, response)
                
        except Exception as e:
            print(f"[Error] Failed processing message: {e}")
            
        time.sleep(3)

if __name__ == "__main__":
    if WA_AVAILABLE:
        start_whatsapp_agent()
    else:
        print("CRITICAL: WhatsApp Bridge Dependencies Missing.")
        print("Please run: pip install wa-automate-python")
        print("You will also need Google Chrome installed to scan the QR code.")

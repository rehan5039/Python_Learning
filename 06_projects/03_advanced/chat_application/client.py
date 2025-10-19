import socket
import threading
import json
import sys
from datetime import datetime

class ChatClient:
    def __init__(self, host='localhost', port=12345):
        self.host = host
        self.port = port
        self.client_socket = None
        self.username = None
        self.connected = False
        self.current_room = 'general'
    
    def connect_to_server(self):
        """Connect to the chat server"""
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((self.host, self.port))
            self.connected = True
            print(f"Connected to server {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            return False
    
    def send_message(self, message_data):
        """Send message to server"""
        try:
            message_json = json.dumps(message_data)
            message_bytes = message_json.encode('utf-8')
            message_length = len(message_bytes)
            
            # Send length first, then the message
            self.client_socket.send(message_length.to_bytes(4, byteorder='big'))
            self.client_socket.send(message_bytes)
            return True
        except Exception as e:
            print(f"Error sending message: {e}")
            return False
    
    def receive_message(self):
        """Receive message from server"""
        try:
            # First, receive the length of the message
            length_bytes = self.client_socket.recv(4)
            if not length_bytes:
                return None
            
            message_length = int.from_bytes(length_bytes, byteorder='big')
            
            # Then receive the actual message
            message_data = b''
            while len(message_data) < message_length:
                chunk = self.client_socket.recv(message_length - len(message_data))
                if not chunk:
                    return None
                message_data += chunk
            
            return json.loads(message_data.decode('utf-8'))
        except Exception as e:
            print(f"Error receiving message: {e}")
            return None
    
    def login(self, username):
        """Login to the chat server"""
        if not self.connected:
            print("Not connected to server")
            return False
        
        # Send login request
        login_data = {
            'action': 'connect',
            'username': username
        }
        
        if not self.send_message(login_data):
            return False
        
        # Wait for response
        response = self.receive_message()
        if not response:
            print("Failed to receive login response")
            return False
        
        if response.get('action') == 'connected':
            self.username = username
            self.current_room = response.get('room', 'general')
            print(response.get('message', 'Connected successfully'))
            return True
        else:
            print(response.get('message', 'Login failed'))
            return False
    
    def start_receiving(self):
        """Start thread to receive messages"""
        receive_thread = threading.Thread(target=self.receive_messages)
        receive_thread.daemon = True
        receive_thread.start()
    
    def receive_messages(self):
        """Continuously receive messages from server"""
        while self.connected:
            try:
                message_data = self.receive_message()
                if not message_data:
                    break
                
                self.handle_server_message(message_data)
                
            except Exception as e:
                if self.connected:
                    print(f"Error receiving message: {e}")
                break
        
        if self.connected:
            print("Disconnected from server")
            self.connected = False
    
    def handle_server_message(self, message_data):
        """Handle messages from server"""
        action = message_data.get('action')
        
        if action == 'chat':
            timestamp = message_data.get('timestamp', '')
            username = message_data.get('username', 'Unknown')
            message = message_data.get('message', '')
            room = message_data.get('room', '')
            
            if room == self.current_room:
                print(f"[{timestamp}] {username}: {message}")
        
        elif action == 'private_message':
            timestamp = message_data.get('timestamp', '')
            sender = message_data.get('from', 'Unknown')
            message = message_data.get('message', '')
            print(f"[{timestamp}] [PRIVATE] {sender}: {message}")
        
        elif action == 'user_joined':
            timestamp = message_data.get('timestamp', '')
            message = message_data.get('message', '')
            print(f"[{timestamp}] *** {message} ***")
        
        elif action == 'user_left':
            timestamp = message_data.get('timestamp', '')
            message = message_data.get('message', '')
            print(f"[{timestamp}] *** {message} ***")
        
        elif action == 'user_joined_room':
            timestamp = message_data.get('timestamp', '')
            username = message_data.get('username', 'Unknown')
            room = message_data.get('room', '')
            message = message_data.get('message', '')
            if room == self.current_room:
                print(f"[{timestamp}] *** {message} ***")
        
        elif action == 'user_left_room':
            timestamp = message_data.get('timestamp', '')
            username = message_data.get('username', 'Unknown')
            room = message_data.get('room', '')
            message = message_data.get('message', '')
            if room == self.current_room:
                print(f"[{timestamp}] *** {message} ***")
        
        elif action == 'room_joined':
            self.current_room = message_data.get('room', 'general')
            message = message_data.get('message', '')
            print(f"*** {message} ***")
        
        elif action == 'room_left':
            self.current_room = message_data.get('room', 'general')
            message = message_data.get('message', '')
            print(f"*** {message} ***")
        
        elif action == 'rooms_list':
            rooms = message_data.get('rooms', [])
            print("\nAvailable rooms:")
            for room in rooms:
                print(f"  - {room}")
            print()
        
        elif action == 'users_list':
            room = message_data.get('room', '')
            users = message_data.get('users', [])
            print(f"\nUsers in {room}:")
            for user in users:
                print(f"  - {user}")
            print()
        
        elif action == 'message_sent':
            # Confirmation that message was sent
            pass
        
        elif action == 'private_message_sent':
            target = message_data.get('to', 'Unknown')
            message = message_data.get('message', '')
            print(f"*** {message} to {target} ***")
        
        elif action == 'error':
            error_message = message_data.get('message', 'Unknown error')
            print(f"*** ERROR: {error_message} ***")
        
        else:
            print(f"Unknown message type: {action}")
    
    def send_chat_message(self, message):
        """Send a chat message"""
        if not self.connected or not self.username:
            print("Not connected to chat")
            return
        
        message_data = {
            'action': 'chat',
            'message': message
        }
        
        self.send_message(message_data)
    
    def join_room(self, room_name):
        """Join a chat room"""
        if not self.connected or not self.username:
            print("Not connected to chat")
            return
        
        message_data = {
            'action': 'join_room',
            'room': room_name
        }
        
        self.send_message(message_data)
    
    def leave_room(self):
        """Leave current room and join general"""
        if not self.connected or not self.username:
            print("Not connected to chat")
            return
        
        message_data = {
            'action': 'leave_room'
        }
        
        self.send_message(message_data)
    
    def list_rooms(self):
        """List all available rooms"""
        if not self.connected or not self.username:
            print("Not connected to chat")
            return
        
        message_data = {
            'action': 'list_rooms'
        }
        
        self.send_message(message_data)
    
    def list_users(self):
        """List users in current room"""
        if not self.connected or not self.username:
            print("Not connected to chat")
            return
        
        message_data = {
            'action': 'list_users'
        }
        
        self.send_message(message_data)
    
    def send_private_message(self, target_user, message):
        """Send a private message"""
        if not self.connected or not self.username:
            print("Not connected to chat")
            return
        
        message_data = {
            'action': 'private_message',
            'target': target_user,
            'message': message
        }
        
        self.send_message(message_data)
    
    def disconnect(self):
        """Disconnect from server"""
        if self.connected:
            try:
                disconnect_data = {
                    'action': 'disconnect'
                }
                self.send_message(disconnect_data)
            except:
                pass
            
            self.connected = False
            if self.client_socket:
                self.client_socket.close()
            print("Disconnected from server")
    
    def show_help(self):
        """Show available commands"""
        print("\n=== Chat Commands ===")
        print("/help - Show this help message")
        print("/join <room> - Join a chat room")
        print("/leave - Leave current room")
        print("/rooms - List available rooms")
        print("/users - List users in current room")
        print("/msg <user> <message> - Send private message")
        print("/quit - Quit the chat application")
        print("Any other text - Send message to current room")
        print()

def main():
    """Main function to run the chat client"""
    print("=== Python Chat Client ===")
    
    # Get server details
    host = input("Enter server host (default: localhost): ").strip() or 'localhost'
    
    try:
        port = int(input("Enter server port (default: 12345): ").strip() or '12345')
    except ValueError:
        port = 12345
    
    # Create client
    client = ChatClient(host, port)
    
    # Connect to server
    if not client.connect_to_server():
        return
    
    # Get username
    username = input("Enter your username: ").strip()
    if not username:
        print("Username required!")
        return
    
    # Login
    if not client.login(username):
        return
    
    # Start receiving messages
    client.start_receiving()
    
    # Show help
    client.show_help()
    
    # Main chat loop
    try:
        while client.connected:
            try:
                user_input = input().strip()
                
                if not user_input:
                    continue
                
                if user_input.startswith('/'):
                    # Process commands
                    parts = user_input.split(' ', 3)
                    command = parts[0].lower()
                    
                    if command == '/help':
                        client.show_help()
                    
                    elif command == '/join':
                        if len(parts) >= 2:
                            client.join_room(parts[1])
                        else:
                            print("Usage: /join <room_name>")
                    
                    elif command == '/leave':
                        client.leave_room()
                    
                    elif command == '/rooms':
                        client.list_rooms()
                    
                    elif command == '/users':
                        client.list_users()
                    
                    elif command == '/msg':
                        if len(parts) >= 3:
                            target_user = parts[1]
                            message = ' '.join(parts[2:])
                            client.send_private_message(target_user, message)
                        else:
                            print("Usage: /msg <username> <message>")
                    
                    elif command == '/quit':
                        break
                    
                    else:
                        print(f"Unknown command: {command}. Type /help for available commands.")
                
                else:
                    # Send as chat message
                    client.send_chat_message(user_input)
            
            except KeyboardInterrupt:
                break
            except EOFError:
                break
    
    except Exception as e:
        print(f"Client error: {e}")
    finally:
        client.disconnect()
        print("Chat client terminated.")

if __name__ == "__main__":
    main()
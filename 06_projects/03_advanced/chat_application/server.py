import socket
import threading
import json
import time
from datetime import datetime

class ChatServer:
    def __init__(self, host='localhost', port=12345):
        self.host = host
        self.port = port
        self.clients = {}  # {client_socket: username}
        self.rooms = {'general': []}  # {room_name: [client_sockets]}
        self.server_socket = None
        self.running = False
    
    def start_server(self):
        """Start the chat server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True
            
            print(f"Chat server started on {self.host}:{self.port}")
            print("Waiting for connections...")
            
            while self.running:
                try:
                    client_socket, address = self.server_socket.accept()
                    print(f"Connection from {address}")
                    
                    # Start a new thread to handle the client
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                except socket.error:
                    if self.running:
                        print("Socket error occurred")
                    break
                    
        except Exception as e:
            print(f"Error starting server: {e}")
        finally:
            self.stop_server()
    
    def stop_server(self):
        """Stop the chat server"""
        print("Stopping server...")
        self.running = False
        
        # Close all client connections
        for client_socket in list(self.clients.keys()):
            try:
                client_socket.close()
            except:
                pass
        
        # Close server socket
        if self.server_socket:
            self.server_socket.close()
        
        print("Server stopped.")
    
    def handle_client(self, client_socket, address):
        """Handle individual client connections"""
        username = None
        current_room = 'general'
        
        try:
            while self.running:
                # Receive message from client
                message_data = self.receive_message(client_socket)
                if not message_data:
                    break
                
                # Process the message
                response = self.process_message(
                    client_socket, 
                    message_data, 
                    username, 
                    current_room
                )
                
                # Handle response
                if response == "DISCONNECT":
                    break
                elif isinstance(response, dict) and 'username' in response:
                    username = response['username']
                elif isinstance(response, dict) and 'room' in response:
                    current_room = response['room']
                    
        except Exception as e:
            print(f"Error handling client {address}: {e}")
        finally:
            # Clean up client connection
            self.disconnect_client(client_socket, username)
    
    def receive_message(self, client_socket):
        """Receive message from client"""
        try:
            # First, receive the length of the message
            length_bytes = client_socket.recv(4)
            if not length_bytes:
                return None
            
            message_length = int.from_bytes(length_bytes, byteorder='big')
            
            # Then receive the actual message
            message_data = b''
            while len(message_data) < message_length:
                chunk = client_socket.recv(message_length - len(message_data))
                if not chunk:
                    return None
                message_data += chunk
            
            return json.loads(message_data.decode('utf-8'))
        except Exception as e:
            print(f"Error receiving message: {e}")
            return None
    
    def send_message(self, client_socket, message_data):
        """Send message to client"""
        try:
            message_json = json.dumps(message_data)
            message_bytes = message_json.encode('utf-8')
            message_length = len(message_bytes)
            
            # Send length first, then the message
            client_socket.send(message_length.to_bytes(4, byteorder='big'))
            client_socket.send(message_bytes)
            return True
        except Exception as e:
            print(f"Error sending message: {e}")
            return False
    
    def process_message(self, client_socket, message_data, username, current_room):
        """Process incoming messages"""
        try:
            action = message_data.get('action')
            
            if action == 'connect':
                return self.handle_connect(client_socket, message_data)
            
            elif action == 'disconnect':
                return "DISCONNECT"
            
            elif action == 'chat':
                self.handle_chat(client_socket, message_data, username, current_room)
                
            elif action == 'join_room':
                return self.handle_join_room(client_socket, message_data, username)
                
            elif action == 'leave_room':
                return self.handle_leave_room(client_socket, message_data, username)
                
            elif action == 'list_rooms':
                self.handle_list_rooms(client_socket)
                
            elif action == 'list_users':
                self.handle_list_users(client_socket, current_room)
                
            elif action == 'private_message':
                self.handle_private_message(client_socket, message_data, username)
                
            else:
                self.send_message(client_socket, {
                    'action': 'error',
                    'message': f'Unknown action: {action}'
                })
                
        except Exception as e:
            print(f"Error processing message: {e}")
            self.send_message(client_socket, {
                'action': 'error',
                'message': 'Server error occurred'
            })
        
        return None
    
    def handle_connect(self, client_socket, message_data):
        """Handle client connection"""
        username = message_data.get('username')
        if not username:
            self.send_message(client_socket, {
                'action': 'error',
                'message': 'Username required'
            })
            return "DISCONNECT"
        
        # Check if username is already taken
        if username in self.clients.values():
            self.send_message(client_socket, {
                'action': 'error',
                'message': 'Username already taken'
            })
            return "DISCONNECT"
        
        # Add client to server
        self.clients[client_socket] = username
        self.rooms['general'].append(client_socket)
        
        # Send welcome message
        self.send_message(client_socket, {
            'action': 'connected',
            'message': f'Welcome to the chat, {username}!',
            'username': username,
            'room': 'general'
        })
        
        # Announce to other users
        self.broadcast_message({
            'action': 'user_joined',
            'username': username,
            'message': f'{username} joined the chat',
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }, exclude_client=client_socket)
        
        print(f"{username} connected from {client_socket.getpeername()}")
        return {'username': username}
    
    def handle_chat(self, client_socket, message_data, username, current_room):
        """Handle chat messages"""
        if not username:
            self.send_message(client_socket, {
                'action': 'error',
                'message': 'Not connected'
            })
            return
        
        message = message_data.get('message', '')
        if not message:
            return
        
        # Broadcast message to room
        self.broadcast_to_room(current_room, {
            'action': 'chat',
            'username': username,
            'message': message,
            'room': current_room,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }, exclude_client=client_socket)
        
        # Send confirmation to sender
        self.send_message(client_socket, {
            'action': 'message_sent',
            'message': 'Message sent'
        })
    
    def handle_join_room(self, client_socket, message_data, username):
        """Handle joining a room"""
        if not username:
            self.send_message(client_socket, {
                'action': 'error',
                'message': 'Not connected'
            })
            return None
        
        room_name = message_data.get('room', 'general')
        
        # Create room if it doesn't exist
        if room_name not in self.rooms:
            self.rooms[room_name] = []
        
        # Remove from current rooms
        for room, clients in self.rooms.items():
            if client_socket in clients:
                clients.remove(client_socket)
        
        # Add to new room
        self.rooms[room_name].append(client_socket)
        
        # Notify user
        self.send_message(client_socket, {
            'action': 'room_joined',
            'room': room_name,
            'message': f'Joined room: {room_name}'
        })
        
        # Announce to room
        self.broadcast_to_room(room_name, {
            'action': 'user_joined_room',
            'username': username,
            'room': room_name,
            'message': f'{username} joined the room',
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }, exclude_client=client_socket)
        
        print(f"{username} joined room: {room_name}")
        return {'room': room_name}
    
    def handle_leave_room(self, client_socket, message_data, username):
        """Handle leaving a room"""
        if not username:
            self.send_message(client_socket, {
                'action': 'error',
                'message': 'Not connected'
            })
            return None
        
        current_room = None
        # Find current room
        for room, clients in self.rooms.items():
            if client_socket in clients:
                current_room = room
                clients.remove(client_socket)
                break
        
        # Join general room
        if 'general' not in self.rooms:
            self.rooms['general'] = []
        self.rooms['general'].append(client_socket)
        
        # Notify user
        self.send_message(client_socket, {
            'action': 'room_left',
            'room': 'general',
            'message': 'Left room and joined general'
        })
        
        # Announce to previous room
        if current_room:
            self.broadcast_to_room(current_room, {
                'action': 'user_left_room',
                'username': username,
                'room': current_room,
                'message': f'{username} left the room',
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
        
        print(f"{username} left room and joined general")
        return {'room': 'general'}
    
    def handle_list_rooms(self, client_socket):
        """Handle listing all rooms"""
        room_list = list(self.rooms.keys())
        self.send_message(client_socket, {
            'action': 'rooms_list',
            'rooms': room_list
        })
    
    def handle_list_users(self, client_socket, current_room):
        """Handle listing users in current room"""
        users = []
        if current_room in self.rooms:
            users = [self.clients[sock] for sock in self.rooms[current_room] if sock in self.clients]
        
        self.send_message(client_socket, {
            'action': 'users_list',
            'room': current_room,
            'users': users
        })
    
    def handle_private_message(self, client_socket, message_data, username):
        """Handle private messages"""
        if not username:
            self.send_message(client_socket, {
                'action': 'error',
                'message': 'Not connected'
            })
            return
        
        target_user = message_data.get('target')
        message = message_data.get('message', '')
        
        if not target_user or not message:
            self.send_message(client_socket, {
                'action': 'error',
                'message': 'Target user and message required'
            })
            return
        
        # Find target client
        target_socket = None
        for sock, name in self.clients.items():
            if name == target_user:
                target_socket = sock
                break
        
        if not target_socket:
            self.send_message(client_socket, {
                'action': 'error',
                'message': f'User {target_user} not found'
            })
            return
        
        # Send message to target
        self.send_message(target_socket, {
            'action': 'private_message',
            'from': username,
            'message': message,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
        
        # Confirm to sender
        self.send_message(client_socket, {
            'action': 'private_message_sent',
            'to': target_user,
            'message': 'Private message sent'
        })
    
    def broadcast_message(self, message_data, exclude_client=None):
        """Broadcast message to all connected clients"""
        disconnected_clients = []
        
        for client_socket in list(self.clients.keys()):
            if client_socket == exclude_client:
                continue
                
            try:
                self.send_message(client_socket, message_data)
            except:
                disconnected_clients.append(client_socket)
        
        # Clean up disconnected clients
        for client_socket in disconnected_clients:
            self.disconnect_client(client_socket, self.clients.get(client_socket))
    
    def broadcast_to_room(self, room_name, message_data, exclude_client=None):
        """Broadcast message to specific room"""
        if room_name not in self.rooms:
            return
        
        disconnected_clients = []
        
        for client_socket in self.rooms[room_name]:
            if client_socket == exclude_client:
                continue
                
            try:
                self.send_message(client_socket, message_data)
            except:
                disconnected_clients.append(client_socket)
        
        # Clean up disconnected clients
        for client_socket in disconnected_clients:
            if client_socket in self.rooms[room_name]:
                self.rooms[room_name].remove(client_socket)
            if client_socket in self.clients:
                username = self.clients[client_socket]
                del self.clients[client_socket]
    
    def disconnect_client(self, client_socket, username):
        """Handle client disconnection"""
        try:
            # Remove from rooms
            for room, clients in self.rooms.items():
                if client_socket in clients:
                    clients.remove(client_socket)
            
            # Remove from clients
            if client_socket in self.clients:
                del self.clients[client_socket]
            
            # Close socket
            client_socket.close()
            
            if username:
                print(f"{username} disconnected")
                # Announce to other users
                self.broadcast_message({
                    'action': 'user_left',
                    'username': username,
                    'message': f'{username} left the chat',
                    'timestamp': datetime.now().strftime('%H:%M:%S')
                })
        except Exception as e:
            print(f"Error disconnecting client: {e}")

def main():
    """Main function to start the server"""
    server = ChatServer()
    
    try:
        server.start_server()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.stop_server()
    except Exception as e:
        print(f"Server error: {e}")
        server.stop_server()

if __name__ == "__main__":
    main()
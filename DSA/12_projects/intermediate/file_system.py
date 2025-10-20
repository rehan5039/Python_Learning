"""
Intermediate Project: File System Simulator

This project simulates a file system using tree data structures,
demonstrating hierarchical data organization and traversal algorithms.

Concepts covered:
- Tree data structure implementation
- Hierarchical data organization
- Depth-first and breadth-first traversal
- Path resolution and navigation
- Memory management and garbage collection
"""

from typing import List, Dict, Optional, Iterator
from collections import deque
import time


class FileSystemNode:
    """
    Node in the file system tree representing either a file or directory.
    """
    
    def __init__(self, name: str, is_directory: bool = False, content: str = ""):
        self.name = name
        self.is_directory = is_directory
        self.content = content if not is_directory else ""
        self.children = {} if is_directory else None  # Only directories have children
        self.parent = None
        self.size = len(content) if not is_directory else 0
        self.created_time = time.time()
        self.modified_time = self.created_time
    
    def add_child(self, child: 'FileSystemNode') -> None:
        """Add child node to directory."""
        if not self.is_directory:
            raise ValueError("Cannot add children to a file")
        if child.name in self.children:
            raise ValueError(f"Child '{child.name}' already exists")
        
        child.parent = self
        self.children[child.name] = child
        self._update_size()
    
    def remove_child(self, name: str) -> None:
        """Remove child node from directory."""
        if not self.is_directory:
            raise ValueError("Cannot remove children from a file")
        if name not in self.children:
            raise ValueError(f"Child '{name}' not found")
        
        del self.children[name]
        self._update_size()
    
    def get_child(self, name: str) -> Optional['FileSystemNode']:
        """Get child node by name."""
        if not self.is_directory:
            return None
        return self.children.get(name)
    
    def _update_size(self) -> None:
        """Update directory size based on children."""
        if self.is_directory:
            self.size = sum(child.size for child in self.children.values())
            self.modified_time = time.time()
    
    def is_root(self) -> bool:
        """Check if node is root (no parent)."""
        return self.parent is None
    
    def get_path(self) -> str:
        """Get full path of node."""
        if self.is_root():
            return "/"
        
        path_parts = []
        current = self
        while current and not current.is_root():
            path_parts.append(current.name)
            current = current.parent
        
        return "/" + "/".join(reversed(path_parts))


class FileSystem:
    """
    File system simulator using tree data structure.
    """
    
    def __init__(self):
        self.root = FileSystemNode("/", is_directory=True)
        self.current_directory = self.root
    
    def _resolve_path(self, path: str, create_missing: bool = False) -> FileSystemNode:
        """
        Resolve path to node.
        
        Args:
            path: Path string (absolute or relative)
            create_missing: Whether to create missing directories
            
        Returns:
            FileSystemNode at the path
            
        Raises:
            ValueError: If path is invalid or node not found
        """
        if path.startswith("/"):
            # Absolute path
            current = self.root
            path_parts = [part for part in path.split("/") if part]
        else:
            # Relative path
            current = self.current_directory
            path_parts = [part for part in path.split("/") if part]
        
        for part in path_parts:
            if part == ".":
                continue
            elif part == "..":
                if not current.is_root():
                    current = current.parent
            else:
                if not current.is_directory:
                    raise ValueError(f"'{current.name}' is not a directory")
                
                if part not in current.children:
                    if create_missing and current.is_directory:
                        # Create missing directory
                        new_dir = FileSystemNode(part, is_directory=True)
                        current.add_child(new_dir)
                        current = new_dir
                    else:
                        raise ValueError(f"Path '{part}' not found")
                else:
                    current = current.children[part]
        
        return current
    
    def mkdir(self, path: str) -> None:
        """
        Create directory.
        
        Time Complexity: O(d) where d is depth of path
        Space Complexity: O(1)
        """
        # Resolve parent directory
        if "/" in path and not path.endswith("/"):
            parent_path = "/".join(path.split("/")[:-1])
            dir_name = path.split("/")[-1]
            parent = self._resolve_path(parent_path, create_missing=True)
        else:
            parent = self._resolve_path(".", create_missing=True)
            dir_name = path.rstrip("/")
        
        if not parent.is_directory:
            raise ValueError("Parent must be a directory")
        
        if dir_name in parent.children:
            raise ValueError(f"Directory '{dir_name}' already exists")
        
        new_dir = FileSystemNode(dir_name, is_directory=True)
        parent.add_child(new_dir)
    
    def touch(self, path: str, content: str = "") -> None:
        """
        Create or update file.
        
        Time Complexity: O(d) where d is depth of path
        Space Complexity: O(n) where n is content size
        """
        # Resolve parent directory
        if "/" in path:
            parent_path = "/".join(path.split("/")[:-1])
            file_name = path.split("/")[-1]
            parent = self._resolve_path(parent_path, create_missing=True)
        else:
            parent = self.current_directory
            file_name = path
        
        if not parent.is_directory:
            raise ValueError("Parent must be a directory")
        
        if file_name in parent.children:
            # Update existing file
            file_node = parent.children[file_name]
            if file_node.is_directory:
                raise ValueError(f"'{file_name}' is a directory")
            file_node.content = content
            file_node.size = len(content)
            file_node.modified_time = time.time()
        else:
            # Create new file
            new_file = FileSystemNode(file_name, is_directory=False, content=content)
            parent.add_child(new_file)
    
    def cd(self, path: str) -> None:
        """
        Change current directory.
        
        Time Complexity: O(d) where d is depth of path
        Space Complexity: O(1)
        """
        node = self._resolve_path(path)
        if not node.is_directory:
            raise ValueError(f"'{path}' is not a directory")
        self.current_directory = node
    
    def ls(self, path: str = ".", long_format: bool = False) -> List[str]:
        """
        List directory contents.
        
        Time Complexity: O(n) where n is number of children
        Space Complexity: O(n)
        """
        node = self._resolve_path(path)
        
        if not node.is_directory:
            # If it's a file, return just the file name
            return [node.name]
        
        if not long_format:
            return sorted(node.children.keys())
        else:
            # Long format with details
            details = []
            for name, child in sorted(node.children.items()):
                type_indicator = "d" if child.is_directory else "f"
                details.append(f"{type_indicator} {child.size:8d} {name}")
            return details
    
    def pwd(self) -> str:
        """
        Print working directory.
        
        Time Complexity: O(d) where d is depth
        Space Complexity: O(d)
        """
        return self.current_directory.get_path()
    
    def cat(self, path: str) -> str:
        """
        Read file content.
        
        Time Complexity: O(d + n) where d is depth, n is content size
        Space Complexity: O(n)
        """
        node = self._resolve_path(path)
        if node.is_directory:
            raise ValueError(f"'{path}' is a directory")
        return node.content
    
    def rm(self, path: str, recursive: bool = False) -> None:
        """
        Remove file or directory.
        
        Time Complexity: O(d) where d is depth
        Space Complexity: O(1)
        """
        # Resolve parent directory
        if "/" in path:
            parent_path = "/".join(path.split("/")[:-1]) or "/"
            name = path.split("/")[-1]
            parent = self._resolve_path(parent_path)
        else:
            parent = self.current_directory
            name = path
        
        if name not in parent.children:
            raise ValueError(f"'{path}' not found")
        
        node = parent.children[name]
        
        if node.is_directory and not recursive:
            if node.children:  # Directory not empty
                raise ValueError(f"Directory '{path}' is not empty. Use recursive flag.")
        
        parent.remove_child(name)
    
    def find(self, name: str, path: str = "/") -> List[str]:
        """
        Find files/directories by name using breadth-first search.
        
        Time Complexity: O(n) where n is total nodes
        Space Complexity: O(w) where w is maximum width
        """
        start_node = self._resolve_path(path)
        results = []
        
        # BFS traversal
        queue = deque([start_node])
        
        while queue:
            node = queue.popleft()
            
            if node.name == name:
                results.append(node.get_path())
            
            if node.is_directory:
                queue.extend(node.children.values())
        
        return results
    
    def du(self, path: str = ".") -> int:
        """
        Calculate disk usage.
        
        Time Complexity: O(n) where n is nodes in subtree
        Space Complexity: O(d) where d is depth (recursion stack)
        """
        node = self._resolve_path(path)
        return node.size
    
    def tree(self, path: str = ".", max_depth: int = None) -> str:
        """
        Display directory tree structure.
        
        Time Complexity: O(n) where n is nodes in subtree
        Space Complexity: O(d) where d is depth
        """
        node = self._resolve_path(path)
        
        def _tree_recursive(node: FileSystemNode, prefix: str = "", 
                          is_last: bool = True, depth: int = 0) -> List[str]:
            if max_depth is not None and depth > max_depth:
                return []
            
            lines = []
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{node.name}")
            
            if node.is_directory and (max_depth is None or depth < max_depth):
                children = list(node.children.values())
                for i, child in enumerate(children):
                    is_last_child = (i == len(children) - 1)
                    child_prefix = prefix + ("    " if is_last else "│   ")
                    lines.extend(_tree_recursive(child, child_prefix, is_last_child, depth + 1))
            
            return lines
        
        return "\n".join(_tree_recursive(node, "", True, 0))


def demonstrate_file_system():
    """Demonstrate file system simulator functionality."""
    print("=== File System Simulator Demo ===\n")
    
    # Create file system
    fs = FileSystem()
    print(f"Initial directory: {fs.pwd()}")
    
    # Test directory operations
    print("\n1. Directory Operations:")
    fs.mkdir("documents")
    fs.mkdir("pictures")
    fs.mkdir("documents/projects")
    print(f"  Created directories: documents, pictures, documents/projects")
    
    # Test file operations
    print("\n2. File Operations:")
    fs.touch("documents/readme.txt", "This is a readme file.")
    fs.touch("documents/projects/todo.txt", "1. Complete project\n2. Write documentation")
    fs.touch("pictures/vacation.jpg", "Binary content would be here")
    print(f"  Created files: readme.txt, todo.txt, vacation.jpg")
    
    # Test navigation
    print("\n3. Navigation:")
    print(f"  Current directory: {fs.pwd()}")
    fs.cd("documents")
    print(f"  Changed to: {fs.pwd()}")
    
    # Test listing
    print("\n4. Directory Listing:")
    print("  Contents of current directory:")
    for item in fs.ls(long_format=True):
        print(f"    {item}")
    
    # Test file reading
    print("\n5. File Reading:")
    content = fs.cat("readme.txt")
    print(f"  Content of readme.txt: {content}")
    
    # Test tree display
    print("\n6. Tree Structure:")
    fs.cd("/")  # Go back to root
    tree_output = fs.tree(max_depth=3)
    print(tree_output)
    
    # Test find operation
    print("\n7. Find Operation:")
    results = fs.find("todo.txt")
    print(f"  Found 'todo.txt' at: {results}")
    
    # Test disk usage
    print("\n8. Disk Usage:")
    usage = fs.du("documents")
    print(f"  Disk usage of 'documents': {usage} bytes")
    
    # Test removal
    print("\n9. Removal Operations:")
    fs.rm("pictures/vacation.jpg")
    print(f"  Removed vacation.jpg")
    fs.rm("pictures")  # Should work since it's now empty
    print(f"  Removed pictures directory")
    
    print("\nFinal tree structure:")
    fs.cd("/")
    print(fs.tree())


def performance_comparison():
    """Compare performance of different file system operations."""
    import time
    
    print("\n=== Performance Comparison ===\n")
    
    fs = FileSystem()
    
    # Test directory creation performance
    print("1. Directory Creation Performance:")
    start_time = time.time()
    for i in range(1000):
        fs.mkdir(f"dir_{i}")
    mkdir_time = time.time() - start_time
    print(f"   Time to create 1000 directories: {mkdir_time:.6f} seconds")
    print(f"   Average time per directory: {mkdir_time/1000:.8f} seconds")
    
    # Test file creation performance
    print("\n2. File Creation Performance:")
    start_time = time.time()
    for i in range(1000):
        fs.touch(f"dir_{i}/file_{i}.txt", f"Content of file {i}")
    touch_time = time.time() - start_time
    print(f"   Time to create 1000 files: {touch_time:.6f} seconds")
    print(f"   Average time per file: {touch_time/1000:.8f} seconds")
    
    # Test file reading performance
    print("\n3. File Reading Performance:")
    start_time = time.time()
    for i in range(0, 1000, 10):  # Read every 10th file
        content = fs.cat(f"dir_{i}/file_{i}.txt")
    read_time = time.time() - start_time
    read_count = 100
    print(f"   Time to read {read_count} files: {read_time:.6f} seconds")
    print(f"   Average time per read: {read_time/read_count:.8f} seconds")
    
    # Test find performance
    print("\n4. Find Operation Performance:")
    start_time = time.time()
    results = fs.find("file_500.txt")
    find_time = time.time() - start_time
    print(f"   Time to find file_500.txt: {find_time:.6f} seconds")
    print(f"   Found {len(results)} matches")
    
    # Test tree performance
    print("\n5. Tree Display Performance:")
    start_time = time.time()
    tree_output = fs.tree(max_depth=2)
    tree_time = time.time() - start_time
    print(f"   Time to generate tree: {tree_time:.6f} seconds")
    print(f"   Tree lines generated: {len(tree_output.split(chr(10)))}")


if __name__ == "__main__":
    demonstrate_file_system()
    performance_comparison()
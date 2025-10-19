# 📚 Library Management System

A comprehensive library management system that allows librarians to manage books, members, and borrowing transactions. This project demonstrates object-oriented programming, file I/O, data persistence, and complex system design.

## 🎯 Features

- **Book Management**: Add, remove, and search books
- **Member Management**: Register and remove library members
- **Borrowing System**: Borrow and return books with due dates
- **Fine Management**: Calculate and collect overdue fines
- **Search Functionality**: Search books by title, author, or genre
- **Data Persistence**: Save and load data using JSON
- **Transaction Tracking**: Maintain borrowing history

## 🚀 How to Use

1. Run the system: `python main.py`
2. Use the menu to navigate different functions:
   - Add new books to the library
   - Register new members
   - Borrow and return books
   - Search for books
   - View library data
   - Manage fines

## 🧠 Learning Concepts

This project demonstrates:
- **Object-Oriented Programming** with multiple classes
- **Data Persistence** using JSON file I/O
- **Exception Handling** for robust operation
- **Data Validation** for user inputs
- **Date and Time Management** for due dates
- **Complex System Design** with multiple interacting components
- **Menu-Driven Interface** for user interaction

## 📁 Project Structure

```
library_management/
├── main.py          # Main application logic
├── library_data.json # Data storage file (auto-generated)
└── README.md        # This file
```

## 🎮 Sample Workflow

```
=== Library Management System ===
1. Add Book
2. Remove Book
3. Register Member
4. Remove Member
5. Borrow Book
6. Return Book
7. Search Books
8. Display All Books
9. Display All Members
10. Display Member's Books
11. Pay Fines
12. Exit

Enter your choice (1-12): 1
Enter ISBN: 978-0134685991
Enter title: Effective Java
Enter author: Joshua Bloch
Enter genre: Programming
Enter number of copies (default 1): 3
Added new book: 'Effective Java' by Joshua Bloch
Library data saved successfully!
```

## 🛠 Requirements

- Python 3.x
- No external dependencies

## 🏃‍♂️ Running the System

```bash
python main.py
```

## 🎯 Educational Value

This project is perfect for intermediate learners to practice:
1. **Class Design** with multiple related classes
2. **Data Modeling** for real-world entities
3. **File I/O Operations** with JSON serialization
4. **Error Handling** with try-except blocks
5. **Date/Time Calculations** for due dates and fines
6. **User Interface Design** with menu systems
7. **System Architecture** with modular components

## 🤔 System Components

### Book Class
- Manages book information (ISBN, title, author, genre)
- Tracks total and available copies
- Handles book-related operations

### Member Class
- Stores member information (ID, name, contact details)
- Tracks borrowed books and outstanding fines
- Manages member-specific data

### Transaction Class
- Records borrowing and returning activities
- Stores timestamps and due dates
- Maintains transaction history

### Library Class
- Central system controller
- Manages all books, members, and transactions
- Handles data persistence and business logic

## 📚 Key Concepts Covered

- **Object-Oriented Design**: Classes, objects, inheritance concepts
- **Data Structures**: Dictionaries, lists for efficient data management
- **File Operations**: JSON serialization for data persistence
- **Exception Handling**: Robust error management
- **Date/Time Operations**: Due date calculations and comparisons
- **User Interface**: Menu-driven console interface
- **Business Logic**: Library rules and workflows
- **Data Validation**: Input sanitization and verification

## 🔧 Advanced Features

- **Data Persistence**: Automatic saving/loading using JSON
- **Search Functionality**: Multi-criteria book searching
- **Fine Management**: Automated fine calculation and collection
- **Borrowing Limits**: Enforces maximum books per member
- **Duplicate Prevention**: Prevents duplicate books/members
- **Overdue Tracking**: Monitors and calculates late fees
- **Transaction History**: Maintains complete borrowing records

## 📊 System Workflow

1. **Initialization**: Load existing data or create new system
2. **User Interaction**: Menu-driven interface for all operations
3. **Data Management**: Add/remove books and members
4. **Borrowing Process**: Check availability, update records
5. **Return Process**: Calculate fines, update availability
6. **Search Operations**: Query books by various criteria
7. **Data Display**: Show books, members, and borrowing status
8. **Persistence**: Save all changes to JSON file

## 🎨 Design Patterns Used

- **Singleton Pattern**: Single Library instance manages all data
- **Factory Pattern**: Class methods for object creation from data
- **Observer Pattern**: Data changes trigger automatic saves
- **Command Pattern**: Menu options as discrete commands

## 📈 Learning Outcomes

After completing this project, you'll understand:
- How to design complex class hierarchies
- Implementing data persistence strategies
- Managing relationships between different entities
- Creating robust user interfaces
- Handling real-world business rules
- Working with dates, times, and calculations
- Building scalable system architectures

---

**Happy coding and library managing!** 📚